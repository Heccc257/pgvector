#include "postgres.h"

#include "access/relscan.h"
#include "hnsw.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"
/*
 * Algorithm 5 from paper
 */
static List *
GetScanItems(IndexScanDesc scan, Datum q)
{

	const float *query = (const float *)DatumGetPointer(q) + 2;

	HnswScanOpaque so = (HnswScanOpaque)scan->opaque;
	Relation index = scan->indexRelation;
	FmgrInfo *procinfo = so->procinfo;
	Oid collation = so->collation;
	List *ep;
	List *w;
	int m;
	HnswElement entryPoint;
	char *base = NULL;

	/* Get m and entry point */
	HnswGetMetaPageInfo(index, &m, &entryPoint);

	if (entryPoint == NULL)
		return NIL;

	int use_pq = HnswGetUsePQ(index);
	PQDist* pqdist;
	if (use_pq)
	{
		pqdist = HnswGetPQDist(index);
		load_query_data_and_cache(pqdist, query);
	}
	int pq_m = HnswGetPqM(index);
	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, procinfo, collation, false, 0, NULL));

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		w = HnswSearchLayer(base, q, ep, 1, lc, index, procinfo, collation, m, false, NULL, use_pq, pqdist, true);
		ep = w;
	}
	return HnswSearchLayer(base, q, ep, hnsw_ef_search, 0, index, procinfo, collation, m, false, NULL, use_pq, pqdist, true);
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{

	HnswScanOpaque so = (HnswScanOpaque)scan->opaque;
	Datum value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
		value = PointerGetDatum(NULL);
	else
	{
		value = scan->orderByData->sk_argument;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->normprocinfo != NULL)
			value = HnswNormValue(so->typeInfo, so->collation, value);
	}

	return value;
}

/*
 * Prepare for an index scan
 */
IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	int use_pq = HnswGetUsePQ(index);

	PQDist *pqdist = (PQDist *)palloc(sizeof(PQDist));
	if (use_pq)
	{
		const char *pq_dist_file_name = HnswGetPQDistFileName(index);
		// PQDist_load(pqdist, pq_dist_file_name);
		HnswSetPQDist(index, pq_dist_file_name);
	}

	IndexScanDesc scan;
	HnswScanOpaque so;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (HnswScanOpaque)palloc(sizeof(HnswScanOpaqueData));
	so->typeInfo = HnswGetTypeInfo(index);
	so->first = true;
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Hnsw scan temporary context",
									   ALLOCSET_DEFAULT_SIZES);

	/* Set support functions */
	so->procinfo = index_getprocinfo(index, 1, HNSW_DISTANCE_PROC);
	so->normprocinfo = HnswOptionalProcInfo(index, HNSW_NORM_PROC);
	so->collation = index->rd_indcollation[0];

	scan->opaque = so;

	return scan;
}

/*
 * Start or restart an index scan
 */
void hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque)scan->opaque;

	so->first = true;
	MemoryContextReset(so->tmpCtx);

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
bool hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{

	HnswScanOpaque so = (HnswScanOpaque)scan->opaque;
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan hnsw index without order");

		/* Requires MVCC-compliant snapshot as not able to maintain a pin */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with hnsw");

		/* Get scan value */
		value = GetScanValue(scan);

		/*
		 * Get a shared lock. This allows vacuum to ensure no in-flight scans
		 * before marking tuples as deleted.
		 */
		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->w = GetScanItems(scan, value);

		/* Release shared lock */
		UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->first = false;

#if defined(HNSW_MEMORY) && PG_VERSION_NUM >= 130000
		elog(INFO, "memory: %zu MB", MemoryContextMemAllocated(so->tmpCtx, false) / (1024 * 1024));
#endif
	}

	while (list_length(so->w) > 0)
	{
		char *base = NULL;
		HnswCandidate *hc = llast(so->w);
		HnswElement element = HnswPtrAccess(base, hc->element);
		ItemPointer heaptid;

		/* Move to next element if no valid heap TIDs */
		if (element->heaptidsLength == 0)
		{
			so->w = list_delete_last(so->w);
			continue;
		}

		heaptid = &element->heaptids[--element->heaptidsLength];

		MemoryContextSwitchTo(oldCtx);

		scan->xs_heaptid = *heaptid;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	return false;
}

/*
 * End a scan and release resources
 */
void hnswendscan(IndexScanDesc scan)
{

	HnswScanOpaque so = (HnswScanOpaque)scan->opaque;

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}
