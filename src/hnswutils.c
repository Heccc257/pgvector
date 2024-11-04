#include "postgres.h"

#include <math.h>
#include "pq_dist.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "utils/elog.h"
#include "access/generic_xlog.h"
#include "catalog/pg_type.h"
#include "catalog/pg_type_d.h"
#include "fmgr.h"
#include "hnsw.h"
#include "lib/pairingheap.h"
#include "sparsevec.h"
#include "storage/bufmgr.h"
#include "utils/datum.h"
#include "utils/memdebug.h"
#include "utils/rel.h"
#include "utils/elog.h"
#include "pq_dist.h"
#include <immintrin.h>
#if PG_VERSION_NUM >= 130000
#include "common/hashfn.h"
#else
#include "utils/hashutils.h"
#endif

#if PG_VERSION_NUM < 170000
static inline uint64
murmurhash64(uint64 data)
{
	uint64 h = data;

	h ^= h >> 33;
	h *= 0xff51afd7ed558ccd;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53;
	h ^= h >> 33;

	return h;
}
#endif

/* TID hash table */
static uint32
hash_tid(ItemPointerData tid)
{
	union
	{
		uint64 i;
		ItemPointerData tid;
	} x;

	/* Initialize unused bytes */
	x.i = 0;
	x.tid = tid;

	return murmurhash64(x.i);
}

#define SH_PREFIX tidhash
#define SH_ELEMENT_TYPE TidHashEntry
#define SH_KEY_TYPE ItemPointerData
#define SH_KEY tid
#define SH_HASH_KEY(tb, key) hash_tid(key)
#define SH_EQUAL(tb, a, b) ItemPointerEquals(&a, &b)
#define SH_SCOPE extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Pointer hash table */
static uint32
hash_pointer(uintptr_t ptr)
{
#if SIZEOF_VOID_P == 8
	return murmurhash64((uint64)ptr);
#else
	return murmurhash32((uint32)ptr);
#endif
}

#define SH_PREFIX pointerhash
#define SH_ELEMENT_TYPE PointerHashEntry
#define SH_KEY_TYPE uintptr_t
#define SH_KEY ptr
#define SH_HASH_KEY(tb, key) hash_pointer(key)
#define SH_EQUAL(tb, a, b) (a == b)
#define SH_SCOPE extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Offset hash table */
static uint32
hash_offset(Size offset)
{
#if SIZEOF_SIZE_T == 8
	return murmurhash64((uint64)offset);
#else
	return murmurhash32((uint32)offset);
#endif
}

#define SH_PREFIX offsethash
#define SH_ELEMENT_TYPE OffsetHashEntry
#define SH_KEY_TYPE Size
#define SH_KEY offset
#define SH_HASH_KEY(tb, key) hash_offset(key)
#define SH_EQUAL(tb, a, b) (a == b)
#define SH_SCOPE extern
#define SH_DEFINE
#include "lib/simplehash.h"

typedef union
{
	pointerhash_hash *pointers;
	offsethash_hash *offsets;
	tidhash_hash *tids;
} visited_hash;

/*
 * Get the max number of connections in an upper layer for each element in the index
 */
int HnswGetM(Relation index)
{
	HnswOptions *opts = (HnswOptions *)index->rd_options;

	if (opts)
		return opts->m;

	return HNSW_DEFAULT_M;
}
int HnswGetUsePQ(Relation index)
{
	if (index == NULL)
		return 0;
	HnswOptions *opts = (HnswOptions *)index->rd_options;

	if (opts)
		return opts->use_pq;

	return HNSW_DEFAULT_USE_PQ;
}
int HnswGetPqM(Relation index)
{
	HnswOptions *opts = (HnswOptions *)index->rd_options;

	if (opts)
		return opts->pq_m;
	return HNSW_DEFAULT_PQ_M;
}
int HnswGetNbits(Relation index)
{
	HnswOptions *opts = (HnswOptions *)index->rd_options;

	if (opts)
		return opts->nbits;
	return HNSW_DEFAULT_NBITS;
}
const char *HnswGetPQDistFileName(Relation index)
{
	HnswOptions *opts = (HnswOptions *)index->rd_options;

	if (opts)
	{
		FILE *file = fopen("/root/pqfile.pth", "r");
		if (file == NULL)
		{
			elog(INFO, "pq config file not exist\n");
		}
		fseek(file, 0, SEEK_END);
		size_t length = ftell(file);
		fseek(file, 0, SEEK_SET);
		char *buffer = (char *)palloc(length + 1);
		fread(buffer, 1, length, file);
		buffer[length] = '\0';
		opts->pq_dist_file_name = buffer;
		return opts->pq_dist_file_name;
	}
	return NULL;
}

/*
 * Get the size of the dynamic candidate list in the index
 */
int HnswGetEfConstruction(Relation index)
{
	HnswOptions *opts = (HnswOptions *)index->rd_options;

	if (opts)
		return opts->efConstruction;

	return HNSW_DEFAULT_EF_CONSTRUCTION;
}
PQDist *HnswGetPQDist(Relation index)
{
	if (index == NULL)
		return NULL;
	HnswOptions *opts = (HnswOptions *)index->rd_options;

	if (opts)
	{
		return opts->pqdist;
	}
	return NULL;
}
void HnswSetPQDist(Relation index, const char *pq_dist_file_name)
{
	HnswOptions *opts = (HnswOptions *)index->rd_options;

	if (opts)
	{
		opts->pqdist = (PQDist *)palloc(sizeof(PQDist));
		PQDist_load(opts->pqdist, pq_dist_file_name);
	}
}

/*
 * Get proc
 */
FmgrInfo *
HnswOptionalProcInfo(Relation index, uint16 procnum)
{
	if (!OidIsValid(index_getprocid(index, 1, procnum)))
		return NULL;

	return index_getprocinfo(index, 1, procnum);
}

/*
 * Normalize value
 */
Datum HnswNormValue(const HnswTypeInfo *typeInfo, Oid collation, Datum value)
{
	return DirectFunctionCall1Coll(typeInfo->normalize, collation, value);
}

/*
 * Check if non-zero norm
 */
bool HnswCheckNorm(FmgrInfo *procinfo, Oid collation, Datum value)
{
	return DatumGetFloat8(FunctionCall1Coll(procinfo, collation, value)) > 0;
}

/*
 * New buffer
 */
Buffer
HnswNewBuffer(Relation index, ForkNumber forkNum)
{
	Buffer buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	return buf;
}

/*
 * Init page
 */
void HnswInitPage(Buffer buf, Page page)
{
	PageInit(page, BufferGetPageSize(buf), sizeof(HnswPageOpaqueData));
	HnswPageGetOpaque(page)->nextblkno = InvalidBlockNumber;
	HnswPageGetOpaque(page)->page_id = HNSW_PAGE_ID;
}

/*
 * Allocate a neighbor array
 */
static HnswNeighborArray *
HnswInitNeighborArray(int lm, HnswAllocator *allocator)
{
	HnswNeighborArray *a = HnswAlloc(allocator, HNSW_NEIGHBOR_ARRAY_SIZE(lm));

	a->length = 0;
	a->closerSet = false;
	return a;
}

/*
 * Allocate neighbors
 */
void HnswInitNeighbors(char *base, HnswElement element, int m, HnswAllocator *allocator)
{
	int level = element->level;
	HnswNeighborArrayPtr *neighborList = (HnswNeighborArrayPtr *)HnswAlloc(allocator, sizeof(HnswNeighborArrayPtr) * (level + 1));

	HnswPtrStore(base, element->neighbors, neighborList);

	for (int lc = 0; lc <= level; lc++)
		HnswPtrStore(base, neighborList[lc], HnswInitNeighborArray(HnswGetLayerM(m, lc), allocator));
}

/*
 * Allocate memory from the allocator
 */
void *
HnswAlloc(HnswAllocator *allocator, Size size)
{
	if (allocator)
		return (*(allocator)->alloc)(size, (allocator)->state);

	return palloc(size);
}

/*
 * Allocate an element
 */
HnswElement
HnswInitElement(char *base, ItemPointer heaptid, int m, double ml, int maxLevel, int use_pq, HnswAllocator *allocator, PQDist *pqdist)
{

	HnswElement element = HnswAlloc(allocator, sizeof(HnswElementData));

	int level = (int)(-log(RandomDouble()) * ml);

	/* Cap level */
	if (level > maxLevel)
		level = maxLevel;

	element->heaptidsLength = 0;
	HnswAddHeapTid(element, heaptid);

	element->level = level;
	element->deleted = 0;

	HnswInitNeighbors(base, element, m, allocator);

	HnswPtrStore(base, element->value, (Pointer)NULL);
	// elog(INFO, "开始导入encode data\n");
	if (use_pq)
	{
		element->id = pqdist->tuple_id++;
		// elog(INFO, "id:%d", element->id);
		// element->encoded_data = (Encode_Data*)HnswAlloc(allocator, sizeof(Encode_Data));
		// element->encoded_data->length = pqdist->code_nums;
		// element->encoded_data->data = pqdist->codes + (heaptid->ip_posid - 1) * pqdist->code_nums / sizeof(uint8_t);
	}

	return element;
}

/*
 * Add a heap TID to an element
 */
void HnswAddHeapTid(HnswElement element, ItemPointer heaptid)
{
	element->heaptids[element->heaptidsLength++] = *heaptid;
}

/*
 * Allocate an element from block and offset numbers
 */
HnswElement
HnswInitElementFromBlock(BlockNumber blkno, OffsetNumber offno)
{
	HnswElement element = palloc(sizeof(HnswElementData));
	char *base = NULL;

	element->blkno = blkno;
	element->offno = offno;
	HnswPtrStore(base, element->neighbors, (HnswNeighborArrayPtr *)NULL);
	HnswPtrStore(base, element->value, (Pointer)NULL);
	return element;
}

/*
 * Get the metapage info
 */
void HnswGetMetaPageInfo(Relation index, int *m, HnswElement *entryPoint)
{
	Buffer buf;
	Page page;
	HnswMetaPage metap;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (unlikely(metap->magicNumber != HNSW_MAGIC_NUMBER))
		elog(ERROR, "hnsw index is not valid");

	if (m != NULL)
		*m = metap->m;

	if (entryPoint != NULL)
	{
		if (BlockNumberIsValid(metap->entryBlkno))
		{
			*entryPoint = HnswInitElementFromBlock(metap->entryBlkno, metap->entryOffno);
			(*entryPoint)->level = metap->entryLevel;
		}
		else
			*entryPoint = NULL;
	}

	UnlockReleaseBuffer(buf);
}

/*
 * Get the entry point
 */
HnswElement
HnswGetEntryPoint(Relation index)
{
	HnswElement entryPoint;

	HnswGetMetaPageInfo(index, NULL, &entryPoint);

	return entryPoint;
}

/*
 * Update the metapage info
 */
static void
HnswUpdateMetaPageInfo(Page page, int updateEntry, HnswElement entryPoint, BlockNumber insertPage)
{
	HnswMetaPage metap = HnswPageGetMeta(page);

	if (updateEntry)
	{
		if (entryPoint == NULL)
		{
			metap->entryBlkno = InvalidBlockNumber;
			metap->entryOffno = InvalidOffsetNumber;
			metap->entryLevel = -1;
		}
		else if (entryPoint->level > metap->entryLevel || updateEntry == HNSW_UPDATE_ENTRY_ALWAYS)
		{
			metap->entryBlkno = entryPoint->blkno;
			metap->entryOffno = entryPoint->offno;
			metap->entryLevel = entryPoint->level;
		}
	}

	if (BlockNumberIsValid(insertPage))
		metap->insertPage = insertPage;
}

/*
 * Update the metapage
 */
void HnswUpdateMetaPage(Relation index, int updateEntry, HnswElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building)
{
	Buffer buf;
	Page page;
	GenericXLogState *state;

	buf = ReadBufferExtended(index, forkNum, HNSW_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
	{
		state = NULL;
		page = BufferGetPage(buf);
	}
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	HnswUpdateMetaPageInfo(page, updateEntry, entryPoint, insertPage);

	if (building)
		MarkBufferDirty(buf);
	else
		GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
}

/*
 * Set element tuple, except for neighbor info
 */
void HnswSetElementTuple(char *base, HnswElementTuple etup, HnswElement element)
{
	Pointer valuePtr = HnswPtrAccess(base, element->value);

	etup->type = HNSW_ELEMENT_TUPLE_TYPE;
	etup->level = element->level;
	etup->deleted = 0;
	etup->id = element->id;
	for (int i = 0; i < HNSW_HEAPTIDS; i++)
	{
		if (i < element->heaptidsLength)
			etup->heaptids[i] = element->heaptids[i];
		else
			ItemPointerSetInvalid(&etup->heaptids[i]);
	}
	memcpy(&etup->data, valuePtr, VARSIZE_ANY(valuePtr));
}

/*
 * Set neighbor tuple
 */
void HnswSetNeighborTuple(char *base, HnswNeighborTuple ntup, HnswElement e, int m, int use_pq, PQDist *pqdist)
{

	int idx = 0;

	ntup->type = HNSW_NEIGHBOR_TUPLE_TYPE;

	for (int lc = e->level; lc >= 0; lc--)
	{
		HnswNeighborArray *neighbors = HnswGetNeighbors(base, e, lc);
		int lm = HnswGetLayerM(m, lc);

		for (int i = 0; i < lm; i++)
		{
			ItemPointer indextid = &ntup->indextids[idx++];

			if (i < neighbors->length)
			{
				HnswCandidate *hc = &neighbors->items[i];
				HnswElement hce = HnswPtrAccess(base, hc->element);

				ItemPointerSet(indextid, hce->blkno, hce->offno);
			}
			else
				ItemPointerSetInvalid(indextid);
		}
	}
	ntup->count = idx;
	if (use_pq)
	{
		uint8_t *encode_data = get_centroids_id(pqdist, e->id);
		void *pq_start = (void *)(ntup->indextids + idx);
		void *pq_store;

		memcpy(pq_start, encode_data, pqdist->code_nums);

		idx = 0;
		int lc = 0;

		HnswNeighborArray *neighbors = HnswGetNeighbors(base, e, lc);
		int lm = HnswGetLayerM(m, lc);

		for (int i = 0; i < lm; i++)
		{
			pq_store = pq_start + pqdist->code_nums * (idx + 1);
			idx += 1;
			if (i < neighbors->length)
			{

				HnswCandidate *hc = &neighbors->items[i];
				HnswElement hce = HnswPtrAccess(base, hc->element);
				encode_data = get_centroids_id(pqdist, hce->id);
				memcpy(pq_store, encode_data, pqdist->code_nums);
			}
			else
			{
				memset(pq_store, 0, pqdist->code_nums);
			}
		}
	}
}

/*
 * Load neighbors from page
 */
static void
LoadNeighborsFromPage(HnswElement element, Relation index, Page page, int m, int use_pq, int pq_m)
{
	elog(INFO, "load neighbors from page");
	elog(INFO, "element->level:%d", element->level);
	char *base = NULL;

	HnswNeighborTuple ntup = (HnswNeighborTuple)PageGetItem(page, PageGetItemId(page, element->neighborOffno));
	int neighborCount = (element->level + 2) * m;

	Assert(HnswIsNeighborTuple(ntup));

	HnswInitNeighbors(base, element, m, NULL);

	/* Ensure expected neighbors */
	if (ntup->count != neighborCount)
		return;

	void *pq_store = (void *)(ntup->indextids + ntup->count) + pq_m;


	for (int i = 0; i < neighborCount; i++)
	{
		HnswElement e;
		int level;
		HnswCandidate *hc;
		ItemPointer indextid;
		HnswNeighborArray *neighbors;

		indextid = &ntup->indextids[i];
		level = element->level - i / m;
		if (level < 0)
			level = 0;

		if (!ItemPointerIsValid(indextid))
		{
			if(use_pq && level == 0)
				pq_store += pq_m;
			continue;
		}

		e = HnswInitElementFromBlock(ItemPointerGetBlockNumber(indextid), ItemPointerGetOffsetNumber(indextid));

		/* Calculate level based on offset */

		neighbors = HnswGetNeighbors(base, element, level);
		hc = &neighbors->items[neighbors->length++];
		if (use_pq && level == 0)
		{
			Encode_Data *encode_data = (Encode_Data *)palloc(offsetof(Encode_Data, data) + pq_m);
			encode_data->length = pq_m;
			elog(INFO, "encode_data->length:%d", pq_m);
			memcpy(encode_data->data, pq_store, pq_m);
			e->encode_data = encode_data;
		}
		HnswPtrStore(base, hc->element, e);
	}
}

/*
 * Load neighbors
 */
void HnswLoadNeighbors(HnswElement element, Relation index, int m)
{
	elog(INFO, "load neighbors");
	elog(INFO, "element->neighborPage:%d", element->neighborPage);
	elog(INFO, "element->lc:%d", element->level);
	Buffer buf;
	Page page;

	buf = ReadBuffer(index, element->neighborPage);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	int use_pq = HnswGetUsePQ(index);
	int pq_m = HnswGetPqM(index);

	LoadNeighborsFromPage(element, index, page, m, use_pq, pq_m);

	UnlockReleaseBuffer(buf);
}

/*
 * Load an element from a tuple
 */
void HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec)
{
	element->level = etup->level;
	element->deleted = etup->deleted;
	element->id = etup->id;
	element->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
	element->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	element->heaptidsLength = 0;

	if (loadHeaptids)
	{
		for (int i = 0; i < HNSW_HEAPTIDS; i++)
		{
			/* Can stop at first invalid */
			if (!ItemPointerIsValid(&etup->heaptids[i]))
				break;

			HnswAddHeapTid(element, &etup->heaptids[i]);
		}
	}

	if (loadVec)
	{
		char *base = NULL;
		Datum value = datumCopy(PointerGetDatum(&etup->data), false, -1);

		HnswPtrStore(base, element->value, DatumGetPointer(value));
	}
}

/*
 * Load an element and optionally get its distance from q
 */
void HnswLoadElement(HnswElement element, float *distance, Datum *q, Relation index, FmgrInfo *procinfo, Oid collation, bool loadVec, float *maxDistance, int use_pq, PQDist *pqdist)
{

	Buffer buf;
	Page page;
	HnswElementTuple etup;

	/* Read vector */
	buf = ReadBuffer(index, element->blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	etup = (HnswElementTuple)PageGetItem(page, PageGetItemId(page, element->offno));

	Assert(HnswIsElementTuple(etup));

	/* Calculate distance */
	if (distance != NULL)
	{
		if (DatumGetPointer(*q) == NULL)
			*distance = 0;
		else if (!use_pq)
			*distance = (float)DatumGetFloat8(FunctionCall2Coll(procinfo, collation, *q, PointerGetDatum(&etup->data)));
		else
			*distance = (float)calc_dist_pq_loaded_simd(pqdist, etup->id);
	}

	/* Load element */
	if (distance == NULL || maxDistance == NULL || *distance < *maxDistance)
		HnswLoadElementFromTuple(element, etup, true, loadVec);

	UnlockReleaseBuffer(buf);
}

/*
 * Get the distance for a candidate
 */
static float
GetCandidateDistance(char *base, HnswCandidate *hc, Datum q, FmgrInfo *procinfo, Oid collation, bool use_pq, PQDist *pqdist)
{

	HnswElement hce = HnswPtrAccess(base, hc->element);
	Datum value = HnswGetValue(base, hce);
	if (!use_pq)
	{

		return DatumGetFloat8(FunctionCall2Coll(procinfo, collation, q, value));
	}
	assert(pqdist);
	float distance = 0;
	distance = calc_dist_pq_loaded_simd(pqdist, hce->id);
	return distance;
}
/* static float
GetCandidateDistancePQ(char *base, HnswCandidate * hc, Datum q, FmgrInfo *procinfo, Oid collation, PQDist* pqdist)
{
	HnswElement hce = HnswPtrAccess(base, hc->element);
	Datum		value = HnswGetValue(base, hce);
	float distance = 0;
	assert(pqdist);

	Encode_Data* encode_data = hce->encoded_data;
	distance = calc_dist_pq_loaded_simd(encode_data->data, hce->heaptids[0].ip_posid - 1);

	return distance;
} */
/*
 * Create a candidate for the entry point
 */
HnswCandidate *
HnswEntryCandidate(char *base, HnswElement entryPoint, Datum q, Relation index, FmgrInfo *procinfo, Oid collation, bool loadVec, int use_pq, PQDist *pqdist)
{

	HnswCandidate *hc = palloc(sizeof(HnswCandidate));

	HnswPtrStore(base, hc->element, entryPoint);
	if (index == NULL)
		hc->distance = GetCandidateDistance(base, hc, q, procinfo, collation, use_pq, pqdist);
	else
		HnswLoadElement(entryPoint, &hc->distance, &q, index, procinfo, collation, loadVec, NULL, use_pq, pqdist);
	return hc;
}

/*
 * Compare candidate distances
 */
static int
CompareNearestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (((const HnswPairingHeapNode *)a)->inner->distance < ((const HnswPairingHeapNode *)b)->inner->distance)
		return 1;

	if (((const HnswPairingHeapNode *)a)->inner->distance > ((const HnswPairingHeapNode *)b)->inner->distance)
		return -1;

	return 0;
}

/*
 * Compare candidate distances
 */
static int
CompareFurthestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (((const HnswPairingHeapNode *)a)->inner->distance < ((const HnswPairingHeapNode *)b)->inner->distance)
		return -1;

	if (((const HnswPairingHeapNode *)a)->inner->distance > ((const HnswPairingHeapNode *)b)->inner->distance)
		return 1;

	return 0;
}

/*
 * Create a pairing heap node for a candidate
 */
static HnswPairingHeapNode *
CreatePairingHeapNode(HnswCandidate *c)
{
	HnswPairingHeapNode *node = palloc(sizeof(HnswPairingHeapNode));

	node->inner = c;
	return node;
}

/*
 * Init visited
 */
static inline void
InitVisited(char *base, visited_hash *v, Relation index, int ef, int m)
{
	if (index != NULL)
		v->tids = tidhash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else if (base != NULL)
		v->offsets = offsethash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else
		v->pointers = pointerhash_create(CurrentMemoryContext, ef * m * 2, NULL);
}

/*
 * Add to visited
 */
static inline void
AddToVisited(char *base, visited_hash *v, HnswCandidate *hc, Relation index, bool *found)
{
	if (index != NULL)
	{
		HnswElement element = HnswPtrAccess(base, hc->element);
		ItemPointerData indextid;

		ItemPointerSet(&indextid, element->blkno, element->offno);
		tidhash_insert(v->tids, indextid, found);
	}
	else if (base != NULL)
	{
#if PG_VERSION_NUM >= 130000
		HnswElement element = HnswPtrAccess(base, hc->element);

		offsethash_insert_hash(v->offsets, HnswPtrOffset(hc->element), element->hash, found);
#else
		offsethash_insert(v->offsets, HnswPtrOffset(hc->element), found);
#endif
	}
	else
	{
#if PG_VERSION_NUM >= 130000
		HnswElement element = HnswPtrAccess(base, hc->element);

		pointerhash_insert_hash(v->pointers, (uintptr_t)HnswPtrPointer(hc->element), element->hash, found);
#else
		pointerhash_insert(v->pointers, (uintptr_t)HnswPtrPointer(hc->element), found);
#endif
	}
}

/*
 * Count element towards ef
 */
static inline bool
CountElement(char *base, HnswElement skipElement, HnswCandidate *hc)
{
	HnswElement e;

	if (skipElement == NULL)
		return true;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	e = HnswPtrAccess(base, hc->element);
	return e->heaptidsLength != 0;
}

/*
 * Algorithm 2 from paper
 */
List *
HnswSearchLayer(char *base, Datum q, List *ep, int ef, int lc, Relation index, FmgrInfo *procinfo, Oid collation, int m, bool inserting, HnswElement skipElement, int use_pq, PQDist *pqdist, bool is_search_knn)
{

	// 不在最底层不使用pq
	if (lc != 0 || !is_search_knn)
	{
		use_pq = 0;
	}

	List *w = NIL;
	pairingheap *C = pairingheap_allocate(CompareNearestCandidates, NULL);
	pairingheap *W = pairingheap_allocate(CompareFurthestCandidates, NULL);
	int wlen = 0;
	visited_hash v;
	ListCell *lc2;
	HnswNeighborArray *neighborhoodData = NULL;
	Size neighborhoodSize = 0;

	InitVisited(base, &v, index, ef, m);

	/* Create local memory for neighborhood if needed */
	if (index == NULL)
	{
		neighborhoodSize = HNSW_NEIGHBOR_ARRAY_SIZE(HnswGetLayerM(m, lc));
		neighborhoodData = palloc(neighborhoodSize);
	}

	/* Add entry points to v, C, and W */
	foreach (lc2, ep)
	{
		HnswCandidate *hc = (HnswCandidate *)lfirst(lc2);
		bool found;

		AddToVisited(base, &v, hc, index, &found);

		pairingheap_add(C, &(CreatePairingHeapNode(hc)->ph_node));
		pairingheap_add(W, &(CreatePairingHeapNode(hc)->ph_node));

		/*
		 * Do not count elements being deleted towards ef when vacuuming. It
		 * would be ideal to do this for inserts as well, but this could
		 * affect insert performance.
		 */
		if (CountElement(base, skipElement, hc))
			wlen++;
	}

	while (!pairingheap_is_empty(C))
	{
		HnswNeighborArray *neighborhood;
		HnswCandidate *c = ((HnswPairingHeapNode *)pairingheap_remove_first(C))->inner;
		HnswCandidate *f = ((HnswPairingHeapNode *)pairingheap_first(W))->inner;
		HnswElement cElement;

		if (c->distance > f->distance)
			break;

		cElement = HnswPtrAccess(base, c->element);

		if (HnswPtrIsNull(base, cElement->neighbors))
			HnswLoadNeighbors(cElement, index, m);

		/* Get the neighborhood at layer lc */
		neighborhood = HnswGetNeighbors(base, cElement, lc);

		/* Copy neighborhood to local memory if needed */
		if (index == NULL)
		{
			LWLockAcquire(&cElement->lock, LW_SHARED);
			memcpy(neighborhoodData, neighborhood, neighborhoodSize);
			LWLockRelease(&cElement->lock);
			neighborhood = neighborhoodData;
		}

		for (int i = 0; i < neighborhood->length; i++)
		{
			HnswCandidate *e = &neighborhood->items[i];
			bool visited;

			AddToVisited(base, &v, e, index, &visited);

			if (!visited)
			{
				float eDistance;
				HnswElement eElement = HnswPtrAccess(base, e->element);
				bool alwaysAdd = wlen < ef;

				f = ((HnswPairingHeapNode *)pairingheap_first(W))->inner;
				if (lc > 0)
				{
					if (index == NULL)
						eDistance = GetCandidateDistance(base, e, q, procinfo, collation, use_pq, pqdist);
					else
						HnswLoadElement(eElement, &eDistance, &q, index, procinfo, collation, inserting, alwaysAdd ? NULL : &f->distance, use_pq, pqdist);
				}
				else
				{

					if (index == NULL)
						eDistance = GetCandidateDistance(base, e, q, procinfo, collation, use_pq, pqdist);
					else
					{
						elog(INFO, "yyyyy");
						eDistance = calc_dist_pq_loaded_by_id(pqdist, eElement->encode_data->data + pqdist->code_nums * i);
						elog(INFO, "distance:%f", eDistance);
					}
				}

				if (eDistance < f->distance || alwaysAdd)
				{
					HnswCandidate *ec;

					Assert(!eElement->deleted);

					/* Make robust to issues */
					if (eElement->level < lc)
						continue;

					/* Copy e */
					ec = palloc(sizeof(HnswCandidate));
					HnswPtrStore(base, ec->element, eElement);
					ec->distance = eDistance;

					pairingheap_add(C, &(CreatePairingHeapNode(ec)->ph_node));
					pairingheap_add(W, &(CreatePairingHeapNode(ec)->ph_node));

					/*
					 * Do not count elements being deleted towards ef when
					 * vacuuming. It would be ideal to do this for inserts as
					 * well, but this could affect insert performance.
					 */
					if (CountElement(base, skipElement, e))
					{
						wlen++;

						/* No need to decrement wlen */
						if (wlen > ef)
							pairingheap_remove_first(W);
					}
				}
			}
		}
	}

	if (lc == 0 && use_pq)
	{
		List *realDistanceCandidates = NIL;

		while (!pairingheap_is_empty(W))
		{
			HnswCandidate *hc = ((HnswPairingHeapNode *)pairingheap_remove_first(W))->inner;
			realDistanceCandidates = lappend(realDistanceCandidates, hc);
		}

		foreach (lc2, realDistanceCandidates)
		{

			HnswCandidate *hc = (HnswCandidate *)lfirst(lc2);
			float realDistance;
			if (index == NULL)
				realDistance = GetCandidateDistance(base, hc, q, procinfo, collation, 0, NULL);
			else
				HnswLoadElement(HnswPtrAccess(base, hc->element), &realDistance, &q, index, procinfo, collation, inserting, NULL, 0, NULL);
			hc->distance = realDistance;

			pairingheap_add(W, &(CreatePairingHeapNode(hc)->ph_node));
		}
	}

	/* Add each element of W to w */
	while (!pairingheap_is_empty(W))
	{
		HnswCandidate *hc = ((HnswPairingHeapNode *)pairingheap_remove_first(W))->inner;

		w = lappend(w, hc);
	}

	return w;
}

/*
 * Compare candidate distances with pointer tie-breaker
 */
static int
#if PG_VERSION_NUM >= 130000
CompareCandidateDistances(const ListCell *a, const ListCell *b)
{
	HnswCandidate *hca = lfirst(a);
	HnswCandidate *hcb = lfirst(b);
#else
CompareCandidateDistances(const void *a, const void *b)
{
	HnswCandidate *hca = lfirst(*(ListCell **)a);
	HnswCandidate *hcb = lfirst(*(ListCell **)b);
#endif

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (HnswPtrPointer(hca->element) < HnswPtrPointer(hcb->element))
		return 1;

	if (HnswPtrPointer(hca->element) > HnswPtrPointer(hcb->element))
		return -1;

	return 0;
}

/*
 * Compare candidate distances with offset tie-breaker
 */
static int
#if PG_VERSION_NUM >= 130000
CompareCandidateDistancesOffset(const ListCell *a, const ListCell *b)
{
	HnswCandidate *hca = lfirst(a);
	HnswCandidate *hcb = lfirst(b);
#else
CompareCandidateDistancesOffset(const void *a, const void *b)
{
	HnswCandidate *hca = lfirst(*(ListCell **)a);
	HnswCandidate *hcb = lfirst(*(ListCell **)b);
#endif

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (HnswPtrOffset(hca->element) < HnswPtrOffset(hcb->element))
		return 1;

	if (HnswPtrOffset(hca->element) > HnswPtrOffset(hcb->element))
		return -1;

	return 0;
}

/*
 * Calculate the distance between elements
 */
static float
HnswGetDistance(char *base, HnswElement a, HnswElement b, FmgrInfo *procinfo, Oid collation)
{

	Datum aValue = HnswGetValue(base, a);
	Datum bValue = HnswGetValue(base, b);

	return DatumGetFloat8(FunctionCall2Coll(procinfo, collation, aValue, bValue));
}

/*
 * Check if an element is closer to q than any element from R
 */
static bool
CheckElementCloser(char *base, HnswCandidate *e, List *r, FmgrInfo *procinfo, Oid collation)
{
	HnswElement eElement = HnswPtrAccess(base, e->element);
	ListCell *lc2;

	foreach (lc2, r)
	{
		HnswCandidate *ri = lfirst(lc2);
		HnswElement riElement = HnswPtrAccess(base, ri->element);
		float distance = HnswGetDistance(base, eElement, riElement, procinfo, collation);

		if (distance <= e->distance)
			return false;
	}

	return true;
}

/*
 * Algorithm 4 from paper
 */
static List *
SelectNeighbors(char *base, List *c, int lm, int lc, FmgrInfo *procinfo, Oid collation, HnswElement e2, HnswCandidate *newCandidate, HnswCandidate **pruned, bool sortCandidates)
{
	List *r = NIL;
	List *w = list_copy(c);
	HnswCandidate **wd;
	int wdlen = 0;
	int wdoff = 0;
	HnswNeighborArray *neighbors = HnswGetNeighbors(base, e2, lc);
	bool mustCalculate = !neighbors->closerSet;
	List *added = NIL;
	bool removedAny = false;

	if (list_length(w) <= lm)
		return w;

	wd = palloc(sizeof(HnswCandidate *) * list_length(w));

	/* Ensure order of candidates is deterministic for closer caching */
	if (sortCandidates)
	{
		if (base == NULL)
			list_sort(w, CompareCandidateDistances);
		else
			list_sort(w, CompareCandidateDistancesOffset);
	}

	while (list_length(w) > 0 && list_length(r) < lm)
	{
		/* Assumes w is already ordered desc */
		HnswCandidate *e = llast(w);

		w = list_delete_last(w);

		/* Use previous state of r and wd to skip work when possible */
		if (mustCalculate)
			e->closer = CheckElementCloser(base, e, r, procinfo, collation);
		else if (list_length(added) > 0)
		{
			/* Keep Valgrind happy for in-memory, parallel builds */
			if (base != NULL)
				VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

			/*
			 * If the current candidate was closer, we only need to compare it
			 * with the other candidates that we have added.
			 */
			if (e->closer)
			{
				e->closer = CheckElementCloser(base, e, added, procinfo, collation);

				if (!e->closer)
					removedAny = true;
			}
			else
			{
				/*
				 * If we have removed any candidates from closer, a candidate
				 * that was not closer earlier might now be.
				 */
				if (removedAny)
				{
					e->closer = CheckElementCloser(base, e, r, procinfo, collation);
					if (e->closer)
						added = lappend(added, e);
				}
			}
		}
		else if (e == newCandidate)
		{
			e->closer = CheckElementCloser(base, e, r, procinfo, collation);
			if (e->closer)
				added = lappend(added, e);
		}

		/* Keep Valgrind happy for in-memory, parallel builds */
		if (base != NULL)
			VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

		if (e->closer)
			r = lappend(r, e);
		else
			wd[wdlen++] = e;
	}

	/* Cached value can only be used in future if sorted deterministically */
	neighbors->closerSet = sortCandidates;

	/* Keep pruned connections */
	while (wdoff < wdlen && list_length(r) < lm)
		r = lappend(r, wd[wdoff++]);

	/* Return pruned for update connections */
	if (pruned != NULL)
	{
		if (wdoff < wdlen)
			*pruned = wd[wdoff];
		else
			*pruned = linitial(w);
	}

	return r;
}

/*
 * Add connections
 */
static void
AddConnections(char *base, HnswElement element, List *neighbors, int lc)
{
	ListCell *lc2;
	HnswNeighborArray *a = HnswGetNeighbors(base, element, lc);

	foreach (lc2, neighbors)
		a->items[a->length++] = *((HnswCandidate *)lfirst(lc2));
}

/*
 * Update connections
 */
void HnswUpdateConnection(char *base, HnswElement element, HnswCandidate *hc, int lm, int lc, int *updateIdx, Relation index, FmgrInfo *procinfo, Oid collation)
{
	HnswElement hce = HnswPtrAccess(base, hc->element);
	HnswNeighborArray *currentNeighbors = HnswGetNeighbors(base, hce, lc);
	HnswCandidate hc2;
	// bool use_pq = HnswGetUsePQ(index);
	// PQDist* pqdist = HnswGetPQDist(index);
	HnswPtrStore(base, hc2.element, element);
	hc2.distance = hc->distance;

	if (currentNeighbors->length < lm)
	{
		currentNeighbors->items[currentNeighbors->length++] = hc2;

		/* Track update */
		if (updateIdx != NULL)
			*updateIdx = -2;
	}
	else
	{
		/* Shrink connections */
		HnswCandidate *pruned = NULL;

		/* Load elements on insert */
		if (index != NULL)
		{
			Datum q = HnswGetValue(base, hce);

			for (int i = 0; i < currentNeighbors->length; i++)
			{
				HnswCandidate *hc3 = &currentNeighbors->items[i];
				HnswElement hc3Element = HnswPtrAccess(base, hc3->element);

				if (HnswPtrIsNull(base, hc3Element->value))
					HnswLoadElement(hc3Element, &hc3->distance, &q, index, procinfo, collation, true, NULL, 0, NULL);
				else
					hc3->distance = GetCandidateDistance(base, hc3, q, procinfo, collation, 0, NULL);

				/* Prune element if being deleted */
				if (hc3Element->heaptidsLength == 0)
				{
					pruned = &currentNeighbors->items[i];
					break;
				}
			}
		}

		if (pruned == NULL)
		{
			List *c = NIL;

			/* Add candidates */
			for (int i = 0; i < currentNeighbors->length; i++)
				c = lappend(c, &currentNeighbors->items[i]);
			c = lappend(c, &hc2);

			SelectNeighbors(base, c, lm, lc, procinfo, collation, hce, &hc2, &pruned, true);

			/* Should not happen */
			if (pruned == NULL)
				return;
		}

		/* Find and replace the pruned element */
		for (int i = 0; i < currentNeighbors->length; i++)
		{
			if (HnswPtrEqual(base, currentNeighbors->items[i].element, pruned->element))
			{
				currentNeighbors->items[i] = hc2;

				/* Track update */
				if (updateIdx != NULL)
					*updateIdx = i;

				break;
			}
		}
	}
}

/*
 * Remove elements being deleted or skipped
 */
static List *
RemoveElements(char *base, List *w, HnswElement skipElement)
{
	ListCell *lc2;
	List *w2 = NIL;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	foreach (lc2, w)
	{
		HnswCandidate *hc = (HnswCandidate *)lfirst(lc2);
		HnswElement hce = HnswPtrAccess(base, hc->element);

		/* Skip self for vacuuming update */
		if (skipElement != NULL && hce->blkno == skipElement->blkno && hce->offno == skipElement->offno)
			continue;

		if (hce->heaptidsLength != 0)
			w2 = lappend(w2, hc);
	}

	return w2;
}

#if PG_VERSION_NUM >= 130000
/*
 * Precompute hash
 */
static void
PrecomputeHash(char *base, HnswElement element)
{
	HnswElementPtr ptr;

	HnswPtrStore(base, ptr, element);

	if (base == NULL)
		element->hash = hash_pointer((uintptr_t)HnswPtrPointer(ptr));
	else
		element->hash = hash_offset(HnswPtrOffset(ptr));
}
#endif

/*
 * Algorithm 1 from paper
 */
void HnswFindElementNeighbors(char *base, HnswElement element, HnswElement entryPoint, Relation index, FmgrInfo *procinfo, Oid collation, int m, int efConstruction, int use_pq, PQDist *pqdist, bool existing)
{

	Datum value = HnswGetValue(base, element);

	/* const float* query = (const float *) DatumGetPointer(value) + 2;

	if(use_pq)
	load_query_data_and_cache(pqdist, query); */

	List *ep;
	List *w;

	int level = element->level;

	int entryLevel;
	Datum q = HnswGetValue(base, element);

	HnswElement skipElement = existing ? element : NULL;

#if PG_VERSION_NUM >= 130000
	/* Precompute hash */
	if (index == NULL)
		PrecomputeHash(base, element);
#endif

	/* No neighbors if no entry point */
	if (entryPoint == NULL)
	{
		return;
	}

	/* Get entry point and level */
	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, procinfo, collation, true, use_pq, pqdist));

	entryLevel = entryPoint->level;

	/* 1st phase: greedy search to insert level */
	for (int lc = entryLevel; lc >= level + 1; lc--)
	{
		w = HnswSearchLayer(base, q, ep, 1, lc, index, procinfo, collation, m, true, skipElement, use_pq, pqdist, false);
		ep = w;
	}

	if (level > entryLevel)
		level = entryLevel;

	/* Add one for existing element */
	if (existing)
		efConstruction++;

	/* 2nd phase */
	for (int lc = level; lc >= 0; lc--)
	{
		int lm = HnswGetLayerM(m, lc);
		List *neighbors;
		List *lw;

		w = HnswSearchLayer(base, q, ep, efConstruction, lc, index, procinfo, collation, m, true, skipElement, use_pq, pqdist, false);

		/* Elements being deleted or skipped can help with search */
		/* but should be removed before selecting neighbors */
		if (index != NULL)
			lw = RemoveElements(base, w, skipElement);
		else
			lw = w;

		/*
		 * Candidates are sorted, but not deterministically. Could set
		 * sortCandidates to true for in-memory builds to enable closer
		 * caching, but there does not seem to be a difference in performance.
		 */
		neighbors = SelectNeighbors(base, lw, lm, lc, procinfo, collation, element, NULL, NULL, false);

		AddConnections(base, element, neighbors, lc);

		ep = w;
	}
}

PGDLLEXPORT Datum l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum halfvec_l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum sparsevec_l2_normalize(PG_FUNCTION_ARGS);

static void
SparsevecCheckValue(Pointer v)
{
	SparseVector *vec = (SparseVector *)v;

	if (vec->nnz > HNSW_MAX_NNZ)
		elog(ERROR, "sparsevec cannot have more than %d non-zero elements for hnsw index", HNSW_MAX_NNZ);
}

/*
 * Get type info
 */
const HnswTypeInfo *
HnswGetTypeInfo(Relation index)
{
	FmgrInfo *procinfo = HnswOptionalProcInfo(index, HNSW_TYPE_INFO_PROC);

	if (procinfo == NULL)
	{
		static const HnswTypeInfo typeInfo = {
			.maxDimensions = HNSW_MAX_DIM,
			.normalize = l2_normalize,
			.checkValue = NULL};

		return (&typeInfo);
	}
	else
		return (const HnswTypeInfo *)DatumGetPointer(FunctionCall0Coll(procinfo, InvalidOid));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_halfvec_support);
Datum hnsw_halfvec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 2,
		.normalize = halfvec_l2_normalize,
		.checkValue = NULL};

	PG_RETURN_POINTER(&typeInfo);
};

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_bit_support);
Datum hnsw_bit_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 32,
		.normalize = NULL,
		.checkValue = NULL};

	PG_RETURN_POINTER(&typeInfo);
};

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_sparsevec_support);
Datum hnsw_sparsevec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = SPARSEVEC_MAX_DIM,
		.normalize = sparsevec_l2_normalize,
		.checkValue = SparsevecCheckValue};

	PG_RETURN_POINTER(&typeInfo);
};

void pqdist_init(PQDist *pqdist, int _d, int _m, int _nbits)
{
	pqdist->d = _d;
	pqdist->m = _m;
	pqdist->nbits = _nbits;
	pqdist->code_nums = 1 << _nbits;
	pqdist->d_pq = _d / _m;
	pqdist->codes = (uint8_t *)malloc(sizeof(uint8_t) * _m * _nbits);
	pqdist->centroids = (float *)malloc(sizeof(float) * _m * _d);
	pqdist->pq_dist_cache_data = (float *)malloc(sizeof(float) * _m * pqdist->code_nums);
}

void PQDist_load(PQDist *pq, const char *filename)
{
	FILE *fin = fopen(filename, "rb");
	if (fin == NULL)
	{
		// printf("open %s fail\n", filename);
		elog(ERROR, "open %s fail", filename);
		exit(-1);
	}

	int N;
	fread(&N, sizeof(int), 1, fin);
	fread(&pq->d, sizeof(int), 1, fin);
	fread(&pq->m, sizeof(int), 1, fin);
	fread(&pq->nbits, sizeof(int), 1, fin);
	elog(INFO, "load: %d %d %d %d", N, pq->d, pq->m, pq->nbits);
	assert(8 % pq->nbits == 0);
	pq->code_nums = 1 << pq->nbits;
	pq->tuple_id = 0;
	pq->d_pq = pq->d / pq->m;
	pq->table_size = pq->m * pq->code_nums;

	// if (pq->pq_dist_cache_data != NULL) {
	//     free(pq->pq_dist_cache_data);
	// }
	// pq->pq_dist_cache_data = (float*)aligned_alloc(64, sizeof(float) * pq->table_size);
	pq->pq_dist_cache_data = (float *)palloc(sizeof(float) * pq->table_size);
	pq->qdata = (float *)palloc(sizeof(float) * pq->d);
	// elog(INFO, "table_size: %d\n", pq->table_size);
	size_t codes_size = N / 8 * pq->m * pq->nbits;
	pq->codes = (uint8_t *)palloc(codes_size);
	if (pq->codes == NULL)
	{
		// printf("Memory allocation failed for codes\n");
		elog(ERROR, "Memory allocation failed for codes");
		exit(-1);
	}
	fread(pq->codes, sizeof(uint8_t), codes_size, fin);

	// elog(INFO, "codes_size: %d\n", codes_size);
	pq->centroids = (float *)palloc(sizeof(float) * pq->code_nums * pq->d);
	if (pq->centroids == NULL)
	{
		elog(ERROR, "Memory allocation failed for centroids");
		exit(-1);
	}
	fread(pq->centroids, sizeof(float), pq->code_nums * pq->d, fin);
	fclose(fin);
}

void PQDist_free(PQDist *pq)
{
	if (pq->codes != NULL)
	{
		free(pq->codes);
	}
	if (pq->centroids != NULL)
	{
		free(pq->centroids);
	}
	if (pq->pq_dist_cache_data != NULL)
	{
		free(pq->pq_dist_cache_data);
	}
}
uint8_t *get_centroids_id(PQDist *pq, int id)
{

	const uint8_t *code = pq->codes + id * (pq->m * pq->nbits / 8);
	uint8_t *centroids_id = (uint8_t *)palloc(pq->m * sizeof(uint8_t));
	memset(centroids_id, 0, pq->m);

	if (pq->nbits == 8)
	{
		size_t num_ids = pq->m;
		size_t num_bytes = num_ids;

		size_t i = 0;
		size_t j = 0;

		for (; i + 32 <= num_bytes; i += 32)
		{
			__m256i input = _mm256_loadu_si256((__m256i const *)(code + i));
			_mm256_storeu_si256((__m256i *)(centroids_id + i), input);
		}
		for (; i < num_bytes; i++)
			centroids_id[i] = code[i];
	}
	else
	{
		size_t num_ids = pq->m;
		size_t num_bytes = (num_ids + 1) / 2;

		size_t i = 0;
		size_t j = 0;

		for (; i + 32 <= num_bytes; i += 32, j += 64)
		{
			__m256i input = _mm256_loadu_si256((__m256i const *)(code + i));

			__m256i low_mask = _mm256_set1_epi8(0x0F);
			__m256i low = _mm256_and_si256(input, low_mask);

			__m256i high = _mm256_srli_epi16(input, 4);
			high = _mm256_and_si256(high, low_mask);

			__m256i interleave_lo = _mm256_unpacklo_epi8(low, high);
			__m256i interleave_hi = _mm256_unpackhi_epi8(low, high);

			__m128i seg0 = _mm256_extracti128_si256(interleave_lo, 0);
			__m128i seg1 = _mm256_extracti128_si256(interleave_hi, 0);
			__m128i seg2 = _mm256_extracti128_si256(interleave_lo, 1);
			__m128i seg3 = _mm256_extracti128_si256(interleave_hi, 1);

			_mm_storeu_si128((__m128i *)(centroids_id + j), seg0);
			_mm_storeu_si128((__m128i *)(centroids_id + j + 16), seg1);
			_mm_storeu_si128((__m128i *)(centroids_id + j + 32), seg2);
			_mm_storeu_si128((__m128i *)(centroids_id + j + 48), seg3);
		}

		for (; i < num_bytes; ++i, j += 2)
		{
			centroids_id[j] = code[i] & 0x0F;
			centroids_id[j + 1] = (code[i] >> 4) & 0x0F;
		}
	}
	return centroids_id;
}
float *get_centroid_data(PQDist *pqdist, int quantizer, int code_id)
{
	return pqdist->centroids + (quantizer * pqdist->code_nums + code_id) * pqdist->d_pq;
}
float calc_dist(int d, float *vec1, float *vec2)
{
	float distance = 0;
	for (int i = 0; i < d; i++)
	{
		distance += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
	}
	return distance;
}
void clear_pq_dist_cache(PQDist *pqdist)
{
	memset(pqdist->pq_dist_cache_data, 0, (pqdist->m * pqdist->code_nums) * sizeof(float));
}

void load_query_data_and_cache(PQDist *pqdist, const float *_qdata)
{

	memcpy(pqdist->qdata, _qdata, sizeof(float) * pqdist->d);

	clear_pq_dist_cache(pqdist);

	pqdist->use_cache = true;

	for (int i = 0; i < pqdist->m * pqdist->code_nums; i++)
	{
		pqdist->pq_dist_cache_data[i] = calc_dist(pqdist->d_pq, get_centroid_data(pqdist, i / pqdist->code_nums, i % pqdist->code_nums), pqdist->qdata + (i / pqdist->code_nums) * pqdist->d_pq);
	}

	__builtin_prefetch(pqdist->pq_dist_cache_data, 0, 3);
	size_t prefetch_size = 128;
	for (int i = 0; i < pqdist->table_size * 4; i += prefetch_size / 4)
	{
		__builtin_prefetch(pqdist->pq_dist_cache_data + i, 0, 3);
	}
}
float calc_dist_pq_simd(PQDist *pqdist, int data_id, float *qdata, bool use_cache)
{
	float dist = 0;
	uint8_t *ids = get_centroids_id(pqdist, data_id);
	__m256 simd_dist = _mm256_setzero_ps();
	int q;
	const int stride = 8;

	for (q = 0; q <= pqdist->m - stride; q += stride)
	{
		__m128i id_vec_128 = _mm_loadl_epi64((__m128i *)(ids + q));
		__m256i id_vec = _mm256_cvtepu8_epi32(id_vec_128);

		__m256i offset_vec = _mm256_setr_epi32(0 * pqdist->code_nums, 1 * pqdist->code_nums, 2 * pqdist->code_nums, 3 * pqdist->code_nums,
											   4 * pqdist->code_nums, 5 * pqdist->code_nums, 6 * pqdist->code_nums, 7 * pqdist->code_nums);

		id_vec = _mm256_add_epi32(id_vec, offset_vec);
		__m256 dist_vec = _mm256_i32gather_ps(pqdist->pq_dist_cache_data + q * pqdist->code_nums, id_vec, 4);
		simd_dist = _mm256_add_ps(simd_dist, dist_vec);
	}

	simd_dist = _mm256_hadd_ps(simd_dist, simd_dist);
	simd_dist = _mm256_hadd_ps(simd_dist, simd_dist);

	float dist_array[8];
	_mm256_storeu_ps(dist_array, simd_dist);
	dist += dist_array[0] + dist_array[4];

	for (; q < pqdist->m; q++)
	{
		dist += pqdist->pq_dist_cache_data[q * pqdist->code_nums + ids[q]];
	}

	return dist;
}
float calc_dist_pq_loaded_simd(PQDist *pqdist, int data_id)
{
	// elog(INFO, "data_id: %d", data_id);
	float distance = calc_dist_pq_simd(pqdist, data_id, pqdist->qdata, pqdist->use_cache);

	return distance;
}

float calc_dist_pq_loaded_by_id(PQDist *pqdist, uint8_t *ids)
{
	float distance = calc_dist_pq_by_id(pqdist, ids, pqdist->qdata, pqdist->use_cache);

	return distance;
}

float calc_dist_pq_by_id(PQDist *pqdist, uint8_t *ids, float *qdata, bool use_cache)
{
	float dist = 0;
	__m256 simd_dist = _mm256_setzero_ps();
	int q;
	const int stride = 8;

	for (q = 0; q <= pqdist->m - stride; q += stride)
	{
		__m128i id_vec_128 = _mm_loadl_epi64((__m128i *)(ids + q));
		__m256i id_vec = _mm256_cvtepu8_epi32(id_vec_128);

		__m256i offset_vec = _mm256_setr_epi32(0 * pqdist->code_nums, 1 * pqdist->code_nums, 2 * pqdist->code_nums, 3 * pqdist->code_nums,
											   4 * pqdist->code_nums, 5 * pqdist->code_nums, 6 * pqdist->code_nums, 7 * pqdist->code_nums);

		id_vec = _mm256_add_epi32(id_vec, offset_vec);
		__m256 dist_vec = _mm256_i32gather_ps(pqdist->pq_dist_cache_data + q * pqdist->code_nums, id_vec, 4);
		simd_dist = _mm256_add_ps(simd_dist, dist_vec);
	}

	simd_dist = _mm256_hadd_ps(simd_dist, simd_dist);
	simd_dist = _mm256_hadd_ps(simd_dist, simd_dist);

	float dist_array[8];
	_mm256_storeu_ps(dist_array, simd_dist);
	dist += dist_array[0] + dist_array[4];

	for (; q < pqdist->m; q++)
	{
		dist += pqdist->pq_dist_cache_data[q * pqdist->code_nums + ids[q]];
	}

	return dist;
}
