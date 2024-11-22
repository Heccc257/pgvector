SET maintenance_work_mem = '20GB';
DROP TABLE IF EXISTS test_vectors;
CREATE TABLE test_vectors (
    id serial PRIMARY KEY,
    vec vector(960)
);

COPY test_vectors (vec) FROM '/root/gist/train_10000.csv' WITH (FORMAT csv);
DROP TABLE IF EXISTS query;
CREATE TABLE query(
    id serial PRIMARY KEY,
    vec vector(960)
);

COPY query (vec) FROM '/root/gist/test.csv' WITH (FORMAT csv);


CREATE INDEX hnswpq_idx ON test_vectors USING hnsw (vec vector_l2_ops) WITH (use_PQ=1,pq_m=120,nbits=4);


SET enable_seqscan = off;
SELECT id, vec <-> (SELECT vec FROM query WHERE id = 1) AS distance
FROM test_vectors
ORDER BY vec <-> (SELECT vec FROM query WHERE id = 1)
LIMIT 10;  

DROP INDEX IF EXISTS hnswpq_idx;


DROP TABLE IF EXISTS test_vectors;
DROP TABLE IF EXISTS query;
