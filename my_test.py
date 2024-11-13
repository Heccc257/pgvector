import numpy as np
from pgvector.psycopg2 import register_vector, SparseVector
import psycopg2
import time
from psycopg2.extras import DictCursor, RealDictCursor, NamedTupleCursor

def read_fvecs(filename, limit = 1e8):
    with open(filename, 'rb') as f:
        data = f.read()

    offset = 0
    vectors = []
    cnt = 0
    while offset < len(data) and cnt < limit:
        dim = np.frombuffer(data, dtype=np.int32, count=1, offset=offset)[0]
        offset += 4 
        vector = np.frombuffer(data, dtype=np.float32, count=dim, offset=offset)
        vectors.append(vector)
        offset += dim * 4
        cnt += 1

    return list(vectors)

def read_gnd(filename, limit = 1e8):
    with open(filename, 'rb') as f:
        data = f.read()

    offset = 0
    n = np.frombuffer(data, dtype=np.int32, count=1, offset=offset)[0]
    offset += 4
    neighbor_num = np.frombuffer(data, dtype=np.int32, count=1, offset=offset)[0]
    offset += 4
    neighbors = []
    for i in range(n):
        neighbor = []
        for j in range(neighbor_num):
            tmp = np.frombuffer(data, dtype=np.int32, count=1, offset=offset)[0] + 1
            offset += 4
            neighbor.append(tmp)
        neighbors.append(neighbor)
    return list(neighbors)

num_limit = 100000
use_pq, pq_m, nbits = 1, 240, 8
need_to_construct = False
datas = read_fvecs("/root/gist/train.fvecs", num_limit)
queries = read_fvecs("/root/gist/test.fvecs")
neighbors = read_gnd(f"/root/gist_gnd/gnd_{num_limit}.gt")

conn = psycopg2.connect(dbname='test')
conn.autocommit = True

cur = conn.cursor()
# cur.execute("SET max_parallel_workers = 2;")
cur.execute("SET maintenance_work_mem = '20GB';")
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
cur.execute('ALTER EXTENSION vector UPDATE')
register_vector(cur, globally=False)
if need_to_construct:
    cur.execute(f'DROP TABLE IF EXISTS psycopg2_items_{num_limit}')
    dimension = 960
    sql = f'''
        CREATE TABLE psycopg2_items_{num_limit} (
            id bigserial PRIMARY KEY, 
            embedding vector({dimension})
        )
    '''

            # half_embedding halfvec({dimension})
    cur.execute(sql)

    for i, data in enumerate(datas):
        # cur.execute('INSERT INTO psycopg2_items (embedding) VALUES (%s), (NULL)', (data,)) 
        cur.execute(f'INSERT INTO psycopg2_items_{num_limit} (embedding) VALUES (%s)', (data,)) 
        if ((i+1) % 1000==0):
            print(f"{i+1} vectors inserted over!")


sql = '''SET enable_seqscan=OFF;'''
cur.execute(sql)
sql = f'DROP INDEX IF EXISTS hnsw_idx_{num_limit}_{use_pq}_{pq_m}_{nbits};'
cur.execute(sql)
sql = f'CREATE INDEX IF NOT EXISTS hnsw_idx_{num_limit}_{use_pq}_{pq_m}_{nbits} ON psycopg2_items_{num_limit} USING hnsw (embedding vector_l2_ops) WITH (use_PQ={use_pq},pq_m={pq_m},nbits={nbits});'

cur.execute(sql)
print('Index already exists or Creating index is Done...')


#ef_list_1 = list(range(10,60,3))
ef_list_2 = list(range(60,100,10))
ef_list_3 = list(range(100,420,20))
ef_list = ef_list_2 + ef_list_3


output = []

for ef in ef_list:
    cur.execute(f'SET hnsw.ef_search = {ef};')
    correct_cnt = 0
    total_cnt = 0
    total_time = 0
    for i, query in enumerate(queries):
        start_time = time.time()
        cur.execute(f'SELECT * FROM psycopg2_items_{num_limit} ORDER BY embedding <-> %s LIMIT 10;', (query, ))
        end_time = time.time()
        total_time += end_time - start_time
        res = cur.fetchall()
        # res = [int(x[0]/2) for x in res]
        res = [x[0] for x in res]
        correct_res = [x for x in res if x in neighbors[i]]
        correct_cnt += len(correct_res)
        total_cnt += len(res)
        if ((i + 1)% 1000==0):
            print(f"{i+1} vectors tested over!")

    ans = correct_cnt / total_cnt
    print(f"correct_cnt:{correct_cnt}, total_cnt:{total_cnt}, ans:{ans}, total_time:{total_time}, use_pq:{use_pq}, ef:{ef}, pq_m:{pq_m}, nbits:{nbits}")
    output.append((ans, total_time, ef))

with open(f"output_{num_limit}_{use_pq}_{pq_m}_{nbits}.txt", "w") as f:
    for a,b,c in output:
        f.write(f"{a} {b} {c}\n")