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

num_limit = 10000
need_to_construct = True
datas = read_fvecs("/root/gist/train.fvecs", num_limit)
queries = read_fvecs("/root/gist/test.fvecs")
neighbors = read_gnd(f"/root/gist/gnd_{num_limit}.gt")

conn = psycopg2.connect(dbname='pgvector_python_test')
conn.autocommit = True

cur = conn.cursor()
# cur.execute("SET max_parallel_workers = 2;")
cur.execute("SET maintenance_work_mem = '20GB';")
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

if need_to_construct:
    cur.execute('DROP TABLE IF EXISTS psycopg2_items')
    dimension = 960
    sql = f'''
        CREATE TABLE psycopg2_items (
            id bigserial PRIMARY KEY, 
            embedding vector({dimension})
        )
    '''

            # half_embedding halfvec({dimension})
    cur.execute(sql)

    register_vector(cur, globally=False)
    for i, data in enumerate(datas):
        # cur.execute('INSERT INTO psycopg2_items (embedding) VALUES (%s), (NULL)', (data,)) 
        cur.execute('INSERT INTO psycopg2_items (embedding) VALUES (%s)', (data,)) 
        if ((i+1) % 100==0):
            print(f"{i+1} vectors inserted over!")


sql = '''SET enable_seqscan=OFF'''
cur.execute(sql)
sql = f'''CREATE INDEX IF NOT EXISTS test_vectors_hnswpq_idx ON psycopg2_items USING hnsw (embedding vector_l2_ops) WITH (use_PQ=1,pq_m=120,nbits=4,pq_dist_file_name="/share/ann_benchmarks/gist_hnsw/encoded_data_{num_limit}_120_4");'''
cur.execute(sql)

correct_cnt = 0
total_cnt = 0
total_time = 0

ef_list_1 = list(range(10,60,3))
ef_list_2 = list(range(60,100,10))
ef_list_3 = list(range(100,800,20))
ef_list = ef_list_1 + ef_list_2 + ef_list_3


output = []

for ef in ef_list:
    cur.execute(f'SET hnsw.ef_search = {ef};')
    for i, query in enumerate(queries):
        start_time = time.time()
        cur.execute('SELECT * FROM psycopg2_items ORDER BY embedding <-> %s LIMIT 10', (query,))
        end_time = time.time()
        total_time += end_time - start_time

        res = cur.fetchall()
        # res = [int(x[0]/2) for x in res]
        res = [x[0] for x in res]
        correct_res = [x for x in res if x in neighbors[i]]
        correct_cnt += len(correct_res)
        total_cnt += len(res)
        if ((i + 1)% 100==0):
            print(f"{i+1} vectors tested over!")

    ans = correct_cnt / total_cnt
    av_time = total_time * 1000 / total_cnt
    print(f"correct_cnt:{correct_cnt}, total_cnt:{total_cnt}, ans:{ans}, total_time:{av_time}")
    output.append((ans, av_time, ef))

with open("output.txt", "w") as f:
    for a,b,c in output:
        f.write(f"{a} {b} {c}\n")