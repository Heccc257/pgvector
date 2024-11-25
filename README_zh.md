# 编译步骤


- 启动pgsql。
- 需要在pgxs文件(·pg_config --pgxs·)中的CPPFLAGS中加入-mavx2选项用来编译simd指令
```bash
git clone https://github.com/Heccc257/pgvector
cd pgvector
make && make install
psql -U postgres
CREATE EXTENSION vector;
```

# python脚本使用说明

`pip install -r requirements.txt`
在pgvector目录下新建config.json文件，示例:

```json
{ 
    "d": 960, 
    "data_path": "/share/ann_benchmarks/gist/train.tsv",
    "max_elements": 10000, 

    "opq": 1, 
    "opq_matrix_file": "/root/python_gist/opq_matrix_100000_240_4_100000_100",
    "sample_size": 20,
    "M": 120,
    "k": 16,
    "max_iter": 2,

    "pq_m_list": [120, 240, 320, 480], 
    "nbits_list": [4, 8], 
    "output_dir": "/root/python_gist/", 
    
}
```
之后运行`python3 run.py`，这会调用opq.py来计算opq旋转矩阵（如果启用opq）以及construct.py来生成pq_dist_file辅助文件并且运行test.py测试程序.

各个参数的说明如下：

- 公共参数：

  - d: 数据维度
  - data_path: 数据文件的路径，要求为tsv格式
  - max_element: 选取数据文件中前max_element个数据点

- 与opq相关:

  - opq: 是否启用opq
  - sample_size: opq中sample多少个数据点
  - M: opq中一个划分为M个子空间
  - k: opq中一个子空间KMeans聚类的簇心数
  - max_iter: opq迭代次数
  - opq_matrix_file: opq矩阵写入以及后续读入的路径

- 与construct相关：

  - pq_m_list, nbits_list: 对于pq_m_list*nbits_list中的每一种(pq_m, nbits)组合生成pq_dist_file文件，文件里写入了簇心数据。
  - output_dir: 输出的文件夹名

opq.py, construct.py也均可单独运行，运行`python3 xxx.py --config_file config.json`即可


**请注意，当前pgvector在gist数据集上测试只支持pq_m = 120（120个子空间），nbits = 4, 8（每个子空间16或156个簇心），后续可以支持(240, 4)和(320, 4)**

# 测试以及使用说明

使用的方法非常简单，只需要在创建索引的时候在WITH里指定use_PQ, pq_m, nbist参数。   
示例：`CREATE INDEX hnswpq_idx ON test_vectors USING hnsw (vec vector_l2_ops) WITH (use_PQ=1,pq_m=120,nbits=4);`

**另外，请新建路径为`/root/pqfile.pth`的文件里面写明pq_dist_file辅助文件的路径**