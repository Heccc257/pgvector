编译步骤
* 安装pgsql
    * 使用apt安装pqsql，安装完成之后应该能直接运行`pg_config`命令。
    * 实验室服务器不支持`systemctl`,直接运行`sudo /etc/init.d/postgresql start
`启动pgsql。
* 进入pg_vector项目目录，`make && make install`
* `CREATE EXTENSION vector;`
* `psql -U postgres`登录pgsql数据库
* `\i script.sql`可以执行sql脚本

* 需要在pgxs文件(pg_config --pgxs)中加入-mavx2选项
* pqdist对象存储在hnsw.h/buildstate中，在hnswbuild.c/initbuildstate中初始化
* hnswelementdata类里增加了存储自己和邻居节点encode_data的结构，具体见hnsw.h 140行
* 在hnswbuild.c/inserttuple里传入自己的encode_data
* 一个问题：index里的pq_dist_file_name不知道为什么是一个无效指针，需要在initbuildstate里手动指明。