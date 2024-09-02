编译步骤
* 安装pgsql
    * 使用apt安装pqsql，安装完成之后应该能直接运行`pg_config`命令。
    * 实验室服务器不支持`systemctl`,直接运行`sudo /etc/init.d/postgresql start
`启动pgsql。
* 进入pg_vector项目目录，`make && make install`
* `CREATE EXTENSION vector;`
* `psql -U postgres`登录pgsql数据库
* `\i script.sql`可以执行sql脚本