---
layout:     post
title:      数据库之Microsoft SQL Server
subtitle:   
date:       2019-05-16
author:     JD
header-img: img/post-jd-pymssql.jpg
catalog: true
tags:
    - pymssql
    - database
---


## 数据库基本信息

Microsoft SQL Server是Microsoft公司推出的关系型数据库管理系统，可以与Windows NT完全集成，利用了NT的许多功能。并且具有良好的伸缩性，目前在用`SQL Server 2008`作为我们项目的数据库。

在项目的驱动下，需要用python脚本来操作数据库。包括了基本的增删改查，也包括调用存储过程等。方法有很多，鉴于MS有专门提供python接口，我这边就使用pymssql。

## pymssql

pymssql有官方文档，[请点我跳转](http://pymssql.org/en/latest/)。

下面列举一些对MS的操作

先定义好数据库的基本参数

    configs = {
        'server': '127.0.0.1',
        'user': 'sa',
        'password': 'test',
        'database': 'testdb'
    }

连接、关闭数据库

	import pymssql

    conn = pymssql.connect(**configs)
    conn.close()

创建、关闭游标

    cursor = conn.cursor()
    cursor.close()

提交、回滚

    conn.commit()
    conn.rollback()

增删改查，只需要执行sql语句，增删改一定要记得提交

    insert_sql = "insert into table_name values ('', '');"
    #delete_sql = "delete from table_name;"
    #update_sql = "update table_name set column_name='';"
    #select_sql = "select * from table_name;"

    cursor.execute(insert_sql)
    conn.commit()

重复执行数据库操作

    cursor.executemany()

    cursor.executemany(
        "INSERT INTO persons VALUES (%d, %s, %s)",
        [(1, 'John Smith', 'John Doe'),
         (2, 'Jane Doe', 'Joe Dog'),
         (3, 'Mike T.', 'Sarah H.')])

获取查询返回值

    select_data = cursor.fetchall()  #返回全部结果
    select_data = cursor.fetchone()  #返回单个值

调取存储过程

    cursor.callproc('proc_name', ('arg', ))

获取存储过程的返回值，必须先`nextset`，然后再`fetchone`

    cursor.nextset()
    return_info = cursor.fetchone()

## 完整代码

增删改操作

    def execute_mssql(insert_sql):
        try:
            conn = pymssql.connect(**configs)
            cursor = conn.cursor()
            cursor.execute(insert_sql)
            conn.commit()
        except:
            conn.rollback()
        finally:
            conn.close()

查询操作

    def select_mssql(select_sql):
        try:
            conn = pymssql.connect(**configs)
            cursor = conn.cursor()
            cursor.execute(select_sql)
            #select_data = cursor.fetchone()[0]
            select_data = cursor.fetchall()
        except:
            pass
        finally:
            conn.close()
        return select_data

查询返回DataFrame格式

    def select_mssql(select_sql):
        try:
            conn = pymssql.connect(**configs)
            select_data = read_sql(select_sql, con=conn)
        except:
            pass
        finally:
            conn.close()
        return select_data

调用存储过程，并获取返回值

    def callmsproc(*args):
        try:
            conn = pymssql.connect(**configs)
            cursor = conn.cursor()
            cursor.callproc(args[0], args[1:])
            cursor.nextset()
            return_info = cursor.fetchone()[0]
            conn.commit()
        except:
            conn.rollback()
        finally:
            conn.close()
        return return_info