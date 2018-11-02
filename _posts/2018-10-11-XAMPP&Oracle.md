---
layout:     post
title:      XAMPP & Oracle
subtitle:   
date:       2018-10-11
author:     JD
header-img: img/post-jd-xampp.jpg
catalog: true
tags:
    - XAMPP
    - ORACLE
---

# Linux下安装XAMPP

## 环境及软件版本

服务器系统(cat /etc/issue)
    
    Red Hat Enterprise Linux Server release 6.5 (Santiago)

XAMPP软件
    
    xampp-linux-x64-7.2.10-0-installer.run

Oracle服务端
    
    Oracle Database 11g Enterprise Edition Release 11.2.0.4.0 - 64bit Production
    PL/SQL Release 11.2.0.4.0 - Production
    "CORE	11.2.0.4.0	Production"
    TNS for 64-bit Windows: Version 11.2.0.4.0 - Production
    NLSRTL Version 11.2.0.4.0 - Production

Oracle客户端所涉及软件

    instantclient-basic-linux.x64-11.2.0.4.0.zip
    instantclient-sqlplus-linux.x64-11.2.0.4.0.zip
    instantclient-sdk-linux.x64-11.2.0.4.0.zip

## 安装Oracle客户端

### 压缩文件解压到/usr/local/oracle

    unzip instantclient-basic-linux.x64-11.2.0.4.0.zip
    unzip instantclient-sqlplus-linux.x64-11.2.0.4.0.zip
    unzip instantclient-sdk-linux.x64-11.2.0.4.0.zip

### 新建tnsnames.ora文件
    
    cd instantclient_11_2
    mkdir -p network/admin
    cd network/admin
    vi tnsnames.ora

其中tnsnames.ora里填写

    ORCL =
      (DESCRIPTION =
        (ADDRESS = (PROTOCOL = TCP)(HOST = 172.16.248.40)(PORT = 1521))
        (CONNECT_DATA =
          (SERVER = DEDICATED)
          (SERVICE_NAME = orcl)
        )
      )

### 配置环境变量

编辑配置文件

    vi .bash_profile

其中.bash_profile里填写

    export ORACLE_HOME=/usr/local/oracle/instantclient_11_2
    export TNS_ADMIN=$ORACLE_HOME/network/admin
    export NLS_LANG=AMERICAN_AMERICA.AL32UTF8
    export LD_LIBRARY_PATH=$ORACLE_HOME
    export PATH=$ORACLE_HOME:$PATH

然后令.bash_profile生效

    source .bash_profile

### 测试连接数据库

使用sqlplus连接数据库，第一个`reporter`是用户名，第二个`reporter`是密码，ORCL就是我们配置tnsnames.ora里面的别名

    sqlplus reporter/reporter@ORCL

就可以连接了

![](http://wx3.sinaimg.cn/mw690/006CxzLigy1fwraqbu37qj30jo06lglo.jpg)

完工，进行下一步！

## XAMPP安装及部署

### 安装XAMPP

首先将`xampp-linux-x64-7.2.10-0-installer.run`放到服务器。给它755的权限

    chmod 755 xampp-linux-x64-7.2.10-0-installer.run

进行安装

    ./xampp-linux-x64-7.2.10-0-installer.run

然后弹出安装界面，进行傻瓜式操作

![](http://wx2.sinaimg.cn/mw690/006F1DTzgy1fwrb37ofc1j30hn0eqmy3.jpg)

一直next

![](http://wx3.sinaimg.cn/mw690/006F1DTzgy1fwrb61c367j30hh0erabd.jpg)

之后手动启动，所以先launch XAMPP前面的勾号去掉，点击Finish

![](http://wx3.sinaimg.cn/mw690/006F1DTzgy1fwrb7kmmpxj30hd0epdgy.jpg)

### 部署XAMPP

手动启动XAMPP

    /opt/lampp/lampp start

出现error

    [root@localhost tools]# /opt/lampp/lampp start
    /opt/lampp/bin/gettext: /lib64/libc.so.6: version `GLIBC_2.14' not found (required by /opt/lampp/bin/gettext)
    /opt/lampp/bin/gettext: /lib64/libc.so.6: version `GLIBC_2.14' not found (required by /opt/lampp/lib/libiconv.so.2)

没有找到`GLIBC_2.14`，查看服务器有没有`GLIBC_2.14`

    [root@localhost tools]# strings /lib64/libc.so.6 |grep GLIBC_ 
    GLIBC_2.2.5
    GLIBC_2.2.6
    GLIBC_2.3
    GLIBC_2.3.2
    GLIBC_2.3.3
    GLIBC_2.3.4
    GLIBC_2.4
    GLIBC_2.5
    GLIBC_2.6
    GLIBC_2.7
    GLIBC_2.8
    GLIBC_2.9
    GLIBC_2.10
    GLIBC_2.11
    GLIBC_2.12
    GLIBC_PRIVATE

发现的确没有，那就动手安装`GLIBC_2.14`

    tar -xvf glibc-2.14.tar.gz
    mkdir glibc-2.14/build
    cd glibc-2.14/build
    ../configure  --prefix=/usr --disable-profile --enable-add-ons --with-headers=/usr/include --with-binutils=/usr/bin
    make
    make install

`make`花费时间会有点长，耐心等待

以上都完成后，查看库文件，发现/lib64/libc.so.6软链接指向了2.14

    [root@localhost ~]# ll /lib64/libc.so.6
    lrwxrwxrwx. 1 root root 12 Oct 31 06:24 /lib64/libc.so.6 -> libc-2.14.so

现在我们发现服务器已经有`GLIBC_2.14`

    [root@localhost ~]# strings /lib64/libc.so.6 |grep GLIBC_
    GLIBC_2.2.5
    GLIBC_2.2.6
    GLIBC_2.3
    GLIBC_2.3.2
    GLIBC_2.3.3
    GLIBC_2.3.4
    GLIBC_2.4
    GLIBC_2.5
    GLIBC_2.6
    GLIBC_2.7
    GLIBC_2.8
    GLIBC_2.9
    GLIBC_2.10
    GLIBC_2.11
    GLIBC_2.12
    GLIBC_2.13
    GLIBC_2.14
    GLIBC_PRIVATE

继续启动XAMPP，依旧报错

    [root@localhost ~]# /opt/lampp/lampp start
    XAMPP is currently only availably as 32 bit application. Please use a 32 bit compatibility library for your system.

这时需要修改/opt/lampp/lampp，搜索`XAMPP is currently 32 bit only`，按照下面加#

    # XAMPP is currently 32 bit only
    # case `uname -m` in
    #         *_64)
    #         if $XAMPP_ROOT/bin/php -v > /dev/null 2>&1
    #         then
    #                 :
    #         else
    #                 $GETTEXT -s "XAMPP is currently only availably as 32 bit application. Please use a 32 bit compatibility library for your system."
    #                 exit 1
    #         fi
    #         ;;
    # esac

再次启动，还是错误，发现没有`GLIBC_2.17`，和上面的方法一样

    [root@localhost lampp]# /opt/lampp/lampp start
    XAMPP:  SELinux is activated. Making XAMPP fit SELinux...
    chcon: cannot access `/opt/lampp/lib/mysql/*.so': No such file or directory
    Starting XAMPP for Linux 7.2.10-0...
    XAMPP: Starting Apache...fail.
    httpd: Syntax error on line 522 of /opt/lampp/etc/httpd.conf: Syntax error on line 10 of /opt/lampp/etc/extra/httpd-xampp.conf: Cannot load modules/libphp7.so into server: /lib64/libc.so.6: version `GLIBC_2.17' not found (required by /opt/lampp/lib/libcurl.so.4)
    XAMPP: Starting MySQL...ok.
    XAMPP: Starting ProFTPD.../opt/lampp/bin/my_print_defaults: /lib64/libc.so.6: version `GLIBC_2.17' not found (required by /opt/lampp/bin/my_print_defaults)
    fail.
    Contents of "/opt/lampp/var/proftpd/start.err":
    /opt/lampp/sbin/proftpd: /lib64/libc.so.6: version `GLIBC_2.17' not found (required by /opt/lampp/lib/libmysqlclient.so.18)
    [root@localhost lampp]# /opt/lampp/bin/my_print_defaults: /lib64/libc.so.6: version `GLIBC_2.17' not found (required by /opt/lampp/bin/my_print_defaults)
    /opt/lampp/bin/my_print_defaults: /lib64/libc.so.6: version `GLIBC_2.17' not found (required by /opt/lampp/bin/my_print_defaults)
    /opt/lampp/bin/mysqld_safe_helper: /lib64/libc.so.6: version `GLIBC_2.17' not found (required by /opt/lampp/bin/mysqld_safe_helper)

安装`GLIBC_2.17`

    tar -xvf glibc-2.17.tar.gz
    cd glibc-2.17
    mkdir build
    cd build
    ../configure --prefix=/usr --disable-profile --enable-add-ons --with-headers=/usr/include --with-binutils=/usr/bin
    make && make install

再次试着启动，成功了！

    [root@localhost lampp]# /opt/lampp/lampp start
    Starting XAMPP for Linux 7.2.10-0...
    XAMPP: Starting Apache...ok.
    XAMPP: Starting MySQL...ok.
    XAMPP: Starting ProFTPD...ok.

启动后，为XAMPP设置相应密码

    [root@localhost lampp]# /opt/lampp/lampp security
    XAMPP:  Quick security check...
    XAMPP:  MySQL is accessable via network. 
    XAMPP: Normaly that's not recommended. Do you want me to turn it off? [yes]     
    XAMPP:  Turned off.
    XAMPP: Stopping MySQL...ok.
    XAMPP: Starting MySQL...ok.
    XAMPP:  The MySQL/phpMyAdmin user pma has no password set!!! 
    XAMPP: Do you want to set a password? [yes] 
    XAMPP: Password: 
    XAMPP: Password (again): 
    XAMPP:  Setting new MySQL pma password.
    XAMPP:  Setting phpMyAdmin's pma password to the new one.
    XAMPP:  MySQL has no root passwort set!!! 
    XAMPP: Do you want to set a password? [yes] 
    XAMPP:  Write the password somewhere down to make sure you won't forget it!!! 
    XAMPP: Password: 
    XAMPP: Password (again): 
    XAMPP:  Setting new MySQL root password.
    XAMPP:  Change phpMyAdmin's authentication method.
    XAMPP:  The FTP password for user 'daemon' is still set to 'xampp'. 
    XAMPP: Do you want to change the password? [yes] 
    XAMPP: Password: 
    XAMPP: Password (again): 
    XAMPP: Reload ProFTPD...ok.
    XAMPP:  Done.

我们还需要修改一些参数，首先介绍几个重要的目录和文件

| 目录/文件 | 用途 |
|-|-|
| /opt/lampp/bin/ | XAMPP命令库（例如/opt/lampp/bin/mysql可执行MySQL监视器） |
| /opt/lampp/htdocs/ | Apache文档根目录 |
| /opt/lampp/etc/httpd.conf | Apache配制文件 |
| /opt/lampp/etc/my.cnf | MySQL配制文件 |
| /opt/lampp/etc/php.ini | PHP配制文件 |
| /opt/lampp/etc/proftpd.conf | ProFTPD配制文件 |
| /opt/lampp/phpmyadmin/config.inc.php | phpMyAdmin配制文件 |

修改端口号

    vi /opt/lampp/etc/httpd.conf

将`Listen 80`中的80修改成需要的端口，我这边修改成8080

    # Listen: Allows you to bind Apache to specific IP addresses and/or
    # ports, instead of the default. See also the <VirtualHost>
    # directive.
    #
    # Change this to Listen on specific IP addresses as shown below to
    # prevent Apache from glomming onto all bound IP addresses.
    #
    #Listen 12.34.56.78:80
    Listen 8080

然后重启服务

    [root@localhost bin]# /opt/lampp/lampp restart
    Restarting XAMPP for Linux 7.2.10-0...
    XAMPP: Stopping Apache...ok.
    XAMPP: Stopping MySQL...ok.
    XAMPP: Stopping ProFTPD...ok.
    XAMPP: Starting Apache...ok.
    XAMPP: Starting MySQL...ok.
    XAMPP: Starting ProFTPD...ok.

## oci8连接Oracle

这个集成环境需要设置才可以连接Oracle。

    [root@localhost lampp]# /opt/lampp/lampp oci8
    Please enter the path to your Oracle or Instant Client installation:
    [/opt/lampp/lib/instantclient-11.2.0.3.0] /usr/local/oracle/instantclient_11_2
    installing symlink...
    patching php.ini...
    OCI8 add-on activation likely successful.
    XAMPP: Stopping Apache...ok.
    XAMPP: Starting Apache...ok.

光选定`/usr/local/oracle/instantclient_11_2`，发现oci8并没有生效，只好老老实实安装oci8的包

    tar -xvf oci8-2.1.0.tgz
    cd oci8-2.1.0
    phpize
    ./configure --with-oci8=shared,instantclient,/usr/local/oracle/instantclient_11_2 --with-php-config=/opt/lampp/bin/php-config
    make
    make test
    make install

执行过程中，遇到以下问题，并给出了解决方法

服务器上没有autoconf

    [root@localhost oci8-2.1.0]# /opt/lampp/bin/phpize
    Configuring for:
    PHP Api Version:         20170718
    Zend Module Api No:      20170718
    Zend Extension Api No:   320170718
    Cannot find autoconf. Please check your autoconf installation and the
    $PHP_AUTOCONF environment variable. Then, rerun this script.

安装autoconf

    tar zxvf autoconf-2.69.tar.gz
    cd autoconf-2.69
    ./configure --prefix=/usr/
    make && make install
    /usr/bin/autoconf -V

服务器上没有gcc

    [root@localhost oci8-2.1.0]# make
    make: *** No targets specified and no makefile found.  Stop.

安装gcc，碰到需要安装依赖包

    [root@localhost tools]# rpm -ivh gcc-4.4.7-23.el6.x86_64.rpm
    warning: gcc-4.4.7-23.el6.x86_64.rpm: Header V3 RSA/SHA1 Signature, key ID c105b9de: NOKEY
    error: Failed dependencies:
    	cpp = 4.4.7-23.el6 is needed by gcc-4.4.7-23.el6.x86_64
    	libgcc >= 4.4.7-23.el6 is needed by gcc-4.4.7-23.el6.x86_64
    	libgomp = 4.4.7-23.el6 is needed by gcc-4.4.7-23.el6.x86_64

依次安装`cpp-4.4.7-23.el6.x86_64.rpm`、`libgcc-4.4.7-23.el6.x86_64.rpm`和`libgomp-4.4.7-23.el6.x86_64.rpm`，然后再安装gcc

    rpm -ivh cpp-4.4.7-23.el6.x86_64.rpm --replacefiles
    rpm -ivh libgcc-4.4.7-23.el6.x86_64.rpm --replacefiles
    rpm -ivh libgomp-4.4.7-23.el6.x86_64.rpm --replacefiles
    rpm -ivh gcc-4.4.7-23.el6.x86_64.rpm --replacefiles

修改/opt/lampp/etc/php.ini文件，把

    ; Directory in which the loadable extensions (modules) reside.
    ; http://php.net/extension-dir
    extension_dir = "/opt/lampp/lib/php/extensions/no-debug-non-zts-20170718"

然后重启服务器

    /opt/lammp/lampp restart

打开网址http://localhost:8080/dashboard/phpinfo.php可以看到已经有oci8了，出现下图代表成功。

![](http://wx3.sinaimg.cn/mw690/006F1DTzgy1fwrm46uiitj30tv0fwt9v.jpg)

我们可以试试连接Oracle，并且生产Json文件

    <?php
    $username='reporter';
    $password='reporter';
    $connection_string='//172.16.248.40:1521/ORCL';
    $conn = oci_connect(
        $username,
        $password,
        $connection_string,
    	'UTF8'
    );
    if (!$conn) { 
    $e = oci_error(); 
    echo $e['message']; 
    exit; 
    } 
    $query = 'SELECT DAY_DATE FROM DIM_DATE';
    $stid = oci_parse($conn, $query);
    $r = oci_execute($stid, OCI_DEFAULT);
    $data="";
      $array= array();
      class User{
        public $day_date;
      }
    while($row=oci_fetch_assoc($stid)){
        $user=new User();
        $user->day_date = $row['DAY_DATE'];
        $array[]=$user;
      }
      $data=json_encode($array);
      echo $data;
    oci_close($conn); 
    ?>

可以看到网页生成json数据

![](http://wx3.sinaimg.cn/mw690/006F1DTzgy1fwrmc1ml7rj30lx082jrr.jpg)

相关软件轻点我~~~
