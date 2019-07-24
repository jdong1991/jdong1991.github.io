---
layout:     post
title:      tesserocr
subtitle:   
date:       2019-06-27
author:     JD
header-img: img/post-jd-tesserocr.jpg
catalog: true
tags:
    - tesserocr
---

`tesserocr`是Python的一个OCR识别库，其实就是对`tesseract`做了一层Python API的封装。本质还是使用`tesseract`。

因此，要想使用`tesserocr`，就需要先安装`tesseract`。

tesseract的下载地址为[https://digi.bib.uni-mannheim.de/tesseract/](https://digi.bib.uni-mannheim.de/tesseract/)

本次下载的是[tesseract-ocr-w64-setup-v4.1.0.20190314.exe](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v4.1.0.20190314.exe)

在Windows10下安装

![](/img/post-jd-tesserocr/1.jpg)

`next`->`I Agree`->`next`，然后到达`Choose Components`，可以选择需要的。

![](/img/post-jd-tesserocr/2.jpg)

`Additional language data(download)`里面可以选择所需要的语言包。然后选择安装路径

![](/img/post-jd-tesserocr/3.jpg)

安装完成后，需要自己配置环境变量

在`Path`里添加`C:\Tesseract-OCR`和`C:\Tesseract-OCR\tessdata`

![](/img/post-jd-tesserocr/4.jpg)

在系统变量中新增变量`TESSDATA_PREFIX`,变量值是`C:\Tesseract-OCR\tessdata`

![](/img/post-jd-tesserocr/5.jpg)

可以使用验证码来测试下

![](/img/post-jd-tesserocr/6.jpg)

使用命令
    
    tesseract image.png result -l eng 

![](/img/post-jd-tesserocr/7.jpg)

得到图片

![](/img/post-jd-tesserocr/8.jpg)

已经成功将图片转成文本。

接下来需要使用python来调用（python版本为3.7）。

然后安装`tesserocr`，使用pip来安装
    
    pip install tesserocr

若出现下面报错，可以下载whl包来安装，下载地址在这 [tesserocr](https://github.com/simonflueckiger/tesserocr-windows_build/releases)

     error: Microsoft Visual C++ 14.0 is required

查看安装成功

![](/img/post-jd-tesserocr/9.jpg)

下面识别验证码实例

有个验证码长这样

![](/img/post-jd-tesserocr/10.jpg)

现在需要先