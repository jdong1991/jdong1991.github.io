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

下面识别验证码实例，有个验证码长这样

![](/img/post-jd-tesserocr/10.jpg)

现在需要先对图片进行处理

import tesserocr
from PIL import Image

    img = Image.open('image.jpg')
	# 将图片变成灰色
	img_gray = img.convert('L')
	# 转成黑白图片
	img_black_white = img_gray.point(lambda x: 0 if x > 200 else 255)
	img_black_white.save('code_black_white.png')

这样得到图片

![](/img/post-jd-tesserocr/11.jpg)

还有许多噪点，可以处理下

	# 噪点处理
	def interference_point(img):
	    filename = 'C:/Users/JD/Pictures/code_result.png'
	    h, w = img.shape[:2]
	    # 遍历像素点进行处理
	    for y in range(0, w):
	        for x in range(0, h):
	            # 去掉边框上的点
	            if y == 0 or y == w - 1 or x == 0 or x == h - 1:
	                img[x, y] = 255
	                continue
	            count = 0
	            if img[x, y - 1] == 255:
	                count += 1
	            if img[x, y + 1] == 255:
	                count += 1
	            if img[x - 1, y] == 255:
	                count += 1
	            if img[x + 1, y] == 255:
	                count += 1
	            if count > 2:
	                img[x, y] = 255
	    cv2.imwrite(filename, img)
	    return img, filename

    img_cv = cv2.imread('code_black_white.png')
    im = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
	interference_point(im)

得到最终要识别的图片

![](/img/post-jd-tesserocr/12.jpg)

通过tesserocr来识别上图

	image = Image.open(r'code_result.png')
	result = tesserocr.image_to_text(image)
	print(result)  # h254

