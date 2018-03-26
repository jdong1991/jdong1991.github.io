---
layout:     post
title:      Kernel Function
subtitle:   
date:       2018-03-23
author:     JD
header-img: img/post-jd-kf.jpg
catalog: true
tags:
    - Kernel
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

最近面试碰到了以前未关注过的知识点，恶补下。

核函数(Kernel Function)是我在使用SVM的时候get到的新知识，当然它还可以应用在很多其他的机器学习算法。

这里主要是介绍`sklearn.svm`中常用的核函数。

### Linear kernel

$$k(x,y)=x^Ty$$

线性核函数是最简单的核函数，主要用于线性可分，它在原始空间中寻找最优线性分类器，具有参数少速度快的优势。

需要注意`sklearn.svm.LinearSVC()`和`sklearn.svm.SVC(kernel='linear')`实现方法并不相同。

- LinearSVC使用liblinear，而SVC使用的是libsvm
- LinearSVC使用One-vs-All，而SVC使用One-vs-One
- linearSVC使用the squared hinge loss，而SVC使用the regular hinge loss

### Cosine similarity

$$k(x,y)=\frac{xy^T}{\parallel x\parallel \parallel y \parallel}$$

余弦相似核函数主要计算用tf-idf向量表示的文档的相似性。

注意当tf-idf在`sklearn.feature_extraction.text`能生成标准化向量，`cosine_similarity`是相当于`linear_kernel`，只不过运行更慢。

### Polynomial kernel

$$k(x,y)=(\gamma x^Ty+c_0)^d$$

多项式核适合正交归一化数据。多项式核函数属于全局核函数，可以实现低维的输入空间映射到高维的特征空间。其中参数d越大，映射的维度越高，和矩阵的元素值越大。故易出现过拟合现象。

### Sigmoid kernel

$$k(x,y)=tanh(\gamma x^Ty+c_0)$$

采用Sigmoid核函数，支持向量机实现的就是一种多层感知器神经网络。

### RBF kernel

$$k(x,y)=exp(-\gamma \parallel x-y\parallel ^2)$$

径向基核函数属于局部核函数，允许相距很远的数据点对核函数的值有影响。径向基核函数具有良好的鲁棒性，并且对于多项式核函数参数要少，对于大样本和小样本都具有比较好的性能，因此在多数情况下不知道使用什么核函数，优先选择径向基核函数。

### Laplacian kernel

$$k(x,y)=exp(-\gamma \parallel x-y\parallel _1)$$

拉普拉斯核完全等价于指数核，唯一区别在于前者对参数的敏感性降低，也是一种径向即核函数。

### Chi-squared kernel

$$k(x,y)=exp(-\gamma \sum_i \frac{(x[i]-y[i])^2}{x[i]+y[i]})$$

卡方核广泛使用在计算机视觉应用程序中非线性SVM的训练。

	from sklearn.svm import SVC
	from sklearn.metrics.pairwise import chi2_kernel
	
	X=[[0,1],[1,0],[.2,.8],[.7,.3]]
	y=[0,1,0,1]
	K=chi2_kernel(X,gamma=.5)
	print('K:\n',K)
	svm=SVC(kernel='precomputed').fit(K,y)
    #svm=SVC(kernel=chi2_kernel).fit(X,y)
	print('predict:',svm.predict(K))

代码运行结果

    K:
     [[1.         0.36787944 0.89483932 0.58364548]
     [0.36787944 1.         0.51341712 0.83822343]
     [0.89483932 0.51341712 1.         0.7768366 ]
     [0.58364548 0.83822343 0.7768366  1.        ]]
    predict:  [0 1 0 1]

### Custom Kernels

我们也可以自己定义核函数。根据Mercer定理：任何半正定的函数都可以作为核函数。

需要注意的是Mercer定理不是核函数的必要条件，只是一个充分条件。

可以参考sklearn官网上的例子，如下图。

[![](http://wx1.sinaimg.cn/mw690/006F1DTzgy1fpq2zymr2jj30hs0dcmy5.jpg)](http://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html)