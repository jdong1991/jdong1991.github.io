---
layout:     post
title:      Model Evaluation
subtitle:   
date:       2018-02-07
author:     JD
header-img: img/post-jd-eval.jpg
catalog: true
tags:
    - Machine Learning
    - Scoring
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

模型性能评估在机器学习中是不可或缺的环节，下面总结了sklearn中比较常用的。

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fo278vjfj5j30lr0flwfw.jpg)

设计时为了统一规则，`scoring`都是越大越好，像`metrics.mean_squared_error`是取它的负值`neg_mean_squared_error`。

# Classification

### accuracy

精确度的计算公式为

$$accuracy(y,\hat{y})=frac{1}{n_{samples}} \sum_{i=0}^{n_{samples}-1} I(\hat{y_i}=y_i)$$

\\(\hat{y_i}\\)是第i个样本的预测值，\\(y_i\\)是第i个样本的真实值,`I`是指示函数。

它的原理很简单，就是预测正确的占全体样本的比率。

在sklearn中`accuracy_score`可以很好实现它。

    import numpy as np
    from sklearn.metrics import accuracy_score
    y_pred = [0, 2, 1, 3]
    y_true = [0, 1, 2, 3]
    print('the correctly classified samples: ',accuracy_score(y_true, y_pred))
    print('the number of correctly classified samples: ',accuracy_score(y_true, y_pred, normalize=False))

代码运行结果

    the correctly classified samples:  0.5
    the number of correctly classified samples:  2

当`normalize=True`时，返回值是正确分类占全体样本的比值，当`normalize=False`时，返回值是正确分类的数目。

也是可以处理多分类问题。

    print(accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2))))

代码运行结果

    0.5

### Cohen's kappa

[Cohen's kappa系数](https://en.wikipedia.org/wiki/Cohen%27s_kappa)作为评价判断的一致性程度的指标。

$$k\equiv \frac{p_0-p_e}{1-p_e}$$

其中\\(p_o\\)在给任何样本的标签上达成一致的经验概率，\\(p_e\\)是随机分配标签时预期的概率。

下面是Wikipedia的例子

有50个人申请拨款，2个人审批，只可以说yes或者no

<table>
   <tr>
      <td rowspan='2' colspan='2'></td>
      <td colspan='2' align='center'>B</td>
   </tr>
   <tr>
      <td>Yes</td>
      <td>No</td>
   </tr>
   <tr>
      <td rowspan='2' align='center'>A</td>
      <td>Yes</td>
      <td>a</td>
      <td>b</td>
   </tr>
   <tr>
      <td>No</td>
      <td>c</td>
      <td>d</td>
   </tr>
</table>

等价于

<table>
   <tr>
      <td rowspan='2' colspan='2'></td>
      <td colspan='2' align='center'>B</td>
   </tr>
   <tr>
      <td>Yes</td>
      <td>No</td>
   </tr>
   <tr>
      <td rowspan='2' align='center'>A</td>
      <td>Yes</td>
      <td>20</td>
      <td>5</td>
   </tr>
   <tr>
      <td>No</td>
      <td>10</td>
      <td>15</td>
   </tr>
</table>

根据Cohen's kappa系数公式

\\(p_0=\frac{a+d}{a+b+c+d}=\frac{20+15}{50}=0.7\\)

\\(p_{Yes}=\frac{a+b}{a+b+c+d}\cdot \frac{a+c}{a+b+c+d}=0.5\ast 0.6=0.3\\)

\\(p_{No}=\frac{c+d}{a+b+c+d}\cdot \frac{b+d}{a+b+c+d}=0.5\ast 0.4=0.2\\)

\\(p_e=p_{Yes}+p_{No}=0.3+0.2=0.5 \\)

\\( k=\frac{p_0-p_e}{1-p_e}=\frac{0.7-0.5}{1-0.5}=0.4\\)

可以用sklearn中`cohen_kappa_score`来实现。

    from sklearn.metrics import cohen_kappa_score
    y_true = [1, 0, 1, 1, 0, 1]
    y_pred = [1, 1, 1, 1, 0, 0]
    print('Cohen's kappa coefficient: 'cohen_kappa_score(y_true, y_pred))

代码运行结果

    ohen's kappa coefficient: 0.25

### Confusion matrix

在机器学习中，有一个普遍适用的称为[混淆矩阵](https://en.wikipedia.org/wiki/Confusion_matrix)(confusion matrix)的工具，它可以帮助人们更好地了解分类中的错误。

有这样一个关于在房子周围可能发现的动物类型的预测，这个预测的三类问题的混淆矩阵如下表所示。

<table>
   <tr>
      <th style="background:white; border:none;" colspan="2" rowspan="2"></th>
      <th colspan="3" style="background:none;">Actual class</th>
   </tr>
   <tr>
      <th>Cat</th>
      <th>Dog</th>
      <th>Rabbit</th>
   </tr>
   <tr>
      <th rowspan="3" style="height:6em;">
         <div style="display: inline-block; -ms-transform: rotate(-90deg); -webkit-transform: rotate(-90deg); transform: rotate(-90deg);;">Predicted<br>
         class</div>
      </th>
      <th>Cat</th>
      <td>5</td>
      <td>2</td>
      <td>0</td>
   </tr>
   <tr>
      <th>Dog</th>
      <td>3</td>
      <td>3</td>
      <td>2</td>
   </tr>
   <tr>
      <th>Rabbit</th>
      <td>0</td>
      <td>1</td>
      <td>11</td>
   </tr>
</table>

利用混淆矩阵就可以更好的理解分类中的错误了。如果矩阵中的非对角线元素均为0，就会得到完美的分类器。

为简化处理，我们可以考虑另外一个混淆矩阵，该矩阵只针对一个简单的二类问题。在这个二类问题中，如果将一个正例判为正例，那么就可以认为产生了一个真正例(True Positive,TP,也称真阳)，如果对一个反例正确的判为反例，则认为产生了一个真反例(True Negative,TN,也称真阴)。相应的，另外两种情况则分别称为伪反例(False Negative,FN,也称假阴)和伪正例(False Positive,FP,也称假阳)，如下表所示。

<table>
   <tr>
      <th style="background:white; border:none;" colspan="2" rowspan="2"></th>
      <th colspan="3" style="background:none;">Actual class</th>
   </tr>
   <tr>
      <th>Cat</th>
      <th>Non-cat</th>
   </tr>
   <tr>
      <th rowspan="2" style="height:6em;">
      <div style="display: inline-block; -ms-transform: rotate(-90deg); -webkit-transform: rotate(-90deg); transform: rotate(-90deg);;">Predicted<br>
      class</div>
      </th>
      <th>Cat</th>
      <td>5 True Positives</td>
      <td>2 False Positives</td>
   </tr>
   <tr>
      <th>Non-cat</th>
      <td>3 False Negatives</td>
      <td>17 True Negatives</td>
   </tr>
</table>

在分类中，当某个类别的重要性高于其他类别时，我们就可以利用上述定义来定义出多个比错误率更好的新指标。其中使用较多的有精确率(Precision)，召回率(Recall),F1分数(F1-score)

\\(Precision=\frac{TP}{TP+FP}\\)

\\(Recall=\frac{TP}{TP+FN}\\)

\\(F1-score=\frac{2}{\frac{1}{Recall}+\frac{1}{Precision}}=2\cdot \frac{Precision\cdot Recall}{Precision+Recall}\\)

\\(F_{\beta}-score=(1+\beta^2)\cdot \frac{Precision\cdot Recall}{(\beta^2\cdot Precision)+Recall}\\)

skearn的`confusion_matrix`用来计算混淆矩阵。

    from sklearn.metrics import confusion_matrix
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    print(confusion_matrix(y_true, y_pred))

代码运行结果

    [[2 0 0]
     [0 0 1]
     [1 0 2]]

对于二分类问题，我们可以计算出TN,FP,FN,TP。

    y_true = [0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(tn, fp, fn, tp)

代码运行结果

    2 1 2 3

### Classification report

`classification_report`可以展现主要分类指标，包括上面说的Precision，Recall，F1-score。

    from sklearn.metrics import classification_report
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 0, 2, 1, 0]
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(y_true, y_pred, target_names=target_names))

代码运行结果

                 precision    recall  f1-score   support
    
        class 0       0.67      1.00      0.80         2
        class 1       0.00      0.00      0.00         1
        class 2       1.00      0.50      0.67         2
    
    avg / total       0.67      0.60      0.59         5

其中Support代表每个类别的样本数，最下面一行是根据support的加权值。precision=(0.67*2+0.00*1+1.00*2)/5=0.67,recall和f1-score类似。

### Hamming loss

`hamming_loss`计算两个样本之间额Hamming距离。

$$L_{Hamming} (y,\hat{y})=\frac{1}{n_{labels}} \sum_{j=0}^{n_{labels}-1} I(\hat{y_j}\ne y_j)$$

其中，\\(\hat{y_j}\\)是第j个样本的预测值，\\(y_j\\)是第j个样本的真实值，\\(n_{labels}\\)是标签总数，I(x)是指示函数。

sklearn中的实现

    from sklearn.metrics import hamming_loss
    y_pred = [1, 2, 3, 4]
    y_true = [2, 2, 3, 4]
    print(hamming_loss(y_true, y_pred))

代码运行结果

    0.25

在二分类的多标签问题中


    print(hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2))))

代码运行结果

    0.75

### Jaccard similarity coefficient score

Jaccard指数是用来比较样本集的相似性和多样性的统计数据，它的公式：

$$J(y_i,\hat{y_i})=\frac{|y_i\bigcap \hat{y_i}|}{|y_i\bigcup \hat{y_i}|}$$

其中，\\(\hat{y_j}\\)是第j个样本的预测值，\\(y_j\\)是第j个样本的真实值。

在二分类和多分类问题中

    import numpy as np
    from sklearn.metrics import jaccard_similarity_score
    y_pred = [0, 2, 1, 3]
    y_true = [0, 1, 2, 3]
    print(jaccard_similarity_score(y_true, y_pred))
    print(jaccard_similarity_score(y_true, y_pred, normalize=False))

代码运行结果

    0.5
    2

在二分类多标签问题中

    print(jaccard_similarity_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2))))

代码运行结果

    0.75

### Precision, recall and F-measures

前面在混淆矩阵有讲过这些指标。

- precision是被分为正例的示例中实际为正例的比例
- recall是实际为正例的示例中预测为正例的比例。
- F-measure是precision和recall的加权平均

其中precision和recall的偏重应该视具体情况，比如查找涉黄读物，那我们应该更希望recall更高，因为这样可以找出更多的危害读物，尽管可能会误杀更多没问题的读物。下面列出了sklearn中计算上述指标的函数。

![](http://wx3.sinaimg.cn/mw690/006F1DTzgy1fo6ilu6rgjj30rp05njs0.jpg)

需要注意的是`precision_recall_curve`值只可以用于二元问题，average_precision_score只可以在二分类多标签的问题。

在二分类问题中

    import numpy as np
    from sklearn import metrics
    
    y_pred = [0, 1, 0, 0]
    y_true = [0, 1, 0, 1]
    print('precision_score:',metrics.precision_score(y_true, y_pred))
    print('recall_score:',metrics.recall_score(y_true, y_pred))
    print('f1_score:',metrics.f1_score(y_true, y_pred))
    print('fbeta_score:',metrics.fbeta_score(y_true, y_pred, beta=0.5))  
    print('fbeta_score:',metrics.fbeta_score(y_true, y_pred, beta=1))
    print('fbeta_score:',metrics.fbeta_score(y_true, y_pred, beta=2))
    print('precision_recall_fscore_support:',metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5))
    
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_scores)
    print('precision:',precision)
    print('recall:',recall)
    print('threshold:',threshold)
    print(metrics.average_precision_score(y_true, y_scores))

代码运行结果

    precision_score: 1.0
    recall_score: 0.5
    f1_score: 0.6666666666666666
    fbeta_score: 0.8333333333333334
    fbeta_score: 0.6666666666666666
    fbeta_score: 0.5555555555555556
    precision_recall_fscore_support: (array([0.66666667, 1.        ]), array([1. , 0.5]), array([0.71428571, 0.83333333]), array([2, 2], dtype=int64))

    precision: [0.66666667 0.5        1.         1.        ]
    recall: [1.  0.5 0.5 0. ]
    threshold: [0.35 0.4  0.8 ]
    average_precision_score: 0.8333333333333333

在多分类和多标签分类中

- y是预测(sample,label)的集合
- \\(\hat{y}\\)是真实(sample,label)的集合
- L是标签的集合
- S是样本的集合
- \\(y_s\\)是样本s的子集，\\(y_s:=\{(s',l)\in y|s'=s \}\\)
- \\(y_l\\)是标签为l的y的子集
- 相似的，\\(\hat{y_s}\\)和\\(\hat{y_l}\\)是\\(\hat{y}\\)的子集
- \\(P(A,B):=\frac{|A\bigcap B|}{|A|}\\)
- \\(R(A,B):=\frac{|A\bigcap B|}{|A|}\\)
- \\(F_{\beta}(A,B):=(1+\beta^2)\frac{P(A,B)\cdot R(A,B)}{\beta^2\cdot P(A,B)+R(A,B)}\\)

![](http://wx2.sinaimg.cn/mw690/006F1DTzgy1fo6kfjhuorj30rn05j3zn.jpg)

    from sklearn import metrics
    
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    print('precision_score:',metrics.precision_score(y_true, y_pred, average='macro'))
    print('recall_score:',metrics.recall_score(y_true, y_pred, average='micro'))
    print('f1_score:',metrics.f1_score(y_true, y_pred, average='weighted')) 
    print('fbeta_score:',metrics.fbeta_score(y_true, y_pred, average='macro', beta=0.5))
    print('precision_recall_fscore_support:',metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5, average=None))

    print('recall_score:',metrics.recall_score(y_true, y_pred, labels=[1, 2], average='micro'))
    print('precision_score:',metrics.precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro'))

代码运行结果

    precision_score: 0.2222222222222222
    recall_score: 0.3333333333333333
    f1_score: 0.26666666666666666
    fbeta_score: 0.23809523809523805
    precision_recall_fscore_support: (array([0.66666667, 0.        , 0.        ]), array([1., 0., 0.]), array([0.71428571, 0.        , 0.        ]), array([2, 2, 2], dtype=int64))

    recall_score: 0.0
    precision_score: 0.16666666666666666

### Hinge loss

在机器学习中，[Hinge loss](https://en.wikipedia.org/wiki/Hinge_loss)是训练分类的一种损失函数，它是被使用在最大边缘分类，尤其是在支持向量机(SVMs)中。

若标签为+1或-1，y是真实值，w是决策函数的输出值，这时Hinge loss如下面公式：

$$L_{Hinge}(y,w)=max\{0,1-wy\}=|1-wy|_{+}$$

对于多分类问题，公式如下：

$$L_{Hinge}(y_w,y_t)=max\{1+y_t-y_w,0\}$$

使用`hinge_loss`在一个解决二分类问题的svm算法中

    from sklearn import svm
    from sklearn.metrics import hinge_loss
    
    X=[[0],[1]]
    y=[-1,1]
    est=svm.LinearSVC(random_state=0)
    est.fit(X,y)
    pre_decision=est.decision_function([[-2],[3],[0.5]])
    print('pre_decision:',pre_decision)
    print('hinge_loss:',hinge_loss([-1,1,1],pre_decision))

代码运行结果

    pre_decision: [-2.18177262  2.36361684  0.09092211]
    hinge_loss: 0.3030259636876493

`hinge_loss`在多分类svm分类器中

    import numpy as np
    from sklearn import svm
    from sklearn.metrics import hinge_loss
    
    X=np.array([[0],[1],[2],[3]])
    y=np.array([0,1,2,3])
    labels=np.array([0,1,2,3])
    est=svm.LinearSVC()
    est.fit(X,y)
    pre_decision=est.decision_function([[-1],[2],[3]])
    y_true=[0,2,3]
    print('hinge_loss:',hinge_loss(y_true,pre_decision,labels))

代码运行结果

    hinge_loss: 0.5641094289148848

### Log loss

Log loss也被叫做逻辑回归损失函数或者交叉熵损失函数。它通常被使用在逻辑回归和神经网络中。

对于二分类，\\(y\in \{0,1\}\\),\\(p=Pr(y=1)\\)，公式如下：

$$L_{log}(y,p)=-logPr(y|p)=-(ylog(p)+(1-y)log(1-p))$$

也可以用于多分类问题，K是表示一共有K个标签，k是K个标签中的第k个，\\(y_{i,k}\\)是样本i有标签k，\\(p_{i,k}=Pr(t_{i,k}=1)\\)，公式如下

$$L_{log}(Y,P)=-logPr(Y|P)=-\frac{1}{N}\sum_{i=0}^{N-1}\sum_{k=0}^{K-1}y_{i,k}logp_{i,k}$$

`log_loss`是根据真实值和预测可能概率来计算log loss。

    from sklearn.metrics import log_loss
    
    y_true=[0,0,1,1]
    y_pred=[[.9,.1],[.8,.2],[.3,.7],[.01,.99]]
    print('log_loss:',log_loss(y_true,y_pred))

代码运行结果

    log_loss: 0.1738073366910675

### Matthews correlation coefficient

本质上，MCC是观察和预测的二元分类之间的一个相关系数，它返回-1到1之间，其中+1代表完美的预测，0代表和随机预测一样，-1代表预测值和真实值完全相反。

MCC可以直接用前面所讲的混淆矩阵计算，如下：

$$MCC=\frac{tp\times tn-fp\times fn}{\sqrt{(tp+fp)(tp+fn)(tn+fp)(tn+fn)}}$$

在多分类问题中，K个不同类，混淆矩阵C为KxK

- \\(t_k=\sum_i^K C_{ik}\\) 第k类真实出现的次数
- \\(p_k=\sum_i^K C_{ki}\\) 第k类预测出现的次数
- \\(c=\sum_k^K C_{kk}\\) 样本被正确预测的总数
- \\(s=\sum_i^K\sum_j^K C_{ij}\\) 样本总数

$$MCC=\frac{c\times s-\sum_k^K p_k\times t_k}{\sqrt{(s^2-\sum_k^K p_k^2)\times (s^2-\sum_k^K t_k^2)}}$$

下面是一个简单的例子

    from sklearn.metrics import matthews_corrcoef
    
    y_true=[+1,+1,+1,-1]
    y_pred=[+1,-1,+1,+1]
    print('matthews_corrcoef:',matthews_corrcoef(y_true,y_pred))

代码运行结果

    matthews_corrcoef: -0.3333333333333333

### Receiver operating characteristic (ROC)

ROC曲线可以表明分类系统的性能。它是由不同的阈值设置绘制TPR和FPR而成，其中\\(TPR=\frac{TP}{TP+FN}\\)，\\(FPR=\frac{FP}{FP+TN}\\)。图形越靠近左上角，说明性能越好。

下面是使用`roc_curve`的实例

    import numpy as np
    from sklearn.metrics import roc_curve
    
    y=np.array([1,1,2,2])
    scores=np.array([0.1,0.4,0.35,0.8])
    fpr,tpr,thresholds=roc_curve(y,scores,pos_label=2)
    print('fpr:',fpr)
    print('tpr:',tpr)
    print('thresholds:',thresholds)

代码运行结果

	fpr: [0.  0.5 0.5 1. ]
	tpr: [0.5 0.5 1.  1. ]
	thresholds: [0.8  0.4  0.35 0.1 ]

下图也是ROC曲线的例子

一个特定类的ROC曲线

[![](http://scikit-learn.org/stable/_images/sphx_glr_plot_roc_001.png)](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)

多分类的ROC曲线

[![](http://scikit-learn.org/stable/_images/sphx_glr_plot_roc_002.png)](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)

### Zero one loss

0-1分类的损失函数在全样本的汇总值或者平均值。当设置`normalize=False`时，\\(L_{0-1}\\)求汇总值。

$$L_{0-1}(y_i,\hat{y_i})=I(\hat{y_i}\ne y_i)$$

其中\\(I(X)\\)是指示函数。

    from sklearn.metrics import zero_one_loss
    
    y_pred=[1,2,3,4]
    y_true=[2,2,3,4]
    print('the average:',zero_one_loss(y_true,y_pred))
    print('the sum:',zero_one_loss(y_true,y_pred,normalize=False))

代码运行结果

    the average: 0.25
    the sum: 1

多标签也是类似，如下：

    import numpy as np
    from sklearn.metrics import zero_one_loss
    
    print('the average:',zero_one_loss(np.array([[0,1],[1,1]]),np.ones((2,2))))
    print('the sum:',zero_one_loss(np.array([[0,1],[1,1]]),np.ones((2,2)),normalize=False))

代码运行结果

    the average: 0.5
    the sum: 1

### Brier score loss

Brier score是用来度量概率预测的正确率。Brier score loss是在0到1之间，值越小说明预测越正确。

$$BS=\frac{1}{N}\sum_{t=1}^N(f_t-o_t)^2$$

其中N是预测的总样本数，\\(f_t\\)是预测概率，\\(o_t\\)是实际输出值。

    import numpy as np
    from sklearn.metrics import brier_score_loss
    
    y_true=np.array([0,1,1,0])
    y_true_categorical=np.array(['spam','ham','ham','spam'])
    y_prob=np.array([0.1,0.9,0.8,0.4])
    y_pred=np.array([0,1,1,0])
    print('brier score:',brier_score_loss(y_true,y_prob))
    print('brier score:',brier_score_loss(y_true,1-y_prob,pos_label=0))
    print('brier score:',brier_score_loss(y_true_categorical,y_prob,pos_label='ham'))
    print('brier score of y_prob>0.5:',brier_score_loss(y_true,y_prob>0.5))

代码运行结果

    brier score: 0.055
    brier score: 0.055
    brier score: 0.055
    brier score of y_prob>0.5: 0.0

前3个`brier_score_loss`是等价的，`pos_label`是指定以哪个为1，最后一个`brier_score_loss`也可以写成`brier_score_loss(y_true,=np.array([0,1,1,0]))`,可以发现，直接求accuracy的效果并不好，但是求概率可以更加精确，当然前提可以提供预测概率。

### Multilabel ranking metrics

多标签问题有以下几种方法来测试性能。

#### Coverage error

Coverage error是计算实际标签所对应的概率标签，该概率标签在预测样本的多标签排序的序号，再求平均值。

$$coverage(y,\hat{f})=\frac{1}{n_{samples}}\sum_{i=0}{n_{samples}-1}max_{j:y_{ij}=1}rank_{ij}$$

$$with rank_{ij}=|\{k:\hat{f_{ik}}\geqslant \hat{f_{ij}} \}|$$

`coverage_error`实现

    import numpy as np
    from sklearn.metrics import coverage_error
    
    y_true=np.array([[1,0,0],[0,0,1]])
    y_score=np.array([[0.75,0.5,1],[1,0.2,0.1]])
    print('coverage_error:',coverage_error(y_true,y_score))

代码运行结果

    coverage_error: 2.5

上面代码是先找出`y_true`中的1，然后找到`y_score`对应的[[0.75],[0.1]]，然后它们在各自排序的序号[[3],[2]],然后求均值(3+2)/2=2.5。

#### Label ranking average precision

`label_ranking_average_precision_score`计算实际标签的平均精度。

$$LRAP(y,\hat{f})=\frac{1}{n_{samples}}\sum_{i=0}^{n_{samples}-1} \frac{1}{|y_i|}\sum_{j:y_{ij}=1}\frac{|L_{ij}|}{rank_{ij}}$$ 

$$with L_{ij}=\{k:y_{ij}=1,\hat{f_{ik}} \geqslant \hat{f_{ij}}\},rank_{ij}=|\{k:\hat{f_{ik}}\geqslant \hat{f_{ij}} \}|$$

下面是个小例子

    import numpy as np
    from sklearn.metrics import label_ranking_average_precision_score
    
    y_true=np.array([[1,0,0],[0,0,1]])
    y_score=np.array([[0.75,0.5,1],[1,0.2,0.1]])
    print('label_ranking_average_precision_score:',label_ranking_average_precision_score(y_true,y_score))

代码运行结果

    label_ranking_average_precision_score: 0.41666666666666663

上面例子中，首先\\(rank_{ij}\\)求法和`Coverage error`一样，分别是[[2],[3]],\\(L_{ij}\\)分别是[[1],[1]]，对应着[[0.75],[0.1]]，然后求平均值\\(\frac{\frac{1}{2}+\frac{1}{3}}{2}\approx 0.416\\)。

#### Ranking loss

python实现

    import numpy as np
    from sklearn.metrics import label_ranking_loss
    
    y_true=np.array([[1,0,0],[0,0,1]])
    y_score=np.array([[0.75,0.5,1],[1,0.2,0.1]])
    print('label_ranking_loss1:',label_ranking_loss(y_true,y_score))
    y_score=np.array([[1.0,0.1,0.2],[0.1,0.2,0.9]])
    print('label_ranking_loss2:',label_ranking_loss(y_true,y_score))

代码运行结果

    label_ranking_loss1: 0.75
    label_ranking_loss2: 0.0

# Regression

回归比较简单，简略介绍下。

### Explained variance score

explained variation是用来度量一个数据模型对给定数据集的变化(分散)的比例。可以用下面公式估计：

$$explained_variance(y,\hat{y})=1-\frac{Var\{y-\hat{y}\}}{Var\{y\}}$$

其中最好的分数是1，越小越差。

实例如下：

    import numpy as np
    from sklearn.metrics import explained_variance_score
    
    y_true=[3,-0.5,2,7]
    y_pred=[2.5,0.0,2,8]
    print('explained_variance_score1:',explained_variance_score(y_true,y_pred))
    
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    print('explained_variance_score2:',explained_variance_score(y_true, y_pred, multioutput='raw_values'))
    print('explained_variance_score3:',explained_variance_score(y_true, y_pred, multioutput=[0.3, 0.7]))

代码运行结果

    explained_variance_score1: 0.9571734475374732
    explained_variance_score2: [0.96774194 1.        ]
    explained_variance_score3: 0.9903225806451612

### Mean absolute error

MAE是2个连续变量之间的差值。

$$MAE(y,\hat{y})=\frac{1}{n_{samples}}\sum_{i=0}^{n_{samples}-1}|y_i-\hat{y_i}|$$

其中\\(n_{samples}\\)是样本数量，\\(y_i\\)第i个样本的正确值，\\(\hat{y_i}\\)是第i个样本的预测值。

sklearn中`mean_absolute_error`来进行计算

    from sklearn.metrics import mean_absolute_error
    
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    print('mean_absolute_error1:',mean_absolute_error(y_true, y_pred))
    
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    print('mean_absolute_error2:',mean_absolute_error(y_true, y_pred))
    print('mean_absolute_error3:',mean_absolute_error(y_true, y_pred, multioutput='raw_values'))
    print('mean_absolute_error4:',mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7]))

代码运行结果

    mean_absolute_error1: 0.5
    mean_absolute_error2: 0.75
    mean_absolute_error3: [0.5 1. ]
    mean_absolute_error4: 0.85

###  Mean squared error

MSE是用来评估估计值质量。公式如下：

$$MSE(y,\hat{y})\sum_{i=0}^{n_{samples}-1}(y_i-\hat{y_i})^2$$

使用`mean_squared_error`实现

    from sklearn.metrics import mean_squared_error
    
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    print('mean_squared_error:',mean_squared_error(y_true, y_pred))
    
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    print('mean_squared_error:',mean_squared_error(y_true, y_pred))

代码运行结果

    mean_squared_error: 0.375
    mean_squared_error: 0.7083333333333334

### Mean squared logarithmic error

MSLE和MSE类似，不过先取`np.log1p`，然后再进行MSE。

$$MSLE(y,\hat{y})\sum_{i=0}^{n_{samples}-1}(log_e(1+y_i)-log_e(1+\hat{y_i}))^2$$

`mean_squared_log_error`实现

    from sklearn.metrics import mean_squared_log_error
    
    y_true = [3, 5, 2.5, 7]
    y_pred = [2.5, 5, 4, 8]
    print('mean_squared_log_error1:',mean_squared_log_error(y_true, y_pred))
    
    y_true = [[0.5, 1], [1, 2], [7, 6]]
    y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
    print('mean_squared_log_error2:',mean_squared_log_error(y_true, y_pred))

代码运行结果

    mean_squared_log_error1: 0.03973012298459379
    mean_squared_log_error2: 0.044199361889160536

### Median absolute error

中位绝对误差是对异常值有很好的鲁棒性。它是所有正确标签和估计标签的差值的绝对值，然后对这些绝对值求中位数。

$$MedAE(y,\hat{y})=median(|y_1-\hat{y_1}|,...,|y_n-\hat{y_n}|)$$

`median_absolute_error`实现

    from sklearn.metrics import median_absolute_error
    
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    print('median_absolute_error:',median_absolute_error(y_true, y_pred))

代码运行结果

    median_absolute_error: 0.5

### R² score, the coefficient of determination

R² score是可以度量模型预测未知样本的性能。R² score为1最好，也可以为负数。当是一个常数模型，对于任何的输入值，总是可以得到预测值y，那么R² score为0.

$$R^2 (y,\hat{y})=1-\frac{\sum_{i=0}^{n_{samples}-1} (y_i-\hat{y_i})^2}{\sum_{i=0}^{n_{samples}-1} (y_i-\hat{y})^2}$$

$$where y^{-}=\frac{1}{n_{samples}}\sum_{i=0}^{n_{samples}y_i$$

`r2_score`实现

    from sklearn.metrics import r2_score
    
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    print('r2_score1:',r2_score(y_true, y_pred))
    
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    print('r2_score2:',r2_score(y_true, y_pred, multioutput='variance_weighted'))
    
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    print('r2_score3:',r2_score(y_true, y_pred, multioutput='uniform_average'))
    print('r2_score4:',r2_score(y_true, y_pred, multioutput='raw_values'))
    print('r2_score5:',r2_score(y_true, y_pred, multioutput=[0.3, 0.7]))

代码运行结果

    r2_score1: 0.9486081370449679
    r2_score2: 0.9382566585956417
    r2_score3: 0.9368005266622779
    r2_score4: [0.96543779 0.90816327]
    r2_score5: 0.9253456221198156

未完待续。
