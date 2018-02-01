---
layout:     post
title:      Catergorical Attributes
subtitle:   
date:       2018-01-31
author:     JD
header-img: img/post-jd-feature.jpg
catalog: true
tags:
    - Feature Engineering
    - Categorical Attribute
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

# 定性特征的处理

最近看到一篇解决**High-Cardinality Categorical Attributes**的论文，收益颇多，在这总结一番。

## low-cardinality categorical attributes
先来看看常见的低基数的定性特征。

1.特征是数值型，可以使用`sklearn.preprocessing.OneHotEncoder`：

    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder()
    enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    print('n_values_: ',enc.n_values_)
    print('feature_indices_: ',enc.feature_indices_)
    print(enc.transform([[0, 1, 1]]).toarray())

代码运行结果

    n_values_:  [2 3 4]
	feature_indices_:  [0 2 5 9]
	[[1. 0. 0. 1. 0. 0. 1. 0. 0.]]

其中`n_values_`是每个特征的基数,`feature_indices_`是`n_values_`的累加，从0开始。

|A|B|C|
|-|:-:|-:|
|0|0|3|
|1|1|0|
|0|2|1|
|1|0|2|
|[0,1]|[0,1,2]|[0,1,2,3]|

`enc.transform([[0, 1, 1]]).toarray()`等价于下表，可以通过`pd.get_dummies()`将ML中的原特征(A,B,C)转化。

|A_0|A_1|B_0|B_1|B_2|C_0|C_1|C_2|C_3|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1|0|0|1|0|0|1|0|0|


2.特征是字符串型，若直接使用`preprocessing.OneHotEncoder`会报错，这时候可以使用sklearn中的`preprocessing.LabelEncoder`。
将字符串转换成数值型，然后再使用`preprocessing.OneHotEncoder`。

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(["paris", "paris", "tokyo", "amsterdam"])
    print('classes_: ',list(le.classes_))
    print('transform: ',le.transform(["tokyo", "tokyo", "paris"]))
    print('inverse_transform: ',list(le.inverse_transform([2, 2, 1])))

代码运行结果

    classes_:  ['amsterdam', 'paris', 'tokyo']
    transform:  [2 2 1]
    inverse_transform:  ['tokyo', 'tokyo', 'paris']

当然也可以直接使用`preprocessing.LabelBinarizer`，如：

    from sklearn.preprocessing import LabelBinarizer

	print('fit_transform1: \n',LabelBinarizer().fit_transform(['yes', 'no', 'no', 'yes']))
    print('fit_transform2: \n',LabelBinarizer().fit_transform(['yes', 'no', 'maybe', 'yes']))

代码运行结果

    fit_transform1: 
	 [[1]
	 [0]
	 [0]
	 [1]]
	fit_transform2: 
	 [[0 0 1]
	 [0 1 0]
	 [1 0 0]
	 [0 0 1]]

3.[sklearn官网](http://scikit-learn.org/dev/whats_new.html)中有提及将在V0.20中新加`CategoricalEncoder`，对比OneHotEncoder，这个新类可以处理各种类型特征，其中也包括包含字符串的特征。

    enc = CategoricalEncoder(handle_unknown='ignore')
    enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    print(enc.transform([[0, 1, 1], [1, 0, 4]]).toarray())

代码运行结果

    [[1. 0. 0. 1. 0. 0. 1. 0. 0.]
     [0. 1. 1. 0. 0. 0. 0. 0. 0.]]

也可以使用`encoder='onehot-dense'`，这样可以直接转换成稠密数列。




## High-Cardinality Categorical Attributes
### 理论基础
有时候并没有如此幸运。有许多高基数的定性特征存在，比如城市名称、人员ID等。这时候很明显使用One-Hot已经不合适了，那我们应该怎么办？

可能你首先会想到用clustering，这的确是个方法，可以将定性特征由N转换成K类，成功的缩小特征基数。

下面粗略介绍另外一种方法，来自这篇[论文](https://pan.baidu.com/s/1bqcNJjh)，感兴趣可以去看看，可以很好的处理高基数定性特征。


假设在一个二分类问题中，预测目标为**Y**,\\(Y\in(0,1)\\),\\(X_i\\)是特征**X**的一个转换后的值，\\(S_i\\)是已知\\(X=X_i\\)，**Y**出现的概率。

$$X_i\to S_i=P(Y|X=X_i)\qquad (1)$$

当训练集中\\(X=X_i\\)的数量是足够大，概率估计可以用(2)来计算，其中\\(n_{iY}\\)为Y=1的特征中为i的个数，\\(n_i\\)是的特征为i的个数。

$$S_i=\frac{n_{iY}}{n_i} \qquad (2)$$

可惜这种估计在高基数的定性特征表现的并不好，因此我们需要加强它，根据主观贝叶斯理论，可以将公式改造成公式(3)

$$ S_i = \lambda(n_i) \frac{n_{iY}}{n_i} + (1-\lambda(n_i)) \frac{n_{iY}}{n_T} \qquad (3)$$

其中\\(\lambda(n_i)\\)是单调递增函数，取值在(0,1)，\\(n_Y\\)是训练集中Y=1的样本数，\\(n_{TR}\\)是训练集的总样本数。

公式(3)中在(2)的基础上新加了\\( \frac{n_{iY}}{n_T}\\)，对于Y=1在训练集上的先验概率。当\\(\lambda\cong 1\\)，满足公式(2)的使用条件，可以使用后验概率来进行估计，
当\\(\lambda\cong 0\\)，\\(n_i\\)很好，这样就可以用先验概率来估计。

其中，\\(\lambda(n_i)\\)有许多种，这边就选择论文中的s-shaped函数

$$\lambda(n)=\frac{1}{1+e^{-\frac{(n-k)}{f}}} \qquad (4)$$

到这里，其实对于classification问题的高基数定性特征的处理已经完成，对于regression问题，在公式(3)上做一些的变化得到公式(5)，但是在实际处理中
，二值分类和连续目标值都是使用一个公式，只是把二值分类的目标值当成是数值目标0和1。

$$S_i=\lambda(n_i)\frac{\sum_{k\in L_i}Y_k}{n_i}+(1-\lambda(n_i))\frac{\sum_{k=1}^{N_{TR}}Y_k}{n_{TR}} \qquad (5)$$

### Python实现

##### 准备好所需的package

    import numpy as np
	import pandas as pd
	from sklearn.model_selection import StratifiedKFold,KFold
	from itertools import product

##### 新建一个类，并定义好所需变量

    class MeanEncoder:
	    def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
	        self.categorical_features = categorical_features
	        self.n_splits = n_splits
	        self.learned_stats = {}
	
	        if target_type == 'classification':
	            self.target_type = target_type
	            self.target_values = []
	        else:
	            self.target_type = 'regression'
	            self.target_values = None
	
	        if isinstance(prior_weight_func, dict):
	            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp(-(x - k) / f))', dict(prior_weight_func, np=np))
	        elif callable(prior_weight_func):
	            self.prior_weight_func = prior_weight_func
	        else:
	            self.prior_weight_func = lambda x: 1 / (1 + np.exp(-(x - 2) / 1))

其中`categorical_features`是所需转换的定性特征，`n_splits`是交叉检验的折数，默认为5折，`learned_stats`是保存学习知识集，本类主要是为区分classification和regression，2种问题的计算略有不同。`prior_weight_func`是\\(\lambda(n)\\)，可以选择自己定义，也可以使用默认。

##### 先验概率和后验概率

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train
        prior = X_train['pred_temp'].mean()

        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * col_avg_y['mean'] + (1 - col_avg_y['beta']) * prior
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

在regression中，`prior`是\\(\frac{n_Y}{n_{TR}}\\)，`col_avg_y['beta']`是\\(\lambda(n_i)\\),`col_avg_y[nf_name]`是\\(S_i\\) 。

##### 训练集转换特征

    def fit_transform(self, X, y):
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new

classification和regression的区别在于先验概率，classification是对不同类别的Y在训练集出现的概率，regression是Y的均值。

##### 测试集特征转换

    def transform(self, X):
        X_new = X.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new

### 实际应用
某公司需要预测店铺销售，为简化问题，假设特征只有店铺名称(ShopID)，城市名称(CityName)，预测目标是销售额(SaleAmount)。
##### 加载数据

    import pandas as pd
    from MeanEncoder import MeanEncoder

    train=pd.read_csv('./train.csv')
    test=pd.read_csv('./train.csv')
    print(train.head(5))
    print('Number of Shop: ',len(train['ShopID'].unique()))
    print('Number of City: ',len(train['CityName'].unique()))
代码运行结果

|ShopID|CityName|SaleAmount
|:-:|:-:|:-:|
|000001|晋中|12.2726|
|000001|晋中|12.8300|
|000001|晋中|12.6036|
|000001|晋中|12.7374|
|000001|晋中|13.5847|

    Number of Shop:  647
    Number of City:  182

我们发现店铺有647个，城市有182个，这些都是高基数的定性特征，所以可以使用上述方法来解决。

##### 高基数定性特征处理

    X_train=train[['ShopID','CityName']].copy()
	y_train=train['SaleAmount'].copy()
    X_test=test[['ShopID','CityName']].copy()
    y_test=test['SaleAmount'].copy()
	me=MeanEncoder(categorical_features=['ShopID','CityName'],n_splits=10,target_type='regression')
	X_train=me.fit_transform(X_train,y_train)
	print(X_train.head())

代码运行结果

|ShopID|CityName|ShopID_pred|CityName_pred|
|:-:|:-:|:-:|:-:|
|000001|晋中|12.3922|12.4891|
|000001|晋中|12.3922|12.4891|
|000001|晋中|12.3922|12.4891|
|000001|晋中|12.3922|12.4891|
|000001|晋中|12.3922|12.4891|

其中ShopID\\(\to\\)ShopID_pred，CityName\\(\to\\)CityName_pred 。

##### 测试集特征转换

    X_test=me.transform(X_test)
    print(X_test.head(5))

代码运行结果

|ShopID|CityName|ShopID_pred|CityName_pred|
|:-:|:-:|:-:|:-:|
|000001|晋中|12.3991|12.4532|
|000001|晋中|12.3991|12.4532|
|000001|晋中|12.3991|12.4532|
|000001|晋中|12.3991|12.4532|
|000001|晋中|12.3991|12.4532|

其中采用10折交叉检验，\\(\lambda(n)\\)应视情况自己设置。