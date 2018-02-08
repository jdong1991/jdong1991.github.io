---
layout:     post
title:      Cross Validation iterators
subtitle:   
date:       2018-02-08
author:     JD
header-img: img/post-jd-cvi.jpg
catalog: true
tags:
    - Cross Validation
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

交叉检验在机器学习项目中起到提高泛化能力的作用。针对不同样本，使用不同抽样方案。

### Cross-validation iterators for i.i.d. data

在机器学习理论中，数据独立同分布是很常见的假设。尽管现实很少见，但是效果还是不错滴。

#### K-fold

将原本分成K份，其中将第i份当做测试集，剩下的当做训练集，其中\\(i\in (1,K)\\)，这样需要训练K次。

下面我们举个K=2的例子

    import numpy as np
    from sklearn.model_selection import KFold
    
    X=['a','b','c','d']
    kf=KFold(n_splits=2)
    for train,test in kf.split(X):
        print(train,test)

代码运行结果

    [2 3] [0 1]
    [0 1] [2 3]

返回的是索引值，其中将X分成2份，一份当训练集，一份当测试集。

#### Repeated K-Fold

RepeatedKFold就是重复使用KFlod的方法，`n_repeats`就是重复的次数。

    import numpy as np
    from sklearn.model_selection import RepeatedKFold
    
    X=np.array([[1,2],[3,4],[1,2],[3,4]])
    random_state=12883823
    rkf=RepeatedKFold(n_splits=2,n_repeats=2,random_state=random_state)
    for train,test in rkf.split(X):
        print(train,test)

代码运行结果

    [2 3] [0 1]
    [0 1] [2 3]
    [0 2] [1 3]
    [1 3] [0 2]

#### Leave One Out (LOO)

LOO就是单样本的交叉检验，每回只把一个当做测试集，剩下都为训练集，这样训练次数就是总样本数。

    from sklearn.model_selection import LeaveOneOut
    
    X=[1,2,3,4]
    loo=LeaveOneOut()
    for train,test in loo.split(X):
        print(train,test)

代码运行结果

    [1 2 3] [0]
    [0 2 3] [1]
    [0 1 3] [2]
    [0 1 2] [3]

#### Leave P Out (LPO)

LPO与LOO类似，就是选择p个样本当做测试集，其他当做训练集。但是它的测试集可能会有重复，由于我们会选择\\(\binom{n}{p}\\)个测试集。

    import numpy as np
    from sklearn.model_selection import LeavePOut
    
    X=np.ones(4)
    lpo=LeavePOut(p=2)
    for train,test in lpo.split(X):
        print(train,test)

代码运行结果

    [2 3] [0 1]
    [1 3] [0 2]
    [1 2] [0 3]
    [0 3] [1 2]
    [0 2] [1 3]
    [0 1] [2 3]

#### Shuffle & Split

ShuffleSplit是先打乱数据集，然后分成一对训练集和测试集。

    import numpy as np
    from sklearn.model_selection import ShuffleSplit
    
    X=np.ones(5)
    ss=ShuffleSplit(n_splits=3,test_size=0.25,random_state=0)
    for train,test in ss.split(X):
        print(train,test)

代码运行结果

    [1 3 4] [2 0]
    [1 4 3] [0 2]
    [4 0 2] [1 3]

### Cross-validation iterators with stratification based on class labels

有许多分类问题中，存在目标类别样本的不均衡现象，这时可以根据目标类别分层处理。

#### Stratified k-fold

StratifiedKFold是一种KFold的变体，不过它会把目标类别考虑进去。

    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    
    X=np.ones(10)
    y=[0,0,0,0,1,1,1,1,1,1]
    skf=StratifiedKFold(n_splits=3)
    for train,test in skf.split(X,y):
        print(train,test)

代码运行结果

    [2 3 6 7 8 9] [0 1 4 5]
    [0 1 3 4 5 8 9] [2 6 7]
    [0 1 2 4 5 6 7] [3 8 9]

#### Stratified Shuffle Split

StratifiedShuffleSplit是ShuffleSplit的变体。在测试集和训练集中，尽量保持每个目标类的百分比和原数据集相同。

    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    
    X= np.array([[1,2],[3,4],[1,2],[3,4]])
    y =np.array([0,0,1,1])
    sss=StratifiedShuffleSplit(n_splits=3,test_size=0.5,random_state=0)
    for train_index, test_index in sss.split(X, y):
       print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]

代码运行结果

    TRAIN: [1 2] TEST: [3 0]
    TRAIN: [0 2] TEST: [1 3]
    TRAIN: [0 2] TEST: [3 1]

### Cross-validation iterators for grouped data

如果样本之间存在依赖，那么就不可以假设为独立同分布，分组数据是个不错的选择。

#### Group k-fold

GroupKFold是基于分组的KFold。

    from sklearn.model_selection import GroupKFold
    
    X = [0.1,0.2,2.2,2.4,2.3,4.55,5.8,8.8,9,10]
    y = ['a','b','b','b','c','c','c','d','d','d']
    groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    gkf=GroupKFold(n_splits=3)
    for train, test in gkf.split(X, y, groups=groups):
        print(train, test)

代码运行结果

    [0 1 2 3 4 5] [6 7 8 9]
    [0 1 2 6 7 8 9] [3 4 5]
    [3 4 5 6 7 8 9] [0 1 2]

#### Leave One Group Out

LeaveOneGroupOut是基于分组的LeaveOneOut。

    from sklearn.model_selection import LeaveOneGroupOut
    
    X=[1,5,10,50,60,70,80]
    y=[0,1,1,2,2,2,2]
    groups=[1,1,2,2,3,3,3]
    logo=LeaveOneGroupOut()
    for train,test in logo.split(X,y,groups=groups):
        print(train,test)

代码运行结果

    [2 3 4 5 6] [0 1]
    [0 1 4 5 6] [2 3]
    [0 1 2 3] [4 5 6]

#### Leave P Groups Out

LeavePGroupsOut基于分组的LeavePsOut。

    from sklearn.model_selection import LeavePGroupsOut
    
    X=[1,5,10,50,60,70,80]
    y=[0,1,1,2,2,2,2]
    groups=[1,1,2,2,3,3,3]
    lpgo=LeavePGroupsOut(n_groups=2)
    for train,test in lpgo.split(X,y,groups=groups):
        print(train,test)

代码运行结果

    [4 5 6] [0 1 2 3]
    [2 3] [0 1 4 5 6]
    [0 1] [2 3 4 5 6]

#### Group Shuffle Split

GroupShuffleSplit是ShuffleSplit和LeavePGroupsOut的组合。

    from sklearn.model_selection import GroupShuffleSplit
    
    X=[0.1,0.2,2.2,2.4,2.3,4.55,5.8,0.001]
    y=['a','b','b','b','c','c','c','a']
    groups=[1,1,2,2,3,3,4,4]
    gss=GroupShuffleSplit(n_splits=4,test_size=0.5,random_state=0)
    for train,test in gss.split(X,y,groups=groups):
        print(train,test)

代码运行结果

    [0 1 2 3] [4 5 6 7]
    [2 3 6 7] [0 1 4 5]
    [2 3 4 5] [0 1 6 7]
    [4 5 6 7] [0 1 2 3]

### Predefined Fold-Splits / Validation-Sets

PredefinedSplit是预先定义好哪些可以作为交叉检验的样本，并可以给样本编号，按照样本编号来分割样本。其中test_fold为-1的时候，该样本只可以做为训练集样本。下面是分为了2-fold，因为test_fold只有0和1。

    import numpy as np
    from sklearn.model_selection import PredefinedSplit
    
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([0, 0, 1, 1])
    test_fold = [0, 1, -1, 1]
    ps = PredefinedSplit(test_fold)
    ps.get_n_splits()
    for train_index, test_index in ps.split():
       print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]

代码运行结果

    TRAIN: [1 2 3] TEST: [0]
    TRAIN: [0 2] TEST: [1 3]

### Cross validation of time series data

对于时间序列的数据，我们有专门的分割方法。

#### Time Series Split

TimeSeriesSplit是基于时间序列的KFold。由于时间数据可能和以往有关联，因此，我们采用递增的数据集，第二个训练集集是第一个训练集和测试集的并集，以此类推。

    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit
    
    X=np.array([[1,2],[3,4],[1,2],[3,4],[1,2],[3,4]])
    y=np.array([1,2,3,4,5,6])
    tss=TimeSeriesSplit(n_splits=3)
    for train,test in tss.split(X,y):
        print(train,test)

代码运行结果

    [0 1 2] [3]
    [0 1 2 3] [4]
    [0 1 2 3 4] [5]