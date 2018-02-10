---
layout:     post
title:      Ensemble Model
subtitle:   
date:       2018-02-09
author:     JD
header-img: img/post-jd-ensemble.jpg
catalog: true
tags:
    - Ensemble
    - Model
---

### Voting Classifiers

所谓三个臭皮匠赛过诸葛亮，这就是投票集成模型的基本。当学习器各自都是独立的，这样它们可能犯不同的错误，这样效果最佳。

#### hard voting

`hard voting`是让每个弱学习器预测，然后进行投票，取获得最多票数的结果。如下图：

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fo94b9bzvjj30vz0hiwhq.jpg)

下面是将逻辑回归，随机森林，支持向量机三种弱学习器使用`VotingClassifier`进行投票集成。

    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import VotingClassifier,RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    X,y=make_moons(n_samples=500,noise=0.30,random_state=42)
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
    log_clf=LogisticRegression(random_state=42)
    rnd_clf=RandomForestClassifier(random_state=42)
    svc_clf=SVC(random_state=42)
    voting_clf=VotingClassifier(estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svc_clf)],voting='hard')
    
    for clf in (log_clf,rnd_clf,svc_clf,voting_clf):
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        print(clf.__class__.__name__,accuracy_score(y_test,y_pred))

代码运行结果

    LogisticRegression 0.864
    RandomForestClassifier 0.872
    SVC 0.888
    VotingClassifier 0.896

#### soft voting

若所有弱学习器都是可以估计类别概率，这时候就可以获取最高概率的类别，这种方法叫`soft voting`。

    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import VotingClassifier,RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    X,y=make_moons(n_samples=500,noise=0.30,random_state=42)
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
    log_clf=LogisticRegression(random_state=42)
    rnd_clf=RandomForestClassifier(random_state=42)
    svc_clf=SVC(probability=True,random_state=42)
    voting_clf=VotingClassifier(estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svc_clf)],voting='soft')
    
    for clf in (log_clf,rnd_clf,svc_clf,voting_clf):
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        print(clf.__class__.__name__,accuracy_score(y_test,y_pred))

代码运行结果

    LogisticRegression 0.864
    RandomForestClassifier 0.872
    SVC 0.888
    VotingClassifier 0.912

发现效果的确很好，又有进一步提升呦。

### Bagging and Pasting

`Bagging`是通过有放回的抽样来获取训练样本，通过训练样本的变化来生成不同的分类器。`Pasting`与`Bagging`类似，但它是通过不放回的抽样来获取训练样本。

![](http://wx1.sinaimg.cn/mw690/006F1DTzgy1fo96ezdnt4j30w30gp0wj.jpg)

使用500个决策树(`n_estimators=500`)，每个分类器使用100个样本(`max_samples=100`)，采取有放回的抽样模式(`bootstrap=True`)，代码如下：

    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import accuracy_score
    
    X,y=make_moons(n_samples=500,noise=0.30,random_state=42)
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
    tree_clf=DecisionTreeClassifier(random_state=42)
    bag_clf=BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42),n_estimators=500,max_samples=100,bootstrap=True,random_state=42)
    
    for clf in (tree_clf,bag_clf):
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        print(clf.__class__.__name__,accuracy_score(y_test,y_pred))

代码运行结果

    DecisionTreeClassifier 0.856
    BaggingClassifier 0.904

我们可以看看DT和Bagging的判定边界，如下图

![](http://wx1.sinaimg.cn/mw690/006F1DTzgy1fo979zifn0j30ix07z3zn.jpg)

可以发现集成模型可能会比决策树模型更具泛化能力。Bagging相比决策树，是增加了偏差，但是减少了方差。上图代码如下：

    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import accuracy_score
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import numpy as np
    
    X,y=make_moons(n_samples=500,noise=0.30,random_state=42)
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
    tree_clf=DecisionTreeClassifier(random_state=42)
    bag_clf=BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42),n_estimators=500,max_samples=100,bootstrap=True,random_state=42)
    
    def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
        x1s = np.linspace(axes[0], axes[1], 100)
        x2s = np.linspace(axes[2], axes[3], 100)
        x1, x2 = np.meshgrid(x1s, x2s)
        X_new = np.c_[x1.ravel(), x2.ravel()]
        y_pred = clf.predict(X_new).reshape(x1.shape)
        custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
        plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap, linewidth=10)
        if contour:
            custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
            plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
        plt.axis(axes)
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    
    plt.figure(figsize=(11,4))
    plt.subplot(121)
    plot_decision_boundary(tree_clf, X, y)
    plt.title("Decision Tree", fontsize=14)
    plt.subplot(122)
    plot_decision_boundary(bag_clf, X, y)
    plt.title("Decision Trees with Bagging", fontsize=14)
    plt.show()

#### Out-of-Bag evaluation

在Bagging模型中，由于采用样本子集，那么有些样本可能就从来没有被抽到过，那么这些没被抽到过的就叫做out-of-bag(oob)样本，用这些样本来进行评估，这就叫oob评估。

    bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42),n_estimators=500,bootstrap=True,oob_score=True, random_state=40)
    bag_clf.fit(X_train, y_train)
    print('oob_score:',bag_clf.oob_score_)

代码运行结果

    oob_score: 0.9013333333333333

#### Random Patchs and Random Subspaces

`Random Patchs`是抽样时，随机抽取训练样本和特征。`Random Subspaces`是抽样时，只随机抽取特征，保持训练全部样本。
在`BaggingClassifier`中

- Random Patchs：bootstrap=True,max_samples<1.0,bootstrap_features=True,max_features<1.0
- Random Subspaces：bootstrap=False,max_samples=1.0,bootstrap_features=True,max_features<1.0

#### Random Forests

随机森林是一个以决策树为弱学习器的集成模型，也是用Bagging方法来实现。

    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    X,y=make_moons(n_samples=500,noise=0.30,random_state=42)
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
    
    bag_clf=BaggingClassifier(DecisionTreeClassifier(splitter='random',max_leaf_nodes=16,random_state=42),n_estimators=500,max_samples=1.0,bootstrap=True,random_state=42)
    bag_clf.fit(X_train,y_train)
    y_pred=bag_clf.predict(X_test)
    
    rnd_clf=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,random_state=42)
    rnd_clf.fit(X_train,y_train)
    y_rnd_pred=rnd_clf.predict(X_test)
    print(np.sum(y_pred==y_rnd_pred)/len(y_pred))

代码运行结果

    0.976

我们发现和BaggingClassifier大致一样。

在随机森林中，有个十分有用的属性，那就是特征重要性(`feature_importances_`)。

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    
    iris=load_iris()
    rnd_clf=RandomForestClassifier(n_estimators=500,random_state=42)
    rnd_clf.fit(iris['data'],iris['target'])
    for name,score in zip(iris['feature_names'],rnd_clf.feature_importances_):
        print(name,score)

代码运行结果

    sepal length (cm) 0.11249225099876374
    sepal width (cm) 0.023119288282510326
    petal length (cm) 0.44103046436395765
    petal width (cm) 0.4233579963547681

可以看出其中`petal length`的影响最大。

#### Extremely Randomized Trees

极端随机树与随机森林算法十分相似，都是由许多决策树构成。但是它们主要有以下两点区别：

- 极端随机树是采用全部训练样本，而不是采用bagging模型。
- 极端随机树的完全随机得到分叉值，而不是通过训练样本得到最优分叉值。

### Boosting

Boosting是通过集中关注被已有分类器错分的那些数据来获得新的分类器。Boosting分类的结果是基于所有分类器的加权求和结果的。

#### AdaBoost

AdaBoost是adaptive boosting的缩写，即自适应boosting。它是将训练数据中的每个样本，并赋予其一贯权重，这些权重构成了向量D。一开始，这些权重都初始化成相等值，首先在训练数据上训练处一贯弱分类器并计算改分类器的错误率，然后在同一数据集上再次训练弱分类器。在分类器的第二次训练当中，将会重新调整每个样本的权重，其中第一次分对的样本的权重将会降低，而第一次分错的样本的权重将会提高。为了从所有弱分类器中得到最终的分类结果，AdaBoost为每个分类器都分配了一个权重值alpha，这些alpha是基于每个弱分类器的错误率进行计算，大致过程如下图。

![](http://wx3.sinaimg.cn/mw690/006F1DTzgy1fo9y0ap79tj30vz0j5wiz.jpg)

下面是AdaBoost的一个例子。

    X,y=make_moons(n_samples=500,noise=0.30,random_state=42)
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
    ada_clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200,algorithm='SAMME.R',learning_rate=0.5,random_state=42)
    ada_clf.fit(X_train,y_train)
    plot_decision_boundary(ada_clf,X,y)

代码运行结果

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fo9z7zcamgj30b607mmxq.jpg)

其中`algorithm='SAMME'`就是AdaBoost，`algorithm='SAMME.R'`是通过分类概率来进行AdaBoost。若出现过拟合问题，可以适当减小`n_estimators`。

#### Gradient Boosting

Gradient Boosting是使用上个分类器得到分类结果的残差来做为当前分类器的训练数据。

##### GBDT

GBDT是一个不断拟合残差并叠加到集成模型上的过程。在这个过程中，残差不断变小，损失函数不断接近最小值。

下面给出原始的GBDT算法框架

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1foa0f1iom0j30xt0j4q5n.jpg)

具体算法推导，可以看[这篇文章](https://pan.baidu.com/s/1jJukVDc)。

在`sklearn`中，已经有现成的算法函数`GradientBoostingClassifier`.

    from sklearn.ensemble import GradientBoostingRegressor
    import matplotlib.pyplot as plt
    import numpy as np
    
    def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
        x1 = np.linspace(axes[0], axes[1], 500)
        y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
        plt.plot(X[:, 0], y, data_style, label=data_label)
        plt.plot(x1, y_pred, style, linewidth=2, label=label)
        if label or data_label:
            plt.legend(loc="upper center", fontsize=16)
        plt.axis(axes)
        
    np.random.seed(42)
    X= np.random.rand(100, 1) - 0.5
    y= 3*X[:, 0]**2 + 0.05 * np.random.randn(100)
    
    gbdt=GradientBoostingRegressor(max_depth=2,n_estimators=3,learning_rate=1.0,random_state=42)
    gbdt.fit(X,y)
    gbdt_slow=GradientBoostingRegressor(max_depth=2,n_estimators=200,learning_rate=0.1,random_state=42)
    gbdt_slow.fit(X,y)
    
    plt.figure(figsize=(11,4))
    plt.subplot(121)
    plot_predictions([gbdt],X,y,axes=[-0.5,0.5,-0.1,0.8],label='Ensemble predictions')
    plt.title('learning rate:{},n_estimators:{}'.format(gbdt.learning_rate,gbdt.n_estimators),fontsize=14)
    plt.subplot(122)
    plot_predictions([gbdt_slow],X,y,axes=[-0.5,0.5,-0.1,0.8])
    plt.title('learning rate:{},n_estimators:{}'.format(gbdt.learning_rate,gbdt.n_estimators),fontsize=14)
    plt.show()

代码运行结果

![](http://wx3.sinaimg.cn/mw690/006F1DTzgy1foa1xamh8lj30ie07ddga.jpg)

##### XGBoost

XGBoost是GB算法的高效实现。XGBoost的基学习器可以是CART(gbtree)，也可以是线性分类器(gblinear)。详细实现看这篇[论文](https://pan.baidu.com/s/1dHhek29)。

XGBoost的安装可以根据自己的配置，安装相应的CPU或GPU版本，详情可以进[官网](http://xgboost.readthedocs.io/en/latest/build.html)查看。在windows下自己编译很麻烦，也可以去[这个网站](http://ssl.picnet.com.au/xgboost/)下载`xgboost.dll`文件，放到源文件目录的`python-package\xgboost`目录，再执行`python setup.py install`即可。

    import numpy as np    
    import xgboost as xgb    
    
    #load Data
    data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
    label = np.random.randint(2, size=5)  # binary target
    dtrain = xgb.DMatrix(data, label=label)
    dtest=dtrain
    
    #Setting Parameters
    param = {'max_depth': 2,'eta': 1,'silent': 1,'objective': 'binary:logistic',
             'nthread':4, 'eval_metric':'auc','tree_method':'gpu_hist'}
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    
    #Training
    num_round = 10
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    
    #Prediction
    data = np.random.rand(7, 10)
    dtest = xgb.DMatrix(data)
    ypred = bst.predict(dtest)

代码运行结果

    [0]     eval-auc:0.5    train-auc:0.5
    [1]     eval-auc:0.5    train-auc:0.5
    [2]     eval-auc:0.5    train-auc:0.5
    [3]     eval-auc:0.5    train-auc:0.5
    [4]     eval-auc:0.5    train-auc:0.5
    [5]     eval-auc:0.5    train-auc:0.5
    [6]     eval-auc:0.5    train-auc:0.5
    [7]     eval-auc:0.5    train-auc:0.5
    [8]     eval-auc:0.5    train-auc:0.5
    [9]     eval-auc:0.5    train-auc:0.5

##### LightGBM

LightGBM是微软推出的一个基于决策树算法的分布式梯度提升框架。详情看[LightGBM官网](http://lightgbm.readthedocs.io/en/latest/index.html)。

    import numpy as np
    import lightgbm as lgb
    
    #load data
    data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
    label = np.random.randint(2, size=500)  # binary target
    train_data = lgb.Dataset(data, label=label)
    test_data=train_data
    
    #Setting Parameters
    param = {'num_leaves':31, 'objective':'binary','metric':'auc'}
    
    #Training
    #num_round = 10
    #bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
    
    #CV
    num_round = 100
    #lgb.cv(param, train_data, num_round, nfold=5)
    
    #Early Stopping
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=5)
    #bst.save_model('model.txt', num_iteration=bst.best_iteration)
    
    #Prediction
    data = np.random.rand(7, 10)
    ypred = bst.predict(data)

代码运行结果

    [1]     training's auc: 0.740289
    Training until validation scores don't improve for 5 rounds.
    [2]     training's auc: 0.773201
    [3]     training's auc: 0.805271
    [4]     training's auc: 0.84728
    [5]     training's auc: 0.875551
    [6]     training's auc: 0.87814
    [7]     training's auc: 0.889314
    [8]     training's auc: 0.909689
    [9]     training's auc: 0.918699
    [10]    training's auc: 0.928918
    [11]    training's auc: 0.938826
    [12]    training's auc: 0.944565
    [13]    training's auc: 0.952244
    [14]    training's auc: 0.954937
    [15]    training's auc: 0.958415
    [16]    training's auc: 0.963529
    [17]    training's auc: 0.96821
    [18]    training's auc: 0.972074
    [19]    training's auc: 0.977508
    [20]    training's auc: 0.981532
    [21]    training's auc: 0.983472
    [22]    training's auc: 0.985107
    [23]    training's auc: 0.987255
    [24]    training's auc: 0.98865
    [25]    training's auc: 0.989981
    [26]    training's auc: 0.991455
    [27]    training's auc: 0.993331
    [28]    training's auc: 0.995014
    [29]    training's auc: 0.995736
    [30]    training's auc: 0.996569
    [31]    training's auc: 0.996842
    [32]    training's auc: 0.997675
    [33]    training's auc: 0.998285
    [34]    training's auc: 0.998589
    [35]    training's auc: 0.998734
    [36]    training's auc: 0.99899
    [37]    training's auc: 0.999198
    [38]    training's auc: 0.999263
    [39]    training's auc: 0.999519
    [40]    training's auc: 0.999551
    [41]    training's auc: 0.999615
    [42]    training's auc: 0.999776
    [43]    training's auc: 0.999744
    [44]    training's auc: 0.999952
    [45]    training's auc: 0.999968
    [46]    training's auc: 0.999984
    [47]    training's auc: 0.999984
    [48]    training's auc: 1
    [49]    training's auc: 1
    [50]    training's auc: 1
    [51]    training's auc: 1
    [52]    training's auc: 1
    [53]    training's auc: 1
    Early stopping, best iteration is:
    [48]    training's auc: 1

### Stacking

Stacking是先通过学习器对训练集进行预测，将预测值作为下一层学习器的输入值进行预测。比如，一个学习器将对训练集分成5份，将4份训练，1份预测，然后依次做5次，这样就得到了一列和元数据集相同样本数的预测值，每个学习器都如此运作，有N个学习器就可以得到N列，这N列就是下一层的输入值。

下图是底部有3个学习器，分别得到(3.1，2.7，2.9)，这这些值将作为元学习器的输入值，最终预测出3.0。

![](http://wx2.sinaimg.cn/mw690/006F1DTzgy1foaacaaxo1j30w10i1wge.jpg)

python实现代码如下

    class StackingModel(BaseEstimator,RegressorMixin,TransformerMixin):
        def __init__(self,base_models,stack_model):
            self.base_models=base_models
            self.stack_model=stack_model
        
        def fit(self,X,y):
            self.base_models_=[list() for x in self.base_models]
            self.stack_model_=clone(self.stack_model)
            newTrain=np.zeros((X.shape[0],len(self.base_models)))
            kf=KFold(n_splits=5,shuffle=True,random_state=156)
            for i,model in enumerate(self.base_models):
                for train_index,predict_index in kf.split(X,y):
                    m=clone(model)
                    self.base_models_[i].append(m)
                    m.fit(X[train_index],y[train_index])
                    pred_y=m.predict(X[predict_index])
                    newTrain[predict_index,i]=pred_y
            self.stack_model_.fit(newTrain,y)
            return self
        
        def predict(self,X):
            pred=np.column_stack([np.column_stack([model.predict(X) for model in models]).mean(axis=1) for models in self.base_models_])
            return self.stack_model_.predict(pred)

本文主要参考`Hands-ON Machine Learning with Scikit-Learn&TensorFlow`的第七章。