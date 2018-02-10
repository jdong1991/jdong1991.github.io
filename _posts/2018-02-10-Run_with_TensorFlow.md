---
layout:     post
title:      Run with TensorFlow
subtitle:   
date:       2018-02-10
author:     JD
header-img: img/post-jd-tf.jpg
catalog: true
tags:
    - TensorFlow
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

### Linear Regression with TensorFlow

根据正常的线性回归公式，如下

$$\hat{\theta}=(X^T\cdot X)^{-1}\cdot X^T \cdot y$$

可用TensorFlow来实现

    import numpy as np
    from sklearn.datasets import fetch_california_housing
    import tensorflow as tf
    
    housing=fetch_california_housing()
    m,n=housing.data.shape
    housing_data_plus_bias=np.c_[np.ones((m,1)),housing.data]
    X=tf.constant(housing_data_plus_bias,dtype=tf.float32,name='X')
    y=tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name='y')
    XT=tf.transpose(X)
    theta=tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)
    
    with tf.Session() as sess:
        theta_value=theta.eval()
        
    print(theta_value)

代码运行结果

    [[-3.7465141e+01]
     [ 4.3573415e-01]
     [ 9.3382923e-03]
     [-1.0662201e-01]
     [ 6.4410698e-01]
     [-4.2513184e-06]
     [-3.7732250e-03]
     [-4.2664889e-01]
     [-4.4051403e-01]]

### Gradient Descent

在线性回归问题中，我们也可以使用梯度下降法。在`TensorFlow`中，可以使用以下三种常见的方法进行计算。
- 人工自行设计计算梯度
- 使用autodiff来计算梯度
- 通过优化器来计算梯度

#### Manually Computing the Gradient

这种方法自由度很高，拥有很高的自主性。

    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    
    n_epochs=1000
    learning_rate=0.01
    
    housing=fetch_california_housing()
    m,n=housing.data.shape
    scaled_housing_data=StandardScaler().fit_transform(housing.data)
    scaled_housing_data_plus_bias=np.c_[np.ones((m,1)),scaled_housing_data]
    X=tf.constant(scaled_housing_data_plus_bias,dtype=tf.float32,name='X')
    y=tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name='y')
    theta=tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0,seed=42),name='theta')
    y_pred=tf.matmul(X,theta,name='predictions')
    error=y_pred-y
    mse=tf.reduce_mean(tf.square(error),name='mse')
    gradients=2/m * tf.matmul(tf.transpose(X),error)
    training_op=tf.assign(theta,theta-learning_rate*gradients)
    
    init=tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(n_epochs):
            if epoch%100==0:
                print('epoch:%d,mse:%.4f' % (epoch,mse.eval()))
            sess.run(training_op)
        print('best_theta: \n',theta.eval())

代码运行结果

    epoch:0,mse:2.7544
    epoch:100,mse:0.6322
    epoch:200,mse:0.5728
    epoch:300,mse:0.5585
    epoch:400,mse:0.5491
    epoch:500,mse:0.5423
    epoch:600,mse:0.5374
    epoch:700,mse:0.5338
    epoch:800,mse:0.5312
    epoch:900,mse:0.5294
    best_theta: 
     [[ 2.06855226e+00]
     [ 7.74078071e-01]
     [ 1.31192401e-01]
     [-1.17845066e-01]
     [ 1.64778143e-01]
     [ 7.44081335e-04]
     [-3.91945131e-02]
     [-8.61356676e-01]
     [-8.23479772e-01]]

#### Using autodiff

主要有四种方法来自动计算梯度，如下图所列。

![](http://wx2.sinaimg.cn/mw690/006F1DTzgy1fob6jhqtk6j30t906v75j.jpg)

在TensorFlow中，使用的是`Reverse-mode autodiff`。

只需将`gradients`修改成`gradients=tf.gradients(mse,[theta])[0]`。

将下面代码：

    gradients=2/m * tf.matmul(tf.transpose(X),error)

修改为

    gradients=tf.gradients(mse,[theta])[0]

修改后代码运行结果

    epoch:0,mse:2.7544
    epoch:100,mse:0.6322
    epoch:200,mse:0.5728
    epoch:300,mse:0.5585
    epoch:400,mse:0.5491
    epoch:500,mse:0.5423
    epoch:600,mse:0.5374
    epoch:700,mse:0.5338
    epoch:800,mse:0.5312
    epoch:900,mse:0.5294
    best_theta: 
     [[ 2.06855226e+00]
     [ 7.74078071e-01]
     [ 1.31192386e-01]
     [-1.17845066e-01]
     [ 1.64778143e-01]
     [ 7.44077959e-04]
     [-3.91945131e-02]
     [-8.61356676e-01]
     [-8.23479772e-01]]

#### Using an Optimizer

TensorFlow也提供了优化器，可以更好帮你计算梯度问题。

我们可以使用梯度下降优化器，也可以使用动量优化器，还有adam等优化器。

只需修改一部分代码。

将

    gradients=tf.gradients(mse,[theta])[0]
    training_op=tf.assign(theta,theta-learning_rate*gradients)

修改为

    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(mse)

修改后代码运行结果

    epoch:0,mse:2.7544
    epoch:100,mse:0.6322
    epoch:200,mse:0.5728
    epoch:300,mse:0.5585
    epoch:400,mse:0.5491
    epoch:500,mse:0.5423
    epoch:600,mse:0.5374
    epoch:700,mse:0.5338
    epoch:800,mse:0.5312
    epoch:900,mse:0.5294
    best_theta: 
     [[ 2.06855226e+00]
     [ 7.74078071e-01]
     [ 1.31192386e-01]
     [-1.17845066e-01]
     [ 1.64778143e-01]
     [ 7.44078017e-04]
     [-3.91945131e-02]
     [-8.61356676e-01]
     [-8.23479772e-01]]

### Mini-batch Gradient Descent

`Mini-batch Gradient Descent`的TensorFlow实现。

    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    
    n_epochs=10
    learning_rate=0.01
    
    housing=fetch_california_housing()
    m,n=housing.data.shape
    scaled_housing_data=StandardScaler().fit_transform(housing.data)
    scaled_housing_data_plus_bias=np.c_[np.ones((m,1)),scaled_housing_data]
    X=tf.placeholder(dtype=tf.float32,shape=(None,n+1),name='X')
    y=tf.placeholder(dtype=tf.float32,shape=(None,1),name='y')
    theta=tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0,seed=42),name='theta')
    y_pred=tf.matmul(X,theta,name='predictions')
    error=y_pred-y
    mse=tf.reduce_mean(tf.square(error),name='mse')
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(mse)
    
    batch_size=100
    n_batchs=int(np.ceil(m/batch_size))
    
    def fetch_batch(epoch,batch_index,batch_size):
        np.random.seed(epoch*n_batchs+batch_index)
        indices=np.random.randint(m,size=batch_size)
        X_batch=scaled_housing_data_plus_bias[indices]
        y_batch=housing.target.reshape(-1,1)[indices]
        return X_batch,y_batch
    
    init=tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(n_epochs):
            for batch_index in range(n_batchs):
                X_batch,y_batch=fetch_batch(epoch,batch_index,batch_size)
                if batch_index==n_batchs-1:
                    print('epoch:%d,mse:%.4f' % (epoch,sess.run(mse,feed_dict={X:X_batch,y:y_batch})))
                sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        print('best_theta:\n',theta.eval())
            
代码实现结果

    epoch:0,mse:0.5039
    epoch:1,mse:0.5963
    epoch:2,mse:0.5058
    epoch:3,mse:0.6501
    epoch:4,mse:0.4005
    epoch:5,mse:0.5019
    epoch:6,mse:0.3570
    epoch:7,mse:0.3966
    epoch:8,mse:0.6940
    epoch:9,mse:0.3803
    best_theta:
     [[ 2.070016  ]
     [ 0.82045615]
     [ 0.11731729]
     [-0.22739057]
     [ 0.31134027]
     [ 0.00353193]
     [-0.01126994]
     [-0.91643935]
     [-0.8795008 ]]

### Visualizing Using TensorBoard

`TensorFlow`的可视化工具`TensorBoard`，可以在浏览器观察模型的流程及数据。

改造`Mini-Batch Gradient Descent`的代码，实现`TensorBoard`可视化。

    from datetime import datetime
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    
    
    tf.reset_default_graph()
    
    now=datetime.now().strftime('%Y%m%d%H%M%S')
    root_logdir='D:\\home\\tf_log'
    logdir='{}\\run-{}\\'.format(root_logdir,now)
    
    n_epochs=10
    learning_rate=0.01
    
    housing=fetch_california_housing()
    m,n=housing.data.shape
    scaled_housing_data=StandardScaler().fit_transform(housing.data)
    scaled_housing_data_plus_bias=np.c_[np.ones((m,1)),scaled_housing_data]
    X=tf.placeholder(dtype=tf.float32,shape=(None,n+1),name='X')
    y=tf.placeholder(dtype=tf.float32,shape=(None,1),name='y')
    theta=tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0,seed=42),name='theta')
    y_pred=tf.matmul(X,theta,name='predictions')
    with tf.name_scope('loss') as scope:
        error=y_pred-y
        mse=tf.reduce_mean(tf.square(error),name='mse')
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(mse)
    
    mse_summary=tf.summary.scalar('MSE',mse)
    file_writer=tf.summary.FileWriter(logdir,tf.get_default_graph())
    
    batch_size=100
    n_batchs=int(np.ceil(m/batch_size))
    
    def fetch_batch(epoch,batch_index,batch_size):
        np.random.seed(epoch*n_batchs+batch_index)
        indices=np.random.randint(m,size=batch_size)
        X_batch=scaled_housing_data_plus_bias[indices]
        y_batch=housing.target.reshape(-1,1)[indices]
        return X_batch,y_batch
    
    init=tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(n_epochs):
            for batch_index in range(n_batchs):
                X_batch,y_batch=fetch_batch(epoch,batch_index,batch_size)
                if batch_index%10==0:
                    summary_str=mse_summary.eval(feed_dict={X:X_batch,y:y_batch})
                    step=epoch*n_batchs+batch_index
                    file_writer.add_summary(summary_str,step)
                sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        print('best_theta:\n',theta.eval())
    
    file_writer.close()

启动`TensorBoard`。在cmd下，cd到生成的events文件的上一级文件，如本例先进入`D:\home`，然后执行`tensorboard --logdir=D:\home\tf_log`。如下图

![](http://wx1.sinaimg.cn/mw690/006F1DTzgy1fobkt29ljaj30h705h3z3.jpg)

在`tf_log`中文件如下

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fobkvptwtdj30ed070dgh.jpg)

我们可以看到浏览器中SCALARS看到

![](http://wx2.sinaimg.cn/mw690/006F1DTzgy1fobm2fxpmaj31hc0jegof.jpg)

还有在GRAPHS看到

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fobm3ofybbj30tp0myjtw.jpg)

其中可以看到模块内部结构

![](http://wx1.sinaimg.cn/mw690/006F1DTzgy1fobm5e6k7cj30t70kugn4.jpg)