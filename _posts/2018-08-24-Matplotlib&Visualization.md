---
layout:     post
title:      Matplotlib & Visualization
subtitle:   
date:       2018-08-24
author:     JD
header-img: img/post-jd-mat.jpg
catalog: true
tags:
    - Matplotlib
    - Visualization
---

Python的`matplotlib`可以满足我们大部分可视化需求。年纪越来越大，记忆力不如以前了，记录下常用的一些操作。

### Simple plot

#### Using defaults

全部采取默认，不设置其他参数。

    import numpy as np
    import matplotlib.pyplot as plt
    
    X=np.arange(-np.pi,np.pi,0.01)
    C,S=np.cos(X),np.sin(X)
    plt.plot(X,C,X,S)
    plt.show()

效果如下图所示

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fv6znraofkj30cs09lglx.jpg)


#### Instantiating defaults

`plt.figure`：设置图片尺寸、像素

`plt.plot`：设置图形颜色、线条宽度、线条风格

`plt.xticks`、`plt.yticks`：设置x、y轴的刻度值

    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8,6),dpi=80)
    plt.subplot(111)
    X=np.linspace(-np.pi,np.pi,256,endpoint=True)
    C,S=np.cos(X),np.sin(X)
    
    plt.plot(X,C,color='blue',linewidth=1.0,linestyle='-')
    plt.plot(X,S,color='green',linewidth=1.0,linestyle='-')
    
    plt.xlim(-4.0,4.0)
    plt.xticks(np.linspace(-4,4,9,endpoint=True))
    
    plt.ylim(-1.0,1.0)
    plt.yticks(np.linspace(-1,1,5,endpoint=True))
    
    plt.show()

效果如下图所示

![](http://wx3.sinaimg.cn/mw690/006F1DTzgy1fv6zqa0lgej30g00c0wes.jpg)

#### Changing colors and line widths

通过`color`和`linewidth`来改变颜色和宽度

    ...
    plt.figure(figsize=(10,6),dpi=80)
    plt.plot(C,color='blue',linewidth=4.5,linestyle='-')
    plt.plot(S,color='red',linewidth=4.5,linestyle='-')
    ...

效果如下图所示

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fv6zu9y823j30g00a0dg3.jpg)

#### Setting limits

`plt.xlim`、`plt.ylim`：控制x、y轴的范围

    ...
    plt.xlim(X.min()*1.1,X.max()*1.1)
    plt.ylim(C.min()*1.1,C.max()*1.1)
    ...

效果如下图所示

![](http://wx2.sinaimg.cn/mw690/006F1DTzgy1fv6zwustimj30g00a074m.jpg)

#### Setting ticks

改变x、y轴的实例

    ...
    plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    plt.yticks([-1,0,+1])
    ...

效果如下图所示

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fv6zzaiyovj30g00a0glv.jpg)

#### Setting tick labels

`plt.xticks`和`plt.yticks`在修改刻度的同时，也可以修改刻度显示值，只需后面再加一个list，可以使用LaTex表示

    ...
    plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],[r'$-\pi$',r'$-\pi/2$',r'$0$',r'$+\pi/2$',r'$+\pi$'])
    plt.yticks([-1,0,+1],[r'$-1$',r'$0$',r'$+1$'])
    ...

效果如下图所示

![](http://wx1.sinaimg.cn/mw690/006F1DTzgy1fv702jnkcvj30g00a03yq.jpg)

#### Moving spines

改变轴的边界，包括隐藏和移动

    ...
    ax=plt.subplot(111)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    ...

效果如下图所示

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fv705ergznj30g00a03yq.jpg)

#### Adding a legend

图例时必不可少的，可以很好的标识出图中信息，在`plt.plot`中加上`label`，然后`plt.legend()`

其中`loc`是代表图例所放的位置，`frameon`是代表图例是否需要边框

    ...
    plt.plot(X,C,color='blue',linewidth=2.5,linestyle='-',label='cosine')
    plt.plot(X,S,color='red',linewidth=2.5,linestyle='-',label='sine')
    plt.legend(loc='upper left',frameon=False)
    ...

效果如下图所示

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fv70allp35j30g00a00sz.jpg)

#### Annotate same points

有时候，我们需要标注一些点，可以使用`plt.annotate`，具体参数可以看[源码](https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/text.py)

    ...
    t=2*np.pi/3
    plt.plot([t,t],[0,np.sin(t)],color='red',linewidth=1.5,linestyle='--')
    plt.scatter([t,],[np.sin(t)],50,color='red')
    plt.annotate(r'$sin\frac{2\pi}{3}=\frac{\sqrt{3}}{2}$',xy=(t,np.sin(t)),
                 xytext=(+10,+30),xycoords='data',textcoords='offset points',
                 fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
    plt.plot([t,t],[0,np.cos(t)],color='blue',linewidth=1.5,linestyle='--')
    plt.scatter([t,],[np.cos(t)],50,color='blue')
    plt.annotate(r'$cos\frac{2\pi}{3}=-\frac{1}{2}$',xy=(t,np.cos(t)),
                 xytext=(-90,-50),xycoords='data',textcoords='offset points',
                 fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
    ...

效果如下图所示

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fv70ejxmndj30g00a0q3b.jpg)

#### Devil is in the details

有点被遮挡的文字需要虚化，但尴尬的是高版本的matplotlib并没有生效

    ...
    for label in ax.xticklabels()+ax.yticklabels():
        label.fontsize(16)
        label.set_bbox(dict(facecolor='white',edgecolor='none',alpha=0.5))
    ...

效果如下图所示

![](http://wx3.sinaimg.cn/mw690/006F1DTzgy1fv70i9p14bj30g00a0mxk.jpg)


### Figures, Subplots, Axes and Ticks

#### Figures

参数详情

    num : integer or string, optional, default: none
        If not provided, a new figure will be created, and the figure number
        will be incremented. The figure objects holds this number in a `number`
        attribute.
        If num is provided, and a figure with this id already exists, make
        it active, and returns a reference to it. If this figure does not
        exists, create it and returns it.
        If num is a string, the window title will be set to this figure's
        `num`.

    figsize : tuple of integers, optional, default: None
        width, height in inches. If not provided, defaults to rc
        figure.figsize.

    dpi : integer, optional, default: None
        resolution of the figure. If not provided, defaults to rc figure.dpi.

    facecolor :
        the background color. If not provided, defaults to rc figure.facecolor.

    edgecolor :
        the border color. If not provided, defaults to rc figure.edgecolor.

    frameon : bool, optional, default: True
        If False, suppress drawing the figure frame.

    FigureClass : class derived from matplotlib.figure.Figure
        Optionally use a custom Figure instance.

    clear : bool, optional, default: False
        If True and the figure already exists, then it is cleared.

注意

    If you are creating many figures, make sure you explicitly call "close"
    on the figures you are not using, because this will enable pylab
    to properly clean up the memory.

    rcParams defines the default values, which can be modified in the
    matplotlibrc file

#### Subplots




#### Axes



#### Tick locators



### Animation

使用`animation`实现动画效果，其他和普通作图类似，就是多了个update的操作

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    fig=plt.figure(figsize=(6,6),facecolor='white')
    ax=fig.add_axes([0.005,0.005,.99,.99],frameon=True,aspect=1)
    
    n=50
    size_min=50
    size_max=50*50
    
    P=np.random.uniform(0,1,(n,2))
    S=np.linspace(size_min,size_max,n)
    C=np.zeros((n,4))
    C[:,3] = np.linspace(0,1,n)
    scat=ax.scatter(P[:,0],P[:,1],s=S,linewidths=0.5,edgecolors=C,facecolors='None')
    ax.set_xlim(0,1),ax.set_ylim(0,1)
    ax.set_xticks([]),ax.set_yticks([])
    
    def update(frame):
        global P,C,S
        
        C[:,3]=np.maximum(0,C[:,3]-1/n)
        S+=(size_max-size_min)/n
        i=frame%50
        P[i]=np.random.uniform(0,1,2)
        S[i]=size_min
        C[i,3]=1
        
        scat.set_edgecolors(C)
        scat.set_sizes(S)
        scat.set_offsets(P)   
        return scat
    
    animation=FuncAnimation(fig,update,interval=10)
    plt.show()

效果如下图所示

![](http://wx3.sinaimg.cn/mw690/006F1DTzgy1fv6zf9cizwg30go0gox6t.gif)

### Other Types of Plots

#### Regular Plots


#### Scatter Plots


#### Bar Plots


#### Contour Plots



#### Imshow



#### Pie Charts



#### Quiver Plots



#### Grids


#### Multi Plots


#### Polar Axis



#### 3D Plots



#### Text

