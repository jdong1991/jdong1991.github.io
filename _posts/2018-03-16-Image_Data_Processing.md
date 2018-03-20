---
layout:     post
title:      Image Data Processing
subtitle:   
date:       2018-03-16
author:     JD
header-img: img/post-jd-idp.jpg
catalog: true
tags:
    - TFRecord
    - image
---

图像数据如何输入到神经网络，如何图像数据规范化，如何人为增加图像数据，如何解决增加图像数据后产生效率下降问题。

下面引用[《TensorFlow：实战Google深度学习框架（第2版）》](https://item.jd.com/12287533.html)第七章内容来解决上述问题。

### TFRecord输入数据格式

使用词典来维护图像和类别的关系的可扩展性非常差，当数据来源更加复杂、信息更加丰富，就很难有效记录输入数据中的信息。于是TensorFlow提供了TFRecord的格式来统一存储数据。

将MNIST数据集写入到TFRecord文件

	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	import numpy as np
	
	def _int64_feature(value):
	    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	
	def _bytes_feature(value):
	    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	
	mnist=input_data.read_data_sets('../../datasets/MNIST_data',dtype=tf.uint8,one_hot=True)
	images=mnist.train.images
	labels=mnist.train.labels
	pixels=images.shape[1]
	num_examples=mnist.train.num_examples
	
	filename='Records/output.tfrecords'
	writer=tf.python_io.TFRecordWriter(filename)
	for index in range(num_examples):
	    image_raw=images[index].tostring()
	    example=tf.train.Example(features=tf.train.Features(feature={'pixels':_int64_feature(pixels),
	                                                                'label':_int64_feature(np.argmax(labels[index])),
	                                                                'image_raw':_bytes_feature(image_raw)}))
	    writer.write(example.SerializeToString())
	writer.close()
	print('TFRecord文件已经存在')

这时候就会在`Records`文件夹中生成`output.tfrecords`

![](http://wx1.sinaimg.cn/mw690/006F1DTzgy1fpeoirftbcj30gu02ggli.jpg)

下面我们需要从TFRecord文件读取数据

    import tensorflow as tf
    
    reader=tf.TFRecordReader()
    filename_queue=tf.train.string_input_producer(['Records/output.tfrecords'])
    _,serialized_example=reader.read(filename_queue)
    
    features=tf.parse_single_example(serialized_example,features={'image_raw':tf.FixedLenFeature([],tf.string),
                                                                  'pixels':tf.FixedLenFeature([],tf.int64),
                                                                  'label':tf.FixedLenFeature([],tf.int64)})
    images=tf.decode_raw(features['image_raw'],tf.uint8)
    labels=tf.cast(features['label'],tf.int32)
    pixels=tf.cast(features['pixels'],tf.int32)
    
    sess=tf.Session()
    
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(10):
        image,label,pixel=sess.run([images,labels,pixels])

### 图像数据处理

#### 读取图片

TensorFlow提供了对jpeg和png格式图像的编码/解码函数。可以将一张图像还原成一个三维矩阵。

    import matplotlib.pyplot as plt
    import tensorflow as tf
    import numpy as np
    
    image_raw_data=tf.gfile.FastGFile('../../datasets/cat.jpg','rb').read()
    img_data=tf.image.decode_jpeg(image_raw_data)
    
    with tf.Session() as sess:
        img_data=tf.image.decode_jpeg(image_raw_data)
        
    #    print(img_data.eval())
        img_data.set_shape([1797,2673,3])
        print(img_data.get_shape())
        
        plt.imshow(img_data.eval())
        plt.show()

代码运行结果

    (1797, 2673, 3)

#### 打印图片

根据读取图片数据打印出来

    with tf.Session() as sess: 
        plt.imshow(img_data.eval())
        plt.show()

代码运行结果

![](http://wx1.sinaimg.cn/mw690/006F1DTzgy1fpeqfr7754j30ad07078b.jpg)

#### 调整图片大小

神经网络输入节点的个数是固定的，就需要先将图像的大小统一。图像大小调整有两种方式，第一种是通过算法使得新的图片尽量保存原始图像上的所有信息。TensorFlow提供了四种不同的方法，并且将它们封装到了`tf.image.resize_iamges`.

|Method取值|图像大小调整算法|
|:-|:-|
|0|Bilinear interpolation|
|1|Nearest neighbor interpolation|
|2|Bicubic interpolation|
|3|Area interpolation|


    with tf.Session() as sess:
        for i in range(4):  
            resized=tf.image.resize_images(img_data,[300,300],method=i)
            print('Digital type: ',resized.dtype)
            cat=np.asarray(resized.eval(),dtype='uint8')
            plt.subplot(2,2,i+1)
            plt.tight_layout()
            plt.title(i)
            plt.imshow(cat)
        plt.show()

代码运行结果

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fpevm8vk30j309m07swgr.jpg)

从上图可以看出，不同算法调整出来的结果会有细微差别，但不会相差太远。

#### 裁剪和填充图片

TensorFlow还提供了API对图像进行裁剪或者填充，通过`tf.image.resize_image_with_crop_or_pad`来实现。

	with tf.Session() as sess:
	    croped=tf.image.resize_image_with_crop_or_pad(img_data,1000,1000)
	    padded=tf.image.resize_image_with_crop_or_pad(img_data,3000,3000)
	    plt.subplot(121)
	    plt.imshow(croped.eval())
	    plt.subplot(122)
	    plt.tight_layout()
	    plt.imshow(padded.eval())
	    plt.show()

代码运行结果

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fpexv9pi8vj30bx05hgo9.jpg)

TensorFlow还支持通过比例调整图像大小

	with tf.Session() as sess:
	    central_croped=tf.image.central_crop(img_data,0.5)
	    plt.imshow(central_croped.eval())
	    plt.show()

代码运行结果

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fpexy02qvtj30a7070aee.jpg)

上面介绍的图像裁剪函数都是截取或者填充图像中间部分。TensorFlow也提供了`tf.image.crop_to_bounding_box`函数和`tf.image.pad_to_bounding_box`函数来裁剪或者填充给定区域的图像。这两个函数都要求给出的尺寸满足一定的要求，否则程序会报错。比如使用`tf.image.crop_to_bounding_box`函数是，TensorFlow要求提供的图像尺寸要大于目标尺寸，也就是要求原始图片能够裁剪出目标图像的大小。

#### 翻转图片

TensorFlow支持将图像上下翻转、左右翻转以及沿对角线翻转的功能。

	with tf.Session() as sess:
	    flipped1=tf.image.flip_left_right(img_data)
	    flipped2=tf.image.flip_up_down(img_data)
	    transpose_img=tf.image.transpose_image(img_data)
	    
	    for i,img in enumerate([flipped1,flipped2,transpose_img]):
	        plt.subplot(1,3,i+1)
	        plt.tight_layout()
	        plt.imshow(img.eval())
	    plt.show()

代码运行结果

![](http://wx1.sinaimg.cn/mw690/006F1DTzgy1fpey9at825j30bo04sq4f.jpg)

在很多图像识别问题中，图像的翻转不会影响识别的结果。于是TensorFlow也支持随机的翻转图像。

    with tf.Session() as sess:
        flipped1=tf.image.random_flip_left_right(img_data)
        flipped2=tf.image.random_flip_up_down(img_data)
        
        for i,img in enumerate([flipped1,flipped2]):
            plt.subplot(1,2,i+1)
            plt.tight_layout()
            plt.imshow(img.eval())
        plt.show()

代码运行结果

![](http://wx1.sinaimg.cn/mw690/006F1DTzgy1fpfmc1p5b5j30bo04340o.jpg)

#### 图片色彩调整

TensorFlow提供了调整图像色彩相应属性的API。以下代码是调整图像的亮度和对比度。

    with tf.Session() as sess:
        adjusted1=tf.image.adjust_brightness(img_data,0.5)
        adjusted2=tf.image.adjust_brightness(img_data,-0.5)
        adjusted3=tf.image.random_brightness(img_data,max_delta=0.5)
        adjusted4=tf.image.adjust_contrast(img_data,-5)
        adjusted5=tf.image.adjust_contrast(img_data,5)
        adjusted6=tf.image.random_contrast(img_data,1, 5)
        
        plt.figure(figsize=(15,8))
        for i,img in enumerate([adjusted1,adjusted2,adjusted3,adjusted4,adjusted5,adjusted6]):
            
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            plt.imshow(img.eval())
            plt.title('adjusted{}'.format(i+1))
        plt.show()

代码运行结果

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fpfmwcanchj30to0ek4ei.jpg)

#### 添加色相和饱和度

以下代码是调整图像的色相。

	with tf.Session() as sess:
	    adjusted1=tf.image.adjust_hue(img_data,0.1)
	    adjusted2=tf.image.adjust_hue(img_data,0.3)
	    adjusted3=tf.image.adjust_hue(img_data,0.6)
	    adjusted4=tf.image.adjust_hue(img_data,0.9)
	    
	    plt.figure(figsize=(10,8))
	    for i,img in enumerate([adjusted1,adjusted2,adjusted3,adjusted4]):
	        plt.subplot(2,2,i+1)
	        plt.tight_layout()
	        plt.imshow(img.eval())
	    plt.show()

代码运行结果

![](http://wx1.sinaimg.cn/mw690/006F1DTzgy1fpfn8pfyf2j30jo0e64dl.jpg)

以下代码是调整图像的饱和度。

	with tf.Session() as sess:
	    adjusted1=tf.image.adjust_saturation(img_data,-5)
	    adjusted2=tf.image.adjust_saturation(img_data,5)
	    adjusted3=tf.image.random_saturation(img_data,1,5)
	    
	    plt.figure(figsize=(15,8))
	    for i,img in enumerate([adjusted1,adjusted2,adjusted3]):
	        plt.subplot(1,3,i+1)
	        plt.tight_layout()
	        plt.imshow(img.eval())
	    plt.show()

代码运行结果

![](http://wx3.sinaimg.cn/mw690/006F1DTzgy1fpfnbpqky6j30to06q47k.jpg)

TensorFlow还提供API来完成图片标准化的过程，这个操作就是讲图像上的亮度均值变为0，方差变为1。

	with tf.Session() as sess:
	    adjusted=tf.image.per_image_standardization(img_data)
	    plt.imshow(np.uint8(adjusted.eval()))
	    plt.show()

代码运行结果

![](http://wx2.sinaimg.cn/mw690/006F1DTzgy1fpfnfz2nd9j30ad070gm8.jpg)

#### 添加标注框并裁剪

图像中需要关注的物体通常会被标注框圈出来，TensorFlow提供了一些工具来处理标注框。下面展示了如何通过`tf.image.draw_bounding_boxes`函数在图像中加入标注框。

    with tf.Session() as sess:
        img_data=tf.image.resize_images(img_data,[180,267],method=1)
        batched=tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)
        boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
        result=tf.image.draw_bounding_boxes(batched,boxes)
        
        plt.imshow(result[0].eval())
        plt.show()

代码运行结果

![](http://wx1.sinaimg.cn/mw690/006F1DTzgy1fpfo3jvbsmj30a607077e.jpg)

和随机翻转图像、随机调整颜色类似，随机截取图像上有信息含量得到部分也是一个提高模型robustness的一种方式。这样可以使训练得到的模型不受被识别物体大小的影响。下面展示通过`tf.image.sample_distorted_bounding_box`函数来完成随机截图图像。

	with tf.Session() as sess:
	    boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
	    begin,size,bbox_for_draw=tf.image.sample_distorted_bounding_box(tf.shape(img_data),bounding_boxes=boxes,min_object_covered=0.1)
	    batched=tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)
	    image_with_box=tf.image.draw_bounding_boxes(batched,bbox_for_draw)
	    distorted_image=tf.slice(img_data,begin,size)
	    plt.imshow(distorted_image.eval())
	    plt.show()

代码运行结果

![](http://wx2.sinaimg.cn/mw690/006F1DTzgy1fpfo9s6uotj305x070tam.jpg)

#### 完整的图片预处理实例

	import tensorflow as tf
	import numpy as np
	import matplotlib.pyplot as plt
	
	def distort_color(image,color_ordering=0):
	    if color_ordering==0:
	        image=tf.image.random_brightness(image,max_delta=32./255.)
	        image=tf.image.random_saturation(image,lower=0.5,upper=1.5)
	        image=tf.image.random_hue(image,max_delta=0.2)
	        image=tf.image.random_contrast(image,0.5,upper=1.5)
	    else:
	        image=tf.image.random_saturation(image,lower=0.5,upper=1.5)
	        image=tf.image.random_brightness(image,max_delta=32./255.)
	        image=tf.image.random_contrast(image,lower=0.5,upper=1.5)
	        image=tf.image.random_hue(image,max_delta=0.2)       
	    return tf.clip_by_value(image,0.0,1.0)
	
	def preprocess_for_train(image,height,width,bbox):
	    if bbox is None:
	        bbox=tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32)
	    if image.dtype!=tf.float32:
	        image=tf.image.convert_image_dtype(image,dtype=tf.float32)
	    bbox_begin,bbox_size,_=tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox,min_object_covered=0.1)
	    distorted_image=tf.slice(image,bbox_begin,bbox_size)
	    distorted_image=tf.image.resize_images(distorted_image,[height,width],method=np.random.randint(4))
	    distorted_image=tf.image.flip_left_right(distorted_image)
	    distorted_image=distort_color(distorted_image,np.random.randint(2))
	    return distorted_image
	
	image_raw_data=tf.gfile.FastGFile('../../datasets/cat.jpg','rb').read()
	
	with tf.Session() as sess:
	    img_data=tf.image.decode_jpeg(image_raw_data)
	    boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
	    plt.figure(figsize=(15,15))
	    for i in range(6):
	        result=preprocess_for_train(img_data,299,299,boxes)
	        plt.subplot(3,3,i+1)
	        plt.tight_layout()
	        plt.imshow(result.eval())
	    plt.show()

代码运行结果

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fpfohvzwhkj30to0jr4q2.jpg)

### 多线程输入数据处理框架

为了避免图像预处理成为神经网络模型训练效率的瓶颈，TensorFlow提供了一套多线程处理输入数据的框架。

![](http://wx4.sinaimg.cn/mw690/006F1DTzgy1fpfs83dui8j30hg0f6dju.jpg)

#### 队列与多线程

在TensorFlow中，队列和变量类似，都是计算图上有状态的节点，修改队列状态的操作主要有`enqueue`、`enqueue_many`、`dequeue`。TensorFlow中提供了`FIFOQueue`和`RandomShuffleQueue`两种队列，`FIFOQueue`是生成一个先进先出队列，而`RandomShuffleQueue`是会将队列中的元素打乱。

    import tensorflow as tf
    
    q=tf.FIFOQueue(2,'int32')
    init=q.enqueue_many(([0,10],))
    x=q.dequeue()
    y=x+1
    q_inc=q.enqueue([y])
    
    with tf.Session() as sess:
        init.run()
        for _ in range(5):
            v,_=sess.run([x,q_inc])
            print(v)

代码运行结果

	0
	10
	1
	11
	2

TensorFlow提供了`tf.Coordinator`和`tf.QueueRunner`来完成多线程协同的功能。`tf.Coordinator`主要用于协同多个线程一起停止，并提供了`should_stop`、`request_stop`和`join`三个函数。`tf.QueueRunner`主要用于启动多个线程来操作同一个队列。

	import tensorflow as tf
	import numpy as np
	import threading
	import time
	
	#def MyLoop(coord,worker_id):
	#    while not coord.should_stop():
	#        if np.random.rand()<0.1:
	#            print('stopping from id:{} \n'.format(worker_id))
	#            coord.request_stop()
	#        else:
	#            print('working from id:{} \n'.format(worker_id))
	#        time.sleep(1)
	#
	#coord=tf.train.Coordinator()
	#threads=[threading.Thread(target=MyLoop,args=(coord,i,)) for i in range(5)]
	#for t in threads:
	#    t.start()
	#coord.join(threads)

	queue=tf.FIFOQueue(100,'float')
	enqueue_op=queue.enqueue(tf.random_normal([1]))
	qr=tf.train.QueueRunner(queue,[enqueue_op]*5)
	tf.train.add_queue_runner(qr)
	out_tensor=queue.dequeue()
	
	with tf.Session() as sess:
	    coord=tf.train.Coordinator()
	    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
	    for _ in range(3):
	        print(sess.run(out_tensor)[0])
	    coord.request_stop()
	    coord.join(threads)

#### 输入文件队列

TensorFlow提供了`tf.train.match_filename_once`函数来获取符合一个正则表达式的所有文件，得到的文件列表可以通过tf.train.string_input_producer函数进行有效的管理。

1.生成新文件

	import tensorflow as tf
	
	def _int64_feature(value):
	    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	
	num_shards=2
	instances_per_shard=2
	for i in range(num_shards):
	    filename=('Records/data.tfrecords-%.5d-of-%.5d' % (i,num_shards))
	    writer=tf.python_io.TFRecordWriter(filename)
	    for j in range(instances_per_shard):
	        example=tf.train.Example(features=tf.train.Features(feature={'i':_int64_feature(i),
	                                                                     'j':_int64_feature(j)}))
	        writer.write(example.SerializeToString())
	    writer.close()


2.获取文件，并读取数据

	files=tf.train.match_filenames_once('Records/data.tfrecords-*')
	filename_queue=tf.train.string_input_producer(files,shuffle=False)
	
	reader=tf.TFRecordReader()
	_,serialized_example=reader.read(filename_queue)
	features=tf.parse_single_example(serialized_example,features={'i':tf.FixedLenFeature([],tf.int64),
	                                                              'j':tf.FixedLenFeature([],tf.int64),})
	
	with tf.Session() as sess:
	    tf.local_variables_initializer().run()
	    print(sess.run(files))
	    
	    coord=tf.train.Coordinator()
	    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
	    for i in range(6):
	        print(sess.run([features['i'],features['j']]))
	    coord.request_stop()
	    coord.join(threads)

代码运行结果

	[b'Records\\data.tfrecords-00000-of-00002'
	 b'Records\\data.tfrecords-00001-of-00002']
	[0, 0]
	[0, 1]
	[1, 0]
	[1, 1]
	[0, 0]
	[0, 1]

#### 组合训练数据（batching）

TensorFlow提供了`tf.train.batch`和`tf.train.shuffle_batch`函数将单个的样例组织成batch的形式输出。

	files=tf.train.match_filenames_once('Records/data.tfrecords-*')
	filename_queue=tf.train.string_input_producer(files,shuffle=False)
	
	reader=tf.TFRecordReader()
	_,serialized_example=reader.read(filename_queue)
	features=tf.parse_single_example(serialized_example,features={'i':tf.FixedLenFeature([],tf.int64),
	                                                              'j':tf.FixedLenFeature([],tf.int64),})
	example,label=features['i'],features['j']
	batch_size=3
	capacity=1000+3*batch_size
	
	example_batch,label_batch=tf.train.batch([example,label],batch_size=batch_size,capacity=capacity)
	#example_batch,label_batch=tf.train.shuffle_batch([example,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=30)
	
	with tf.Session() as sess:
	    tf.local_variables_initializer().run()
	    coord=tf.train.Coordinator()
	    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
	    
	    for i in range(2):
	        cur_exmaple_batch,cur_label_batch=sess.run([example_batch,label_batch])
	        print(cur_exmaple_batch,cur_label_batch)
	    coord.request_stop()
	    coord.join(threads)

代码运行结果

	[0 0 1] [0 1 0]
	[1 0 0] [1 0 1]

其中`tf.train.batch`和`tf.train.shuffle_batch`区别就是前者不会随机打乱顺序。

### 完整实例

下面是完整的TensorFlow来处理输入数据，并实现一层全连接神经网络。

	import tensorflow as tf
	
	files=tf.train.match_filenames_once('Records/output.tfrecords')
	filemane_queue=tf.train.string_input_producer(files,shuffle=False)
	
	reader=tf.TFRecordReader()
	_,serialized_example=reader.read(filemane_queue)
	features=tf.parse_single_example(serialized_example,features={'image_raw':tf.FixedLenFeature([],tf.string),
	                                                              'pixels':tf.FixedLenFeature([],tf.int64),
	                                                              'label':tf.FixedLenFeature([],tf.int64)})
	decoded_images=tf.decode_raw(features['image_raw'],tf.uint8)
	retyped_images=tf.cast(decoded_images,tf.float32)
	labels=tf.cast(features['label'],tf.int32)
	
	images=tf.reshape(retyped_images,[784])
	
	min_after_dequeue=1000
	batch_size=100
	capacity=min_after_dequeue+3*batch_size
	
	image_batch,label_batch=tf.train.shuffle_batch([images,labels],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)
	
	def inference(input_tensor,weights1,biases1,weights2,biases2):
	    layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
	    return tf.matmul(layer1,weights2)+biases2
	
	INPUT_NODE = 784
	OUTPUT_NODE = 10
	LAYER1_NODE = 500
	REGULARAZTION_RATE = 0.0001   
	TRAINING_STEPS = 5000        
	
	weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
	biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
	
	weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
	biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
	
	y = inference(image_batch, weights1, biases1, weights2, biases2)
	
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label_batch)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	    
	regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
	regularaztion = regularizer(weights1) + regularizer(weights2)
	loss = cross_entropy_mean + regularaztion
	
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
	    
	with tf.Session() as sess:
	    tf.global_variables_initializer().run()
	    tf.local_variables_initializer().run()
	    coord = tf.train.Coordinator()
	    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	    for i in range(TRAINING_STEPS):
	        if i % 1000 == 0:
	            print("After %d training step(s), loss is %g " % (i, sess.run(loss)))
	        sess.run(train_step) 
	    coord.request_stop()
	    coord.join(threads) 	