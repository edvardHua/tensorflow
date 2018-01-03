# 评估器

这篇文档介绍的 @{tf.estimator$**Estimator**} 是一个高级的 TensorFlow API，它极大的简化了机器学习编程工作。评估器封装了如下行为：

* 训练
* 评估
* 预测
* 作为服务导出

你可以使用我们预定义好的评估器，也可以自己写一个评估器。但无论预定义的还是自定义的评估器都继承自 @{tf.estimator.Estimator} 类。

注意：TensorFlow 还包含了一个已经被弃用的 `Estimator` 类 @{tf.contrib.learn.Estimator}，我们不要去使用它。

## 使用评估器的优势

评估器可以为我们带来以下几点好处：

* 基于评估器的模型可以运行在单机上，也可以运行在分布式的多台服务器环境上并且不需要做任何修改。更棒的是，你还可以将基于评估器的模型运行在 CPUs，GPUs 或者 TPUs 上。
* 评估器简化了模型开发人员内部共享的实现。
* 你可以用高级的直观的代码编写某种状态的艺术模型。简而言之，使用评估器通常会比使用 TensorFlow 的低级 API 更加简捷。
* 评估器是建立在 tf.layers 上的，简化了自定义的内容。
* 评估器已经为你构建了图表。换句话说，你不需要构建图表了。
* 评估器提供了一个安全的分布式训练的循环，能够控制以下操作的运行时间和运行方式：
  * 创建图表
  * 初始化变量
  * 启动队列
  * 处理异常
  * 创建校验文件和错误恢复
  * 储存给 TensorBoard 展示的数据

当你用评估器写应用时，你必须将数据输入管道和模型分开。这种分离简化了不同数据集的实验。

## 预定义的评估器

比起 TensorFlow 的低级 API，预定义的评估器可以让你在更高抽象的层面上工作。你不再需要操心的创建计算图和会话，因为评估器已经帮你把这一切都`串通`好了。也就是说，预定义的评估器已经帮你创建和管理 @{tf.Graph$`Graph`} 和 @{tf.Session$`Session`} 对象。甚至，预定义的评估器可以让你修改最少的代码来试验不同的模型架构。譬如 @{tf.estimator.DNNClassifier$`DNNClassifier`} 就是一个预定义的评估器，它可以训练密集的前向传递神经网络分类模型。


### 预定义评估器的程序结构

基于预定义评估器的程序一般包含下面四步：

1. **编写一个或多个数据集的导入函数。**举个例子，你可能会创建两个函数，一个用于导入训练数据，另一个用于导入测试数据。每一个数据集的导入函数都会返回下面两个对象：
    
    * 一个字典，它的 key 是特征名，而 values 是对应的张量（或者是稀疏张量），张量里面包含了对应的特征数据。
    * 一个张量，它包含了一个或多个标签。
    
    举个例子，下面的代码是一个输入函数的基本框架：
    
    ```python
    def input_fn(dataset):
    	... # 操作数据集，提取特征名称和标签
    	return feature_dict, label
    ```
    
    更多的细节，请看 @{$programmers_guide/datasets}。

2. **定义特征列。**每一个 @{tf.feature_column} 定义了特征的名字、类型或者各种输入预处理函数。举个例子，下面的代码片段创建了三个特征列，它们的类型是整形或者浮点型。前面两个特征列简单的标识了它们的名称和类型。第三个特征则定义了一个 lambda 表达式来对原始数据做转换：
	
	```python
	# 定义三个数值类型的特征列
   population = tf.feature_column.numeric_column('population')
   crime_rate = tf.feature_column.numeric_column('crime_rate')
   median_education = tf.feature_column.numeric_column('median_education',
                      normalizer_fn='lambda x: x - global_education_mean')
	```
	
3. **实例化相关的预定义评估器。**举个例子，下面有一个 `LinearClassifier` 评估器的实例化的代码：
	
	```python
	estimator = tf.estimator.Estimator.LinearClassifier(
   feature_columns=[population, crime_rate, median_education],
   )
	```

4. **调用训练，评估和推断的方法。**
	譬如说，所有的评估器都提供了 `train` 方法，它可以用来训练模型。
	
	```python
   # my_training_set 是在第一步中创建的函数
   estimator.train(input_fn=my_training_set, steps=2000)
	```

### 预定义评估器的好处

预定义评估器是编码的最佳实践，它有下面两点好处：

* 单机或者集群上运行时，计算图的哪部分应该在哪里运行和其实现策略的最佳实践。
* 事件记录和通用内容摘要的最佳实践。
    
如果你不使用预定义的评估器，那么你需要自己实现上面所说到的功能。

## 自定义评估器

预定义和自定义评估器的核心是**模型函数**， 它可以用来构建训练、评价和预测的图表。当你使用预定义评估器时，里面已经实现了模型函数了。但是当你要使用自定义评估器时，你就要自己编写模型函数。@{$extend/estimators$companion document} 描述了编写模型函数的方法。

## 推荐的工作流

我们推荐的工作流如下：

1. 假设存在一个合适的评估器，使用它来构建你的第一个模型，并以这个模型的结果作为基准。
2. 使用当前的预定义评估器构建、测试你所有的管道，包括数据的完整性和可靠性。
3. 如果存在可选的预定义评估器，可以对这几个评估器做实验，从中选择一个能够产生最好结果的评估器。
4. 或许，可以通过构建你自己的评估器来进一步提升模型的效果。


## 从 Keras 模块中创建评估器

你可以将 Keras 模型转换成评估器。这样 Keras 模型就可以利用到评估器的优点了，譬如分布式训练。可以如下例所示调用 @{tf.keras.estimator.model_to_estimator}：

```python
# 实例化一个 kera inception v3 模型。
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
# 定义好用来训练模型使用的优化器，损失和评价指标，然后再编译它
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')
# 从编译好的 Keras 模型创建一个评估器
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)
# 像使用其他评估器一样使用派生的评估器。
# 以下例子中，派生的评估器调用了训练方法。
est_inception_v3.train(input_fn=my_training_set, steps=2000)
```

想要了解更多的细节，请查阅 @{tf.keras.estimator.model_to_estimator} 。
