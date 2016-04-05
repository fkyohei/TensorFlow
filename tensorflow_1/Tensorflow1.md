# Tensorflow

## install
- インストール

```sh
~  ᐅ sudo easy_install pip
~  ᐅ sudo pip install --upgrade virtualenv
~  ᐅ virtualenv --system-site-packages ~/tensorflow
~  ᐅ source ~/tensorflow/bin/activate
(tensorflow) ~  ᐅ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl
```

- activate tensorflow  

```sh
~  ᐅ source ~/tensorflow/bin/activate
```

- deactivate tensorflow  

```sh
(tensorflow) ~/Documents/tensorflow ᐅ deactivate
```


## Tensorflowの計算処理
Tensorflowの計算処理はグラフで表される。  
(グラフ・・・NodeとEdgeで表現されるデータ構造の一種)  
![グラフ](http://cdn.dev.classmethod.jp/wp-content/uploads/2015/11/graph.png)  
TensorflowではNodeはops(operation)とよばれる計算処理の一つと対応します。
opsはTensorという行列形式のデータを受け取り、計算処理後、結果を次のopsに渡します。
データをグラフ構造にそって次々opsに受け渡し、計算処理を行うことで最終的に求めたい計算結果を得ることができます。

計算処理グラフを構築したあと、書くops毎にCPUコアやGPUコアなどの計算資源を割り当て、計算処理を行います。
Tensorflowでは、この計算資源の割り当てをSessionと呼ばれるコンポーネントが割り当てられる。

(Tensorflowの利用例にあげられるNN([ニューラルネットワーク](https://www.sist.ac.jp/~suganuma/kougi/other_lecture/SE/net/net.htm))系の機械学習は、計算処理として見ると複雑な行列計算の組み合わせで行われているため、より効率よく学習処理を行うために、各行列処理をops単位で区切り、それぞれに計算資源の割り当てる形になっています。また、計算資源の割り当てをSessionに任せることで、環境による差異を吸収して、環境に合わせて効率よく分散処理を行う事ができます。)

まとめると、opsでグラフ構造で計算処理を構築するフローと、Sessionにグラフを渡して計算処理を行う2つのフローの2段階があります。

## run hellow world
```sh
(tensorflow) ~/Documents/tensorflow  ᐅ vim helloworld.py
```
helloworld.py

```python
import tensorflow as tf
# CPU数を取得するためにimport
import multiprocessing as mp

# CPU数取得し、Session作成(通常は環境から自動的に取得・割当がおこなわれる)
core_num = mp.cpu_count()
config = tf.ConfigProto(
    inter_op_parallelism_threads=core_num,
    intra_op_parallelism_threads=core_num )
sess = tf.Session(config=config)

# 0次元の行列データとして、"hello, tensorflow!!"をTensorに格納
hello = tf.constant('hello, tensorflow!!')
# TensorをSessionに渡して、計算結果としてその値をそのまま出力
print sess.run(hello)

# 0次元の行列データとして、10, 32をTensorに格納
a = tf.constant(10)
b = tf.constant(32)
# 計算式を定義(Tensor)
c = a + b
# TensorをSessionに渡して、計算処理を行い、結果を出力
print sess.run(c)
```

```sh
(tensorflow) ~/Documents/tensorflow  ᐅ python helloworld.py
I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op parallelism threads: 4
I tensorflow/core/common_runtime/local_session.cc:45] Local session inter op parallelism threads: 4
hello, tensorflow!!
42
```

**c = a + b**は計算式を定義しているだけで、ここでは計算を行っていない。あくまで計算の処理はSessionに投入した時に行われる。
## MNIST For ML Beginners
(TensorFlow公式チュートリアル)  
※ MNIST・・・28x28ピクセル、70000サンプルの数字の手書き画像データ(Mixed National Institute of Standards and Technology database)

### サンプルコードを含むtensorflowのソースコードを取得
```sh
ᐅ git clone -b r0.7 --recurse-submodules https://github.com/tensorflow/tensorflow
```

### サンプル実行
```sh
ᐅ python tensorflow/examples/tutorials/mnist/fully_connected_feed.py
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting data/t10k-labels-idx1-ubyte.gz
Step 0: loss = 2.31 (0.034 sec)
Step 100: loss = 2.16 (0.005 sec)
Step 200: loss = 1.97 (0.005 sec)
Step 300: loss = 1.65 (0.005 sec)
Step 400: loss = 1.36 (0.005 sec)
Step 500: loss = 0.97 (0.005 sec)
Step 600: loss = 0.85 (0.005 sec)
Step 700: loss = 0.79 (0.005 sec)
Step 800: loss = 0.68 (0.005 sec)
Step 900: loss = 0.53 (0.005 sec)
Training Data Eval:
  Num examples: 55000  Num correct: 47109  Precision @ 1: 0.8565
Validation Data Eval:
  Num examples: 5000  Num correct: 4326  Precision @ 1: 0.8652
Test Data Eval:
  Num examples: 10000  Num correct: 8665  Precision @ 1: 0.8665
Step 1000: loss = 0.63 (0.017 sec)
Step 1100: loss = 0.52 (0.113 sec)
Step 1200: loss = 0.50 (0.005 sec)
Step 1300: loss = 0.53 (0.005 sec)
Step 1400: loss = 0.30 (0.005 sec)
Step 1500: loss = 0.41 (0.005 sec)
Step 1600: loss = 0.37 (0.005 sec)
Step 1700: loss = 0.47 (0.005 sec)
Step 1800: loss = 0.35 (0.005 sec)
Step 1900: loss = 0.33 (0.005 sec)
Training Data Eval:
  Num examples: 55000  Num correct: 49026  Precision @ 1: 0.8914
Validation Data Eval:
  Num examples: 5000  Num correct: 4512  Precision @ 1: 0.9024
Test Data Eval:
  Num examples: 10000  Num correct: 8991  Precision @ 1: 0.8991
```

## 学習結果を可視化
※ 絶対パスで指定

```sh
ᐅ tensorboard --logdir=/Users/fujisakikyo/Documents/tensorflow_study/tensorFlow/data
```


## ref
[http://qiita.com/shu223/items/a4fc17eb3356a6068553](http://qiita.com/shu223/items/a4fc17eb3356a6068553)  
[http://dev.classmethod.jp/machine-learning/tensorflow-hello-world/](http://dev.classmethod.jp/machine-learning/tensorflow-hello-world/)  
[http://www.slideshare.net/masuwo3/tensorflow](http://www.slideshare.net/masuwo3/tensorflow)