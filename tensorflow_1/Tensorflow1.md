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

## Tensorflowとは
2015/11にオープンソース化されたGoogleの機械学習ライブラリ  
基本的にはpythonを使用して操作するが、バックエンドはC++で計算している

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
※ [MNIST](http://yann.lecun.com/exdb/mnist/)・・・28x28ピクセル、70000サンプルの数字の手書き画像データ(Mixed National Institute of Standards and Technology database)  
機械学習の"Hello World"的存在  
画像にはそれぞれどの数字を示すかのラベルが付与されている

手書きの数字画像を見て、何の数字が書かれているかを予測するチュートリアル  


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

- MNISTデータをダウンロード  

データは訓練データが55,000データ・ポイント, テストデータが10,000データポイント, バリデーションデータが5,000データポイントの3つに分かれている。  
学習内容が一般化されているかを確認するために、この分割(訓練データとテストデータに分割すること)が重要。  
(訓練データもテストデータも)データは画像とラベルの両方を持つが、画像をxs, ラベルをysとして表す。  
各画像は28x28ピクセルで、大きな配列として解釈することができるので、784(28x28)のベクトルを配列を使ってフラットにすることができる。(画像間で一貫していれば、フラットにする方法はそれほど重要ではない。)  
データをフラットにする際、画像の2次元構造の情報を捨てることになる。一般的には2次元構造のデータを使用するため捨てないで処理を行うが、このチュートリアルでは「ソフトマックス回帰」という単純な方法をしようするため捨ててしまう。

![MNIST画像配列化例](https://www.tensorflow.org/versions/r0.7/images/MNIST-Matrix.png)
結果、学習データ(55,000枚)のxsは、[55000, 784]の形状を持つtensor(n次元配列)となる。1次元目は画像のインデックス、2次元目は各画像のピクセルのインデックスを表し、tensorの各要素は、特定の画像の特定のピクセルのための、0~1間のピクセル強度を表す。  

![MNIST画像フラット例](https://www.tensorflow.org/versions/r0.7/images/mnist-train-xs.png)
学習データ(55,000枚)のysは、このチュートリアルでは「1 - ホットベクトル」を使って表す。「1 - ホットベクトル」とは、ほとんどの次元が0で、1つの次元だけ1であるベクトルを表す。例えば3は、[0,0,0,1,0,0,0,0,0,0]となる。結果として、floatの[55000, 10]の形状を持つtensorとなる。

![MNIST画像フラット例2](https://www.tensorflow.org/versions/r0.7/images/mnist-train-ys.png)

ここまでで実際にモデルを作成する準備が終了

- ソフトマックス回帰  

MNIST内の画像はどの数値が書かれているかはわからないものの、数値が書かれていることは前提としてわかっていて、画像を見て各々の数字である確率を与えることが出来るようにしたい。(例. とあるモデルが9の画像を見て、80%の確信度で9、上部が丸になっているので5%の確信度で8、その他数値はわずかな確信度)  
ソフトマックス回帰は自然で単純なモデルで、古典的なケース。より洗練されたモデルを使用する場合でも、最後のステップはソフトマックスのレイヤーになる。  
ソフトマックス回帰には、2つのステップがある。最初に、入力がある特定のクラスに含まれる証拠を足しあわせ、次に、証拠を確率に変換する。  
証拠を合計するために、ピクセル強度の加重和(重み付けをした和)を行う。

## 学習結果を可視化
※ 絶対パスで指定

```sh
ᐅ tensorboard --logdir=/Users/fujisakikyo/Documents/tensorflow_study/tensorFlow/data
```


## ref
[http://qiita.com/shu223/items/a4fc17eb3356a6068553](http://qiita.com/shu223/items/a4fc17eb3356a6068553)  
[http://dev.classmethod.jp/machine-learning/tensorflow-hello-world/](http://dev.classmethod.jp/machine-learning/tensorflow-hello-world/)  
[http://www.slideshare.net/masuwo3/tensorflow](http://www.slideshare.net/masuwo3/tensorflow)
[https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html](https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html)
[http://qiita.com/KojiOhki/items/ff6ae04d6cf02f1b6edf](http://qiita.com/KojiOhki/items/ff6ae04d6cf02f1b6edf)
[http://qiita.com/haminiku/items/36982ae65a770565458d](http://qiita.com/haminiku/items/36982ae65a770565458d)