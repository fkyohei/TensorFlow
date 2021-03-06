# 機械学習入門

前回TensorFlowのチュートリアルに沿って機械学習についての説明を始めましたが、説明をしてみて、TensorFlowのチュートリアルに沿って説明をするのではなく、機械学習の概念をちゃんと理解する方が順序として正しいと思い、機械学習入門を行います。  
主に機械学習とニューラルネットワーク等の機械学習周辺を話せればいいなと思っています。

## 今回の教材
[ニューラルネットワークと深層学習](http://nnadl-ja.github.io/nnadl_site_ja/index.html
)  
無料のオンライン書籍(の和訳のベータ版です。まだ和訳も途中のものです)。  
機械学習を学ぶといっているのに「深層学習」となっていますが、深層学習とは多層構造のニューラルネットワークの機械学習を意味します(機械学習の一種というイメージで大丈夫だと思います)。  
書籍に沿って進めるため(一部飛ばしたりするかもですが)、話が少しそれたりすることがあります。

## このテキストのゴール
**"機械学習をなんとなく理解する"**  
なぜなんとなくなのか？  
=> 単純に難しいからです。。  
※下記Qiita引用(引用元: [http://qiita.com/IshitaTakeshi/items/4607d9f729babd273960](http://qiita.com/IshitaTakeshi/items/4607d9f729babd273960))  
> 勉強会ではこんなことを聞きました。  
「アルゴリズムの実装もやりたいんだけど、その前に理論を理解したい」  
アルゴリズムを完璧に理解してからコードを書きたくなるというのはプログラマとしてすごくいい癖(素質)だと思います。  
しかし、機械学習に関しては、最初から完璧に理論を理解しようとするのはあまり良くないと私は思います。なぜなら、単純に難しいからです。  
機械学習の理論の大半は高度な数式で構成されています。  
最も単純なアルゴリズムであるパーセプトロンなら、高校レベルの数学力があれば容易に理解できます。しかし、それより少し性能のいいSVM(サポートベクターマシン)などを理解しようとすると、とたんに難しい部分がたくさん出てきます。  
例えば、SVMの最も重要な部分であるカーネル関数の原理を理解しようとすると、非常に高度な数学が必要になります。どのくらい高度か知りたい方は、「カーネル関数　原理」などでググってみて出てくる数式を眺めてみるといいでしょう。  
要は、「機械学習をやってみたい」もしくは「機械学習を始めてみた」という方がいきなり理論を完璧に理解するのはまず無理だということです。

## 入門を始める前に
入門のための入門をします。。  
先人がスライドを作成しているので、そちらを使用します。  
[http://www.slideshare.net/unnonouno/jubatus-casual-talks
](http://www.slideshare.net/unnonouno/jubatus-casual-talks
)

それでは、機械学習入門スタートします。

## ニューラルネットワークの歴史
プログラミングする従来の方法では、解きたい問題を、コンピュータに実行できるよう明確に定義された無数の小さなタスクに分割し、コンピュータに何をすべきか逐一指示します。  
これに対し、ニューラルネットワークを使う場合は、直接問題の解き方を指示しません。コンピュータが観測データから学習し、問題の解き方を自ら編み出します。  
ただし、2006年まではニューラルネットワークを訓練して、伝統的な手法より良い結果を出させる方法がわかっていませんでした。2006年に起きた変化が深層学習(ディープラーニング)の発見でした。そして手法が進歩した結果、音声認識や自然言語処理等における多くの重要な問題で、優れた実績を達成しています。

## ニューラルネットワークを用いた手書き文字認識
(TensorFlowで説明しようとしてたMNISTと同じか同等のものです)  
次の手書き数列を読んでみてください。  
<img src="http://nnadl-ja.github.io/nnadl_site_ja/images/digits.png" width="300px">  
(わりと簡単に?)**504192**と読めたでしょうか？  
では、このとき脳内ではどう処理されているでしょうか。脳で起こっていることは簡単どころではありません。  
脳の2つの半球にはそれぞれ、一次視覚野(V1とも呼ばれる)、1億4千万のニューロン(神経細胞)と何十億のシナプス(神経細胞間)からなる領域が存在します。さらに人間の視覚に関わっている領域は、V1,V2,V3,V4,V5という一連の視覚野が順次複雑な画像処理に携わっています。  
人は目に見えるものを解釈するという作業を得意とし、その作業を無意識のうちに行っています。  
これを手書き数字を認識するプログラムを書こうとすれば、視覚パターン認識の困難さが明らかになります。数字を認識するための直感的で単純なルール(数字の9は、上に輪があって、右下から下に向かって線が生えている形)をアルゴリズムで表現するのは決して単純ではありません。このようなルールを正確にプログラムで表現しようとすれば、すぐに膨大な例外、落とし穴、特殊ケースに気づくでしょう。  

ニューラルネットワークはこのような問題に違った角度から迫ります。ニューラルネットワークは下記のような手書き数字のデータをあらかじめたくさん用意して(このようなデータを訓練例や訓練データといいます)、その上で、訓練例から学習することのできるシステムを開発するというものです。  
言い換えると、訓練例をもとに、数字認識のルールを自動的に推論します。さらに訓練例を増やすと、ニューラルネットワークは手書き文字に関する知識をより多く獲得し、精度が向上します。
<img src="http://nnadl-ja.github.io/nnadl_site_ja/images/mnist_100_digits.png" width="500px">

手書き文字認識を理解する(書籍だと実装する)過程で、ニューラルネットワークの鍵となるアイデアをいくつか出てくるので、その中でも重要な人口ニューロン(パーセプトロンとシグモイドニューロン)や、ニューラルネットワークの標準的な学習アルゴリズムである確率的勾配降下法を紹介・なぜその手法なのかを説明していきます。(途中で挫折しなければですが。。。  )

### パーセプトロン
ニューラルネットワークとは何か、という解説を始めるにあたり、まずはパーセプトロンと呼ばれる種類の人口ニューロンから話を始めます。今日ではパーセプトロン以外の人口ニューロンモデルを扱うことが一般的です。現代のニューラルネットワーク研究の多くでは、シグモイドニューロンと呼ばれるモデルが主に使われていて、この後その説明もしますが、なぜシグモイドニューロンが今の姿をしているのかを知るために、こちらを理解することにします。

パーセプトロンとは何か?  
パーセプトロンは複数の2進数$x_1, x_2, \ldots$を入力にとり、ひとつの2進数を出力します。  

![パーセプトロンとは](http://nnadl-ja.github.io/nnadl_site_ja/images/tikz0.png)  

上図の例では、パーセプトロンは3つの入力$x_1, x_2, x_3$をとっています(一般的には入力はいくつでも構いません)。開発者は**重み**$w_1, w_2, \ldots$という概念を導入しました。  
重みとは、それぞれの入力が出力に及ぼす影響の大きさを表す実数です。パーセプトロンの出力が0になるか1になるかは、入力の重み付き和 $\sum_j w_j x_j$ と **閾値（しきい値）** の大小比較で決まります。重みと同じく、閾値もパーセプトロンの挙動を決める実数パラメータです。より正確に数式で表現するなら、  
$$\begin{eqnarray}
  \mbox{output} & = & \left\{ \begin{array}{ll}
      0 & \mbox{if } \sum_j w_j x_j \leq \mbox{ threshold} \\
      1 & \mbox{if } \sum_j w_j x_j > \mbox{ threshold}
      \end{array} \right.
\tag{1}\end{eqnarray}$$  
パーセプトロンを動かすルールはこれだけです。  
直感的に言えば、パーセプトロンとは、複数の情報に重みをつけながら決定をくだす機械だといえます。

- 現実的な例  
週末に「チーズ祭り」が開催されます。あなたはチーズが好物で、チーズ祭りに行くかどうか決めようとしています。判断に影響を及ぼしそうなファクターは3つあります。

1. 天気はいいか
2. あなたの恋人も一緒に行きたがっているか？
3. 祭りの会場は駅から近いか？(電車を使用する前提)

これらの3つのファクターを2進数値$x_1,x_2,x_3$で表現し、天気が良いなら$x_1=1$、悪いなら$x_1=0$、同様に$x_2,x_3$を扱う。  
あなたはチーズが大好物で、恋人がなんと言おうが、会場が駅から遠かろうが、チーズ祭りに行くつもりかもしれません。一方、雨が何より苦手で、天気が悪かったら絶対に行くつもりはないかもしれません。パーセプトロンはこのような意思決定を表現することが可能です。  
1つの方法が天気の条件の重みを$w_1=6$、他の重みを$w_2=2$と$w_3=2$にすることです。重みの値が大きい条件はあなたの意思決定で大事なことであることを表します。最後に、パーセプトロンの閾値を5に設定します。以上のパラメータ設定により、パーセプトロンで意思決定モデルを実装できました。このパーセプトロンは天気が良ければ必ず1を出力し、天気が悪ければ0を出力します。恋人の意思や、駅からの距離によって結論が変わることはありません。  

重みと閾値を変化させることで様々な意思決定モデルを得ることが出来ます。たとえば、閾値を5から3に変えてみると、「天気が良い」**または**「会場が駅から近い **かつ** あなたの恋人が一緒に行きたがっている」となります。  
パーセプトロンは人間の意思決定モデルの完全なモデルではないですが、パーセプトロンはことなる種類の情報を考慮し、重みをつけた上で判断を下す能力が有るため、パーセプトロンを複雑に組み合わせたネットワークなら、かなり微妙な判断も扱えます。  

![パーセプトロン組み合わせ](http://nnadl-ja.github.io/nnadl_site_ja/images/tikz1.png)  

上図のネットワークでは、まず1列目の3つのパーセプトロン(第一層のパーセプトロンと呼ぶことにする)が入力情報に重みをつけて、とても単純な判断を行っています。第二層のパーセプトロンはなにをしているかというと、第一層のパーセプトロンの下した判断に重みを付けることで判断を下しています。第二層のパーセプトロンは、第一層のパーセプトロンよりも複雑で、抽象的な判断を下しているといえます。第三層のパーセプトロンはさらに複雑な判断を行っています。このように多層のニューラルネットワークは高度な意思決定を行うことができるます。  

パーセプトロンを定義した時、パーセプトロンは出力を1つしか持たないといいましたが、上図のネットワークの中のパーセプトロンは複数の出力を持つように書かれています。でも、あくまでもパーセプトロンの出力はひとつで、出力の矢印が複数あるのは、ただ、あるパーセプトロンの出力が複数のパーセプトロンの入力として使われていることを示しているに過ぎません。  

パーセプトロンの記法をもっと簡潔にします。パーセプトロンが1を出力する条件 $\sum_j w_j x_j > \mbox{threshold}$ をもっと簡単に書くために、内積を使用して$w \cdot x \equiv \sum_j w_j x_j$と書くことにします。ここで$w$と$x$は重さと入力を要素に持つベクトルです。  
次に閾値を不等式の左辺に移項し、パーセプトロンの**バイアス** $b \equiv-\mbox{threshold}$ と呼ばれる量に置き換えます。閾値の代わりにバイアスを使用すると、パーセプトロンのルールは下記のように書き換えられます。
$$\begin{eqnarray}
  \mbox{output} = \left\{
    \begin{array}{ll}
      0 & \mbox{if } w\cdot x + b \leq 0 \\
      1 & \mbox{if } w\cdot x + b > 0
    \end{array}
  \right.
\tag{2}\end{eqnarray}$$  

バイアスはパーセプトロンが1を出力する傾向の高さを表す量とみなすことができます。バイアスの値が大きければ1を出力するのは簡単となります。パーセプトロンで閾値をバイアスとして表すのは表記を変更にするにすぎませんが、バイアスを使ったほうがシンプルになる場合がこの後出てくるので、バイアスを使用します。  

ここまでの解説では、パーセプトロンを入力情報に重みをつけて判断を行う手続きとして用いてきましたが、パーセプトロンには他の用途もあります。それは論理演算を計算することです。あらゆる計算は、AND、OR、NANDといった基本的な論理関数から構成されているとみなすことができ、パーセプトロンはこういった論理関数を表すことができるのです。  
例えば、2つの入力をとり、どちらも重みが-2で、全体のバイアスが3であるようなパーセプトロンを考えることにします。  
![パーセプトロン論理関数1](http://nnadl-ja.github.io/nnadl_site_ja/images/tikz2.png)  
このパーセプトロンは、00を入力されると1を出力することがわかります。なぜなら、$(-2)*0+(-2)*0+3 = 3$は正の数だからです。同じように計算すると、このパーセプトロンは01や10を入力しても1を出力します。しかし、11を入力した時だけ0を出力します。$(-2)*1+(-2)*1+3 = -1$は正ではないからです。つまり、このパーセプトロンはNANDゲートを実装していることになります。  

![パーセプトロングラフ](http://hokuts.com/wp-content/uploads/2015/11/graph_step2.png)  

このことから、パーセプトロンが単純な論理関数を計算することができることがわかります。さらに、パーセプトロンのネットワークさえあれば任意の論理関数が計算できることまでわかります。NANDゲートは論理計算において万能だからです。NANDゲートさえあれば、どんな計算も構成できます。よって、NANDゲートがどんな論理計算もできるため、パーセプトロンもどんな計算もできるということになります。  
ここまでの話だと、NANDゲートを再発明したような印象になります。車輪の再発明のような感じです。  
しかし、そんなことはありません。我々はニューラルネットワークの重みとバイアスを自動的に最適化する学習アルゴリズムを開発できるからです。この最適化は、プログラマの介入なしに、外部刺激によって勝手に起こります。これらの学習のおかげで、従来の論理ゲートとは異なる使い方が可能になりました。NANDゲートやその他論理ゲートはすべて手動で配線してやる必要があったのに対し、ニューラルネットワークは問題の解き方を自発的に学習してくれるのです。  

### シグモイドニューロン
学習アルゴリズムは素晴らしいが、ニューラルネットワークに対してそのようなアルゴリズムをどう設計すればいいのでしょうか？  
例えば、パーセプトロンのネットワークを使用して、入力は手書き文字のスキャン画像の生ピクセルデータ、ニューラルネットワークには数字を正しく分類出来るよう重みとバイアスを学習してほしいとします。学習がどのように働くのかを知るために、ネットワークの中のいくつかの重みやバイアスを少しだけ変更するとしましょう。このような小さな変更に対応する、ニューラルネットワークからの出力の変化もまた小さなものであれば、ニューラルネットワークがより自分の思った通りの挙動を示すように重みとバイアスを修正できます。（例えば、ニューラルネットワークがある「9」であるべき手書き文字を間違って「8」に分類したとすると、重みやバイアスに小さな変化を与えて、どうすればこのニューラルネットワークが正しく「9」と分類する方向に近づくかを探ることが出来ます。この過程を繰り返すことで次第に結果は改善されるはずで、ニューラルネットワークはこうして学習します。）  
問題は、ニューラルネットワークがパーセプトロンで構成されていたとすると、このような学習は起こらないということです。実際にニューラルネットワーク内のパーセプトロンのうち、どれか１つの重みやバイアスを少し変えてやると、出力は変化がない・もしくは0から1へというようにすっかり反転してしまいます。1箇所反転すると他の部分も連動して複雑に変わってしまいます。つまり、先ほどの例で「9」をなんとか正しく分類させることが出来たとしても、「9」以外の数値の挙動までもが完全に変わってしまいます。現在もパーセプトロンで構成されたニューラルネットワークに上手に学習させる方法は明らかになっていません。  

この問題は、**シグモイドニューロン**と呼ばれる、新しいタイプの人工ニューロンを導入することによって克服することが出来ます。シグモイドニューロンはパーセプトロンと似ていますが、シグモイドニューロンの重みやバイアスに小さな変化を与えた時、それに応じて生じる出力の変化も小さなものに留まるように調整されています。このことが学習を可能にする決定的な違いとなります。  

シグモイドニューロンもパーセプトロンと同じような見た目で描くことにします。  
![シグモイドニューロン](http://nnadl-ja.github.io/nnadl_site_ja/images/tikz9.png)  
パーセプトロンと同様、シグモイドニューロンは$x_1, x_2,\ldots$と言った入力をとりますが、これらの入力は0や1だけでなく、**0から1の間のあらゆる値**をとることが出来ます。シグモイドニューロンはそれぞれの入力に対して重みを持ち、ニューロン全体に対するバイアスと呼ばれる値を持っています。  
出力は $\sigma(w \cdot x+b)$ という値をとります。$\sigma$はシグモイド関数と呼ばれ、次の式で定義されます。  
$$\begin{eqnarray}
  \sigma(z) \equiv \frac{1}{1+e^{-z}}.
\tag{3}\end{eqnarray}$$  
より明確に表現すると  
$$\begin{eqnarray}
  \frac{1}{1+\exp(-\sum_j w_j x_j-b)}.
\tag{4}\end{eqnarray}$$  
となります。  
（※ $\sigma$はロジスティック関数と呼び、このニューロンをロジスティック・ニューロンと呼ぶこともあります）  
上の数式は一見すると、パーセプトロンと大きく異なるようにみえますが、共通部分も多いです。共通部分を理解するために、$z \equiv w \cdot x + b$を大きな正の数とします。このとき、$e^{-z} \approx 0$つまり $\sigma(z) \approx 1$ となります。言い換えると、$z = w \cdot x+b$が大きな数であるとき、シグモイドニューロンの出力はほぼ1となり、パーセプトロンと同じになります。逆に、$z = w \cdot x+b$は大きな負の数とします、その時、$e^{-z} \rightarrow \infty$であり $\sigma(z) \approx 0$ になります。つまり、$z = w \cdot x +b$が大きな負の数になる時も、シグモイドニューロンはパーセプトロンとほぼおなじ動きになります。ただし、$z = w \cdot x +b$がそこまで大きな数でない場合はパーセプトロンと同じにはなりません。  
![シグモイド曲線](http://norimune.net/wp/wp-content/img/lreg5.jpg)  
パーセプトロンと同様な動きであり、かつ重みやバイアスの小さな変化に対して小さなoutputの変化となっていることが重要です。

### ニューラルネットワークのアーキテクチャ
説明の前にニューラルネットワークに名前を付けておきます。  
![ニューラルネットワークアーキテクチャ画像](http://nnadl-ja.github.io/nnadl_site_ja/images/tikz10.png)  
一番左の層は入力層(input layer)と呼ばれその中のニューロンを入力ニューロン(input neurons)といいます。一番右の層は出力層(output layer)と呼ばれその中のニューロンは出力ニューロン(output neurons)といいます。中央の層は入力でも出力でもないことから隠れ層(hidden layer)と呼ばれます。隠れ層をいくつも持っているものも存在します。  
![ニューラルネットワークアーキテクチャ画像2](http://nnadl-ja.github.io/nnadl_site_ja/images/tikz11.png)  
ニューラルネットワークの入出力層の設計は単純です。たとえば、手書きの画像が9であるかを判断したいとします。設計の自然な方法は、その画像のピクセルあたりの色の度合いを入力ニューロンにエンコードすることです。もしその画像が$64 \times 64$の白黒画像であれば、入力ニューロンの数は$4,096 = 64 \times 64$となり、色の度合いは明度を0から1の適切な値で表現します。出力層は1つのニューロンからなり、出力値が0.5以上なら入力画像は9であるということを示し、0.5未満であれば入力画像は9でないということを示します。ある層の出力が次の層の入力になるようニューラルネットワークは**フィードフォワードニューラルネットワーク**と呼ばれる。これは、ネットワーク内にループがないことを意味します。（フィードバックループを用いることが可能な、再帰型ニューラルネットワークと呼ばれるモデルも存在します）  


 ### ソフトマックス関数
 ソフトマックス関数は、シグモイド関数の多変量版。（1変量・2変量・多変量がある。[参考](http://www.macromill.com/landing/words/b011.html)）  
$$
{\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
}
$$  

この定義を掘り下げると数学の話になってしまうので（自分もつらいので）、ここではソフトマックス関数の定義とシグモイド関数の多変量版であることだけ抑えておけばOKです。


### 交差エントロピー(クロスエントロピー)
ざっくり説明すると、「ある数値の軍団Aと、ある数値の軍団Bがどれくらい異なるか」を表す概念です。エントロピーとは、ざっくり「乱雑さの度合い」です。上下に分離したドレッシングの瓶をシャカシャカ振ってよく混ざった状態にすると「エントロピーが増えた」となります。機械学習でいうと「学習データの出力と、実際のニューラルネットワークの学習結果としての出力がどれくらい異なるか」を表します。この交差エントロピーを用いる際に比較対象となる「数値の軍団」は正規化されている必要があり、「ソフトマックス関数」等を使用して正規化を行う。


## TensorFlowのチュートリアルでは何をやっていたのか

まず、手書き数字画像のピクセルデータとラベルを学習・分析できるようにベクトルに変換。
変換したものを使用して、学習・分析を行います。

![ソフトマックス関数1](https://www.tensorflow.org/versions/master/images/softmax-regression-scalargraph.png)  

上で説明したとおり、シグモイドニューロンに近いネットワークを組み合わせたものを使用します。
シグモイドニューロンの説明をした時は、0~1の入力から0~1の出力を出すものでしたが、今回のチュートリアルでは、手書き数字画像が0なのか3なのか9なのかを判定したいため、出力を0~9にしたいです。そのため、ソフトマックス関数を通して結果を出力させます。

学習の際、出力した結果が間違っていたら、重みを修正して最適化していくことで、ニューラルネットワーク全体の最適化を行いいます。たくさんの出力から、ほしい答えと現在の出力の判定する際に考査エントロピーを使用し、「どれだけ異なるか」を見ます。

これをプログラムで実行したものです。

プログラムの流れは書きのqiitaの記事がわかりやすそうです。  
[http://qiita.com/EtsuroHONDA/items/79844b78655ccb3a7ae6](http://qiita.com/EtsuroHONDA/items/79844b78655ccb3a7ae6)

tensorflow全体のプログラムを確認したい場合は下記で確認可能です。  
[https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist)
