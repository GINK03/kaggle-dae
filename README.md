# テーブルデータに対して、DenosingAutoEncoderによるオーバーフィットを防ぎつつ精度向上

Kaggle Porto Seguroの1st placeソリューションの分析と追試

## データセットの問題
Kaggle Porto Seguroでは問題となっている点があって、テストデータとトレインデータのサイズがテストが大きく、トレインだけに着目してしまうと、LeaderBoardにoverfitしてしまう問題があります。  

これはトレインだけで学習するために起こる問題で、正直、Kaggle以外ではこのようなことがあまり起きないと思うのですが、テストデータ・セットを有意義に使う方法として、教師なし学習でまずは次元変換やなんやらを行うという方法が有効なようです。  

ディープを用いることでいくつか有益な変換操作を行うことができて、「すべての情報は何らかのノイズを受けている」という視点に立ったときに、恣意的にAutoEncoderの入力にノイズを乗せ、それを除去するように学習するとはっきりと、物事が観測できるようになったりするという特徴を利用しています。
<div align="center">
  <img width="600px" src="https://d2mxuefqeaa7sj.cloudfront.net/s_395C846F6BB54334ACB188FAC2F01C0FF7D15E56852EC0E8EFD1BA2A22439502_1532149832729_image.png">
</div>
<div align="center"> 図1. よくある画像の例 </div>

画像の利用からテーブルデータの利用に変換する操作を行います。  

この用途はあまり見たことがなかったので、有益でした。（画像にイメージがかかっていますが実際は値に対してかかります）
<div align="center">
  <img width="680px" src="https://d2mxuefqeaa7sj.cloudfront.net/s_395C846F6BB54334ACB188FAC2F01C0FF7D15E56852EC0E8EFD1BA2A22439502_1532150414056_image.png">
</div>
<div align="center"> 図2. テーブルデータのノイズを除去 </div>

## MichaelさんのとったDAE(DenosingAutoEncoder)の特徴

### noiseを掛ける方法
swap　noiseという方法を用います。これは、uniformやgaussian noiseをこれらにかけても適切ではないという点を抱えているため、いくつかのハッキーな方法が取られています。  
swap noiseはランダムに10%程度の確率で、"同じ列"で"他の行"と入れ替える技で、これによりノイズを掛けます。  

これをすべての要素にたいして適応するすると、割と現実的なnoisingになるそうです。  
<div align="center">
  <img width="680px" src="https://d2mxuefqeaa7sj.cloudfront.net/s_41B02D2D66C0D76C571B951DD8B34CC4006073F98B54F9233C265E9EDEABCBB8_1530768386174_image.png">
</div>
<div align="center"> 図3. swap noise </div>

numpyのアレイをコピーしてすべての要素を操作していって、　10%の確率で"同じ列"、"別の行"と入れ替えます
```python
import numpy as np
import random
from numba.decorators import jit
@jit
def noise(array):
  print('now noising') 
  height = len(array)
  width = len(array[0])
  print('start rand')  
  rands = np.random.uniform(0, 1, (height, width) )
  print('finish rand')  
  copy  = np.copy(array)
  for h in range(height):
    for w in range(width):
      if rands[h, w] <= 0.10:
        swap_target_h = random.randint(0,h)
        copy[h, w] = array[swap_target_h, w]
  print('finish noising') 
  return copy
```

### rank gauss
Rank　Gaussという連続値を特定の範囲の閉域に押し込めて、分布の偏りを解消する方法です。  
これも彼の言葉を頼りに実装していましたが、このようなコードになるかとおもいます。  
```python
import pandas as pd
tdf = pd.read_csv('../input/train.csv')
Tdf = pd.read_csv('../input/test.csv')
df = pd.concat([tdf, Tdf], axis=0)
print(df.head())
# catが最後につくとカテゴリ
# binがつくとワンホット
# 何もつかないと、連続値
from scipy.special import erfinv
import re
## to_rank
for c in df.columns:
  if c in ['id', 'target'] or re.search(r'cat$', c) or 'bin' in c:
    continue
  series = df[c].rank()
  M = series.max()
  m = series.min() 
  print(c, m, len(series), len(set(df[c].tolist())))
  series = (series-m)/(M-m)
  series = series - series.mean()
  series = series.apply(erfinv) 
  df[c] = series
df.to_csv('vars/rank_gauss_all.csv', index=None)
```
流れとしては、まずランクを計算し、[0, 1]に押し込めて、平均を計算し、(-0.5,0.5)に変換します。  

これに対してerfiv関数をかけると、ランクの方よりが正規分布のような形変換することができます。  

## DAEパラメータ

<div align="center">
  <img width="680px" src="https://i.imgur.com/z62wCWj.png&hmac=24dHVrGgJIaH4WjlpR8LlMzfhKPampPNSpQg8rLA5Fg=">
</div>
<div align="center"> 図4. michaelさんが調整したパラメータ </div>

このように、何種類かのDenosing AutoEncoderをアンサンブルして、正則化を強めに設定したLGBで学習する.  

## chainerで作成した学習コード
長くなりますので、全体参照には、githubを参照してください。  
```python
from chainer import training
from chainer.training import extensions
import numpy as np
import cupy as xp
from chainer import report
from chainer import Variable
from sklearn.metrics import mean_squared_error
# Network definition
class MLP(chainer.Chain):
  def __init__(self):
    super(MLP, self).__init__()
    with self.init_scope():
      self.l1 = L.Linear(None, 1500)  # n_in -> n_units
      self.l2 = L.Linear(None, 1500)  # n_units -> n_units
      self.l3 = L.Linear(None, 1500)  # n_units -> n_units
      self.l4 = L.Linear(None, 227)   # n_units -> n_out
  def __call__(self, h):
    h1 = F.relu(self.l1(h))
    h2 = F.relu(self.l2(h1))
    h3 = F.relu(self.l3(h2))
    if is_predict:
      return np.hstack([h1.data, h2.data, h3.data])
    h4 = self.l4(h3) 
    return h4
```
学習部分
```python
from chainer.datasets import TupleDataset
import pandas as pd
import chainer.serializers
import sys
import swap_noise
is_predict = None
if '--train' in sys.argv:
  df = pd.read_csv('vars/')
  df = df.set_index('id')
  df = df.drop(['target'], axis=1)
  EPOCHS = 2
  DECAY  = 0.995
  BATCH_SIZE = 128
  INIT_LR = 0.003
  model = L.Classifier(MLP(), lossfun=F.mean_squared_error)
  OPTIMIZER = chainer.optimizers.SGD(lr=INIT_LR)
  OPTIMIZER.setup(model)
  for cycle in range(300):
    noise = swap_noise.noise(df.values).astype(np.float32)
    train = TupleDataset(noise, df.values.astype(np.float32))
    test  = TupleDataset(noise[-10000:].astype(np.float32), df[-10000:].values.astype(np.float32))
    # iteration, which will be used by the PrintReport extension below.
    model.compute_accuracy = False
    chainer.backends.cuda.get_device_from_id(0).use()
    model.to_gpu()  # Copy the model to the GPU
    print(f'cycle {cycle-1:09d}')
    train_iter = chainer.iterators.SerialIterator(train , BATCH_SIZE, repeat=True)
    test_iter  = chainer.iterators.SerialIterator(test,  BATCH_SIZE, repeat=False, shuffle=False)
    updater = training.updaters.StandardUpdater(train_iter, OPTIMIZER, device=0)
    trainer = training.Trainer(updater, (EPOCHS, 'epoch'), out='outputs')
    trainer.extend(extensions.Evaluator(test_iter, model, device=0))
    trainer.extend(extensions.dump_graph('main/loss'))
    frequency = EPOCHS
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'elapsed_time', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    def lr_drop(trainer):
      trainer.updater.get_optimizer('main').lr *= DECAY
      print('now learning rate', trainer.updater.get_optimizer('main').lr)
    def save_model(trainer):
      chainer.serializers.save_hdf5(f'snapshot_15000_model_h5', model)
       
    trainer.extend(lr_drop, trigger=(1, 'epoch'))
    trainer.extend(save_model, trigger=(10, 'epoch'))
    trainer.run()
    model.to_cpu()  # Copy the model to the CPU
    mse1 = mean_squared_error( df[-10000:].values.astype(np.float32),  model.predictor(  noise[-10000:].astype(np.float32) ).data )
    mse2 = mean_squared_error( df[-10000:].values.astype(np.float32),  model.predictor( df[-10000:].values.astype(np.float32) ).data )
    print('mse1', mse1)
    print('mse2', mse2)
    chainer.serializers.save_hdf5(f'models/model_{cycle:09d}_{mse1:0.09f}_{mse2:0.09f}.h5', model)
  chainer.serializers.save_hdf5(f'models/model_1500.h5', model)
```

## 中間層の取り出し
データがそれなりに多いので、CPUで適当なサイズに切り出して予想します  

npyファイル形式にチャンクされたファイルがダンプされます  
```python
if '--predict' in sys.argv:
  df = pd.read_csv('vars/one_hot_all.csv')
  df = df.set_index('id')
  df = df.drop(['target'], axis=1)
  model = L.Classifier(MLP(), lossfun=F.mean_squared_error)
  chainer.serializers.load_hdf5('models/model_000000199_0.007169580_0.001018013.h5', model)
  chainer.backends.cuda.get_device_from_id(0).use()
  model.to_cpu()  # Copy the model to the CPU
  BATCH_SIZE = 512
  print( df.values.shape )
  height, width = df.values.shape
  is_predict = True
  args = [(k, split) for k, split in enumerate(np.split(df.values.astype(np.float32), list(range(0, height, 10000)) + [height] ))]
  for k, split in args:
    r = model.predictor( split ).data
    if r.shape[0] == 0:
      continue
    np.save(f'dumps/{k:04d}', r) 
    print(r.shape)
```

## LightGBMで学習
```console
$ python3 lgb-join-huge.py --kfold
```

## 結果
michaelさんのネットワークは５つのモデルのアンサンブルで、この個数を行うのは割と容易ではないことです。  

そのため、一つのディープスタックネットワークで行いましたが、`loglossが`と完全には追いついていません。  

しかし、Late Sumissionの結果では一つのモデルに対してBenchMakrとPure LightGBM(前処理など最低限)とDAEでこの様になっており、簡単には精度に対して寄与しないということがわかっています。  

**Random Forest benchmark**
```
Public 0.23824
```

**Pure LightGBM**
```
Private 0.28540
Public  0.28091
```

**DAE + LightGBM**
※ Leaves, Depth, 正則化などのパラメータを再調整する必要があります  
```
Private 0.28910
Public  0.28112
```

**DAE + Dense\*2 + Adam**
```
Private 0.26320
Public  0.25920
```

**DAE + Dense\*2 + SGD**
* Parameter
```
Private 0.23926
Public  0.23258
```


## 注意
 中間層を吐き出して、それをもとに再学習する操作が、想像以上にメモリを消耗するので、96GBのマシンと49GBのマシンの２つ必要でした。  
 軽い操作ではないです。
 
 Deepあるあるだとは思うのですが、入れるデータによっても、解くべき問題によっても微妙にパラメータを調整する箇所が多く、膨大な試行錯誤が伴います。
 
 また、何分、手触り感がよくわからない操作も多く、フレームワークのせいでうまくいかないのかバグなのか切り出しが不透明であったため、kerasとchainerの双方で実装することになり、他人のエッジ領域の技術を再現するのはまぁ難しいねという気持ちです。（勉強になりました）

## プログラム

**rank-gauss.py**  
連続値や1hot表現をランクガウスに変換  

**swap_noise.py**  
テーブルデータを提案に従って、スワップノイズをかけます。  
これは他のプログラムからライブラリとして利用されます。  

**dae_1500_sgd.py**  
OptimizerをSGDで学習するDAE(1500の全結合層)  

**dae_1500_adam.py**  
OptimizerをAdamで学習するDAE(1500の全結合層)  


## 考察
 テーブルデータも何らかの確率的な振る舞いをしていて、事象の例外などの影響を受けるとき、このときDenosing AutoEncoderでノイズを除去するようにより一般的て、汎用的な表現に変換できるのかもしれません。  　
 
 michaleさんも膨大な試行錯誤のもと、この方法がうまく行ったということであったので、仮説があり、確定的な理論で、よってこれが強いというわけではないのだと思います。  
 
 様々に微調整したディープラーニングしたモデルと、LightGBMを線形的なアルゴリズムでアンサンブルすると、うまくいくというものでした。
