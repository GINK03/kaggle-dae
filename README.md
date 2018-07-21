# DenosingAutoEncoderによるオーバーフィットを防ぎつつ精度向上

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


### rank gauss
Rank　Gaussという連続値を特定の範囲の閉域に押し込めて、分布の偏りを解消する方法です。  
これも彼の言葉を頼りに実装していましたが、このようなコードになるかとおもいます。  
```python
df = pd.read_csv(fname)
print(df.head())
from scipy.special import erfinv
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
  #for s in series:
  #  print(s)
  df[c] = series
df.to_csv(oname, index=None)
```
流れとしては、まずランクを計算し、[0, 1]に押し込めて、平均を計算し、(-0.5,0.5)に変換します。  

これに対してerfiv関数をかけると、ランクの方よりが正規分布のような形変換することができます。  

## DAEパラメータ

## chainerで作成した学習コード

## 中間層の取り出し

## LightGBMで学習

## 学習結果

## 考察

