import pandas as pd

for fname, oname in [('../input/train.csv', 'vars/rank_gauss_train.csv'), ('../input/test.csv', 'vars/rank_gauss_test.csv')]:
  df = pd.read_csv(fname)
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
    #for s in series:
    #  print(s)
    df[c] = series

  df.to_csv(oname, index=None)

