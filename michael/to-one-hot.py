import pandas as pd
import re
df = pd.read_csv('vars/rank_gauss_all.csv')
all_size = len(df)
for c in list(df.columns):
  if not re.search(r'cat$', c): 
    continue
  print(c)
  series = df[ c ].apply(str)
  _df = pd.get_dummies(series, prefix=f'{c}')
  print(_df.head())
  print(len(_df))
  df = df.drop([c], axis=1)
  df = pd.concat([df, _df], axis=1)

df.to_csv('vars/one_hot_all.csv', index=None)
