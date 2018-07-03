import pandas as pd
import re
tdf = pd.read_csv('vars/rank_gauss_train.csv')
Tdf = pd.read_csv('vars/rank_gauss_test.csv')
train_size = len(tdf)
df = pd.concat([tdf, Tdf], axis=0)
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

df[:train_size].to_csv('vars/one_hot_train.csv', index=None)
df[train_size:].to_csv('vars/one_hot_test.csv', index=None)
