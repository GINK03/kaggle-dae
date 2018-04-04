import os

import math

import csv

import json

import re

import sys

import numpy as np

import pickle
HOME = os.environ['HOME']
print(HOME)

# 極端に大きい値があるわけでないので、数字は加工しない
# 特徴量の数（カテゴリ変数をflatten）して確認する
# targetはクレームがあったかどうか
# idはidなので消す

if '--step1' in sys.argv:
  fp = open(f'{HOME}/.kaggle/competitions/porto-seguro-safe-driver-prediction/train.csv')
  feats = set()
  heads = next(fp).strip().split(',')
  for line in fp:
    vals = line.strip().split(',')
    obj = dict( zip(heads, vals) )
    del obj['target']
    del obj['id']
    #print(json.dumps(obj, indent=2))
    for key, val in obj.items():
      if re.search('_cat$', key) is not None:
        feat = f'{key}:{val}'
      elif val == '-1' : # 欠損値は-1だそうだ
        feat = f'{key}:None'
      else:
        feat = f'{key}' 
      feats.add(feat)

  print(feats)

  feat_index = {feat:index for index, feat in enumerate(feats)}

  json.dump(feat_index, fp=open('metas/feat_index.json', 'w'), indent=2)

# 行列構造に変形する
if '--step2' in sys.argv:

  Xs, ys = [], []

  feat_index = json.load(fp=open('metas/feat_index.json'))
  fp = open(f'{HOME}/.kaggle/competitions/porto-seguro-safe-driver-prediction/train.csv')
  heads = next(fp).strip().split(',')
  for line in fp:

    x = [0.0]*len(feat_index)
    vals = line.strip().split(',')
    obj = dict( zip(heads, vals) )

    y = float(obj['target'])

    del obj['target']
    del obj['id']

    for key, val in obj.items():
      if re.search('_cat$', key) is not None:
        feat = f'{key}:{val}'
        val = 1.0
      elif val == '-1' : # 欠損値は-1だそうだ
        feat = f'{key}:None'
        val = 1.0
      else:
        feat = f'{key}' 
        val = float(val)

      index = feat_index[ feat ]
      x[ index ]  = val 

    Xs.append( x )
    ys.append( y )

  Xs, ys = np.array(Xs), np.array(ys)
  meta = {'Xs.shape':list(Xs.shape), 'ys.shaope':ys.shape}
  json.dump(meta, fp=open('metas/meta.json', 'w'), indent=2)
  print(ys.shape)
  open('metas/data.pkl', 'wb').write( pickle.dumps( (Xs, ys) ) )

