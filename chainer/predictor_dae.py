import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

import numpy as np
import cupy as xp
from chainer import report
from chainer import Variable
from sklearn.metrics import mean_squared_error
import gc
# Network definition
class MLP(chainer.Chain):
  def __init__(self):
    super(MLP, self).__init__()
    with self.init_scope():
      self.l1 = L.Linear(None, 1000)  # n_in -> n_units
      self.l2 = L.Linear(None, 1000)  # n_units -> n_units
      self.l3 = L.Linear(None, 1)     # n_units -> n_out
  def __call__(self, h):
    h1 = F.dropout(F.relu(self.l1(h)),  ratio=0.3)
    h2 = F.dropout(F.relu(self.l2(h1)), ratio=0.3)
    h3 = F.sigmoid(self.l3(h2))
    return h3

from chainer.datasets import TupleDataset
import pandas as pd
import chainer.serializers
import sys, glob
import swap_noise
from sklearn.cross_validation import KFold
import numpy as np
import keras
from keras.layers          import Dropout
from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, multiply
from keras.regularizers    import l2
from keras.layers.core     import Reshape
from keras.layers.normalization import BatchNormalization as BN
import keras.backend as K

def build_model(input_size=4500):
  ix = Input(shape=(input_size,))
  x = Dense(1000, activation='relu')(ix)
  x = Dropout(0.5)(x)
  x = Dense(1000, activation='relu')(x)
  #x = Dropout(0.5)(x)
  x = Dense(1, activation='sigmoid')(x)
  
  model = Model(ix, x)
  model.compile(optimizer=Adam(lr=0.0005), loss='binary_crossentropy')
  return model

def get_oof(clf, x_train, y, x_test):
  NFOLDS=10
  SEED=71
  kf = KFold(len(x_train), n_folds=NFOLDS, shuffle=True, random_state=SEED)
  oof_train = np.zeros((len(x_train),))
  oof_test = np.zeros((len(x_test),))
  EPOCHS     = 100
  DECAY      = 0.995
  BATCH_SIZE = 128
  INIT_LR    = 0.01
  print(x_train.shape)
  print(y.shape)
  oof_test_skf = np.empty((1, len(x_test)))
  for i, (train_index, test_index) in enumerate(kf):
    print('\nFold {}'.format(i))
    #x_tr = x_train[train_index]
    #y_tr = y[train_index]
    x_tr = x_train
    y_tr = y

    y_te = y[test_index]
    x_te = x_train[test_index]
    model = build_model(input_size)
    # - es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    # - model.fit(x_tr, y_tr, validation_data=(x_te, y_te), batch_size=1024, epochs=EPOCHS, callbacks=[es_callback])
    for ii in range(300):
      model.fit(x_tr, y_tr, validation_data=(x_te, y_te), batch_size=128*10, epochs=1, callbacks=[])
      print('now lr', ii, K.get_value(model.optimizer.lr) )
      K.set_value( model.optimizer.lr,  K.get_value(model.optimizer.lr) * 0.98 )
    xx = model.predict(x_te)
    oof_train[test_index] = xx.reshape(len(xx))
    xx = np.array(model.predict(x_test))
    oof_test_skf[i, :]    = xx.reshape(len(xx)) 
    del (xx, model, x_tr, x_te); gc.collect()
    break
  oof_test[:] = oof_test_skf.mean(axis=0)
  return oof_train, oof_test

from sklearn.metrics import log_loss
if '--util' in sys.argv:
  adf    = np.vstack( [np.load(fn) for fn in sorted(glob.glob('dumps/*.npy')) ] )
  np.save('vars/adf', adf)

if '--util2' in sys.argv:
  _df    = pd.read_csv('./vars/one_hot_all.csv')
  adf    = np.load('vars/adf.npy')

  a2df   = np.hstack([adf, _df.values])
  np.save('vars/a2df', a2df)

if '--train' in sys.argv:
  tdf    = pd.read_csv('../input/train.csv')
  tdf    = tdf.set_index('id')

  target = tdf['target'].values

  target = target.reshape(len(target), 1)
  #adf    = np.vstack( [np.load(fn) for fn in sorted(glob.glob('dumps/*.npy'))[:60] ] )
  if '--org' in sys.argv:
    adf    = pd.read_csv('vars/one_hot_all.csv').values
    input_size = 229
  else:
    adf    = np.load('vars/adf.npy')
    input_size = 4500
    
  Tdf    = adf[len(target):] 
  adf    = adf[:len(target)]
  print(adf.shape)
  print(target.shape)
  print(Tdf.shape)
  sub   = pd.read_csv('../input/sample_submission.csv')
  print(sub.shape)
  oof_train, oof_test = get_oof(None, adf, target, Tdf)

  from sklearn.metrics import log_loss
  print(oof_train.shape)
  print(tdf['target'].values.shape)
  loss = log_loss(tdf['target'].values, oof_train) 
  print('loss', loss)

  sub   = pd.read_csv('../input/sample_submission.csv')
  sub['target'] = oof_test
  sub.to_csv('sub_dae.csv', index=False)

  print('output shape', sub.shape)
