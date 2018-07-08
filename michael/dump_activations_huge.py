from keras.layers                   import Input, Dense, SimpleRNN, GRU, LSTM, CuDNNLSTM, RepeatVector
from keras.models                 import Model
from keras.layers.core           import Flatten
from keras.callbacks              import LambdaCallback 
from keras.optimizers           import SGD, RMSprop, Adam
from keras.layers.wrappers      import Bidirectional as Bi
from keras.layers.wrappers      import TimeDistributed as TD
from keras.layers                     import merge
from keras.applications.vgg16   import VGG16 
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise           import GaussianNoise as GN
from keras.layers.merge         import Concatenate
from keras.layers.core            import Dropout
from keras.layers.merge         import Concatenate as Concat
from keras.layers.noise           import GaussianNoise as GN
from keras.layers.merge         import Dot,Multiply
from keras import backend as K
from keras import losses
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
import pandas as pd

input   = Input(shape=(227,))
e = Dense(10000, activation='relu')(input)
e = Dense(10000, activation='relu')(e)
intermediate = Dense(3000, activation='linear')(e)
e = Dense(10000, activation='relu')(intermediate)
e = Dense(10000, activation='relu')(e)
output = Dense(227, activation='linear')(e)

dae = Model(input, output)
loss = lambda y_true, y_pred: 1000 * losses.mse(y_true, y_pred)
init_lr = 0.001

dae.compile(optimizer=Adam(lr=init_lr, decay=0.001), loss=loss)

dae.load_weights('vars/dae_deep_huge_adam_80_000000075.h5')

from keras import backend as K
import numpy as np

get_int_output = K.function([dae.layers[0].input], [dae.layers[3].output])

tdf = pd.read_csv('vars/one_hot_train.csv')
Tdf = pd.read_csv('vars/one_hot_test.csv')

#df = pd.concat([tdf, Tdf], axis=0)
import os
def pmap(k, oname, split):
  if os.path.exists(f'vars/flatten_adam_huge_{oname}_{k:09d}_{len(split)}'):
    return
  test       = split #df.values
  fs = get_int_output([test])[0]
  print(len(test))
  print(len(fs))
  fs = list(map(list, zip(*fs)))
  for i in range(len(fs)):
    fs[i] = np.hstack(fs[i])
    #print(fs[i].shape)
  fs = np.array(fs)
  print(len(fs))
  if len(fs) == 0:
    return
  #fs = np.hstack(fs)
  print( fs.shape )
  np.save(f'vars/flatten_adam_huge_{oname}_{k:09d}_{len(split)}', fs)
import random
import swap_noise
for (df, oname) in [(tdf, 'train'), (Tdf, 'test')]:
  df = df.set_index('id')
  try:
    df = df.drop(['target'], axis=1)
  except Exception: ...
  
  size = len(df)
  args = [(k, oname, split) for k, split in enumerate(np.split(swap_noise.noise(df.values), list(range(0, size, 10000)) + [size] ))]
  random.shuffle(args)
  [pmap(*arg) for arg in args]
