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
e = Dense(1500, activation='relu')(input)
e = Dense(1500, activation='relu')(e)
d = Dense(1500, activation='relu')(e)
#d = Dense(15000, activation='relu')(d)
d = Dense(227, activation='linear')(d)

dae = Model(input, d)
loss = lambda y_true, y_pred: 1000 * losses.mse(y_true, y_pred)
init_lr = 0.001

dae.compile(optimizer=Adam(lr=init_lr, decay=0.001), loss=loss)


dae.load_weights('vars/dae_deep_stack_sgd_200.h5')

from keras import backend as K
import numpy as np

inp      = dae.input                                                          # input placeholder
outputs  = [layer.output for layer in dae.layers]                             # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions


tdf = pd.read_csv('vars/one_hot_train.csv')
Tdf = pd.read_csv('vars/one_hot_test.csv')

#df = pd.concat([tdf, Tdf], axis=0)

for (df, oname) in [(tdf, 'train'), (Tdf, 'test')]:
  df = df.set_index('id')
  try:
    df = df.drop(['target'], axis=1)
  except Exception: ...
  
  size = len(df)
  for k, split in enumerate(np.split(df.values, list(range(0, size, 1000)) + [size] )):
    test       = split #df.values
    layer_outs = [func([test]) for func in functors]
    print(len(test))

    fs =  [ layer_outs[li][0] for li in  range(1, len(layer_outs)-1) ] 
    print(len(fs))
    fs = list(map(list, zip(*fs)))
    for i in range(len(fs)):
      fs[i] = np.hstack(fs[i])
      #print(fs[i].shape)
    fs = np.array(fs)
    print(len(fs))
    if len(fs) == 0:
      continue
    #fs = np.hstack(fs)
    print( fs.shape )
    np.save(f'vars/flatten_sdg_{oname}_{k:09d}_{len(split)}', fs)
