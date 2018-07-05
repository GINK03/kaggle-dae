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
e = Dense(227, activation='linear')(input)
e = Dense(1500, activation='relu')(e)
e = Dense(1500, activation='relu')(e)
e = Dense(1500, activation='relu')(e)
d = Dense(1500, activation='relu')(e)
#d = Dense(15000, activation='relu')(d)
d = Dense(227, activation='linear')(d)

dae = Model(input, d)
loss = lambda y_true, y_pred: 1000 * losses.mse(y_true, y_pred)
dae.compile(optimizer=SGD(lr=0.008, decay=0.001), loss=loss)

def set_debugger_session():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    K.set_session(sess)


tdf = pd.read_csv('vars/one_hot_train.csv')
Tdf = pd.read_csv('vars/one_hot_test.csv')
df = pd.concat([tdf, Tdf], axis=0)

df = df.set_index('id')
df = df.drop(['target'], axis=1)

from sklearn.cross_validation import KFold
import swap_noise

#for k in range(100):
NFOLDS=1000
SEED=777
kf = KFold(len(df.values), n_folds=NFOLDS, shuffle=True, random_state=SEED)
decay = 0.001
for i, (train_index, test_index) in enumerate(kf):

    noised = swap_noise.noise(df.values)
    dae.fit(noised[train_index], df.values[train_index],
                    epochs=1,
                    validation_data=(noised[test_index], df.values[test_index]),
                    batch_size=512,
                    shuffle=True,)
    next_lr = K.get_value(dae.optimizer.lr)*(1.0-decay)
    K.set_value(dae.optimizer.lr, next_lr)
    print('now epoch', i, 'next_lr', next_lr)

dae.save_weights('vars/dae_deep_stack.h5')
