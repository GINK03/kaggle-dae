
from keras.layers               import Input, Dense, SimpleRNN, GRU, LSTM, CuDNNLSTM, RepeatVector
from keras.models               import Model
from keras.layers.core          import Flatten
from keras.callbacks            import LambdaCallback 
from keras.optimizers           import SGD, RMSprop, Adam
from keras.layers.wrappers      import Bidirectional as Bi
from keras.layers.wrappers      import TimeDistributed as TD
from keras.layers               import merge
from keras.applications.vgg16   import VGG16 
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.merge         import Concatenate
from keras.layers.core          import Dropout
from keras.layers.merge         import Concatenate as Concat
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.merge         import Dot,Multiply
from keras import backend as K
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan


input   = Input(shape=(229,))
encoded = Dense(229, activation='linear')(input)
#encoded = Dense(64, activation='relu')(encoded)
#encoded = Dense(32, activation='relu')(encoded)
#encoded = Dense(16, activation='relu')(encoded)

#decoded = Dense(32, activation='relu')(encoded)
#decoded = Dense(64, activation='relu')(decoded)
#decoded = Dense(128, activation='linear')(encoded)
decoded = Dense(229, activation='linear')(encoded)

dae = Model(input, decoded)
dae.compile(optimizer='adam', loss='mae')

def set_debugger_session():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    K.set_session(sess)
#set_debugger_session()
import pandas as pd

tdf = pd.read_csv('vars/one_hot_train.csv')
Tdf = pd.read_csv('vars/one_hot_test.csv')
df = pd.concat([tdf, Tdf], axis=0)

dae.fit(df.values, df.copy().values,
                epochs=100,
                batch_size=256,
                shuffle=True,)
                #validation_data=(x_test, x_test))

