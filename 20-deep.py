from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Lambda, Input, Activation, Dropout, Flatten, Dense, Reshape, merge
from keras.layers import Concatenate, Multiply, Conv1D, MaxPool1D, BatchNormalization
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.core import Dropout
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.callbacks            import LambdaCallback 

import copy
import sys
import numpy as np
import pickle
import time

input_tensor = Input( shape=(228,) )

x = input_tensor
x = Dense(500, activation='relu')(x)
x = Dense(500, activation='relu')(x)
x = Dense(500, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)


model = Model(inputs=input_tensor, outputs=x)

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005), metrics=['acc'])
buff = None
now  = time.strftime("%H_%M_%S")
def callback(epoch, logs):
  global buff
  buff = copy.copy(logs)
batch_callback = LambdaCallback(on_epoch_end=lambda batch,logs: callback(batch,logs) )

if '--train' in sys.argv:
  Xs, ys  = pickle.load(open('metas/data.pkl', 'rb'))  

  for i in range(100):
    model.fit( Xs, ys, validation_split=0.2, epochs=1, batch_size=64, callbacks=[batch_callback] )
    val_loss = buff['val_loss']
    acc = buff['acc']

    model.save(f'models/{acc:0.12f}_{val_loss:0.12f}_{i:09d}.h6')
