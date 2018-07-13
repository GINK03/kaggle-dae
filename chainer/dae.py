import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy as np

from chainer import report
from chainer import Variable
# Network definition
class MLP(chainer.Chain):
  def __init__(self):
    super(MLP, self).__init__()
    with self.init_scope():
      # the size of the inputs to each layer will be inferred
      self.l1 = L.Linear(None, 7000)  # n_in -> n_units
      self.l2 = L.Linear(None, 7000)  # n_units -> n_units
      self.ll = L.Linear(None, 3000)  # n_units -> n_units
      self.l3 = L.Linear(None, 7000)  # n_units -> n_units
      #self.l4 = L.Linear(None, 5000)  # n_units -> n_units
      self.l4 = L.Linear(None, 227)  # n_units -> n_out
  def __call__(self, h):
    h = F.relu(self.l1(h))
    h = F.relu(self.l2(h))
    h = F.relu(self.ll(h))
    if is_predict:
      #for data in h:
      #  print(data.data)
      return h
    h = F.relu(self.l3(h))
    h = self.l4(h) 
    return h

from chainer.datasets import TupleDataset
import pandas as pd
import chainer.serializers
import sys
import swap_noise
from sklearn.metrics import mean_squared_error

is_predict = None
if '--train' in sys.argv:
  tdf = pd.read_csv('../michael/vars/one_hot_train.csv')
  Tdf = pd.read_csv('../michael/vars/one_hot_test.csv')
  df = pd.concat([tdf, Tdf], axis=0)
  df = df.set_index('id')
  df = df.drop(['target'], axis=1)

  EPOCHS = 1
  DECAY  = 0.995
  BATCH_SIZE = 512*2
  INIT_LR = 0.001

  OPTIMIZER = chainer.optimizers.Adam(alpha=INIT_LR)

  # iteration, which will be used by the PrintReport extension below.
  model = L.Classifier(MLP(), lossfun=F.mean_squared_error)
  model.compute_accuracy = False

  chainer.backends.cuda.get_device_from_id(0).use()
  model.to_gpu()  # Copy the model to the GPU
  OPTIMIZER.setup(model)
  
  for i in range(200):
    noise = swap_noise.noise(df.values).astype(np.float32)
    train = TupleDataset(noise, df.values.astype(np.float32))
    test  = TupleDataset(noise[-10000:].astype(np.float32), df[-10000:].values.astype(np.float32))
    train_iter = chainer.iterators.SerialIterator(train , BATCH_SIZE, repeat=True)
    test_iter  = chainer.iterators.SerialIterator(test,  BATCH_SIZE, repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(train_iter, OPTIMIZER, device=0)
    trainer = training.Trainer(updater, (EPOCHS, 'epoch'), out='outputs')
    trainer.extend(extensions.Evaluator(test_iter, model, device=0))
    trainer.extend(extensions.dump_graph('main/loss'))
    #frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    frequency = EPOCHS
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'elapsed_time', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    def lr_drop(trainer):
      trainer.updater.get_optimizer('main').alpha *= DECAY
      print('now learning rate', trainer.updater.get_optimizer('main').alpha)
    def save_model(trainer):
      chainer.serializers.save_hdf5(f'snapshot_model_h5', model)
       
    trainer.extend(lr_drop, trigger=(1, 'epoch'))
    trainer.extend(save_model, trigger=(10, 'epoch'))

    trainer.run()
    
    model.to_cpu()
    mse1 = mean_squared_error( df[-10000:].values.astype(np.float32),  model.predictor(  noise[-10000:].astype(np.float32) ).data )
    mse2 = mean_squared_error( df[-10000:].values.astype(np.float32),  model.predictor( df[-10000:].values.astype(np.float32) ).data )
    print('mse1', mse1)
    print('mse2', mse2)
    chainer.serializers.save_hdf5(f'models/model_{i:09d}_{mse1}_{mse2}.h5', model)

if '--predict' in sys.argv:
  tdf = pd.read_csv('../michael/vars/one_hot_train.csv')
  Tdf = pd.read_csv('../michael/vars/one_hot_test.csv')
  df = pd.concat([tdf, Tdf], axis=0)
  df = df.set_index('id')
  df = df.drop(['target'], axis=1)
  #train = TupleDataset(df.values.astype(np.float32), df.values.astype(np.float32))
  model = L.Classifier(MLP(), lossfun=F.mean_squared_error)
  chainer.serializers.load_hdf5('model.h5', model)

  chainer.backends.cuda.get_device_from_id(0).use()
  model.to_cpu()  # Copy the model to the GPU

  BATCH_SIZE = 512
  #eval_iter = chainer.iterators.SerialIterator(train , BATCH_SIZE)
  
  print( df.values.shape )
  height, width = df.values.shape
  #for ev in eval_iter:
  #  model.predictor(Variable(ev))
  is_predict = True
  r = model.predictor( df.values[:10].astype(np.float32) )
  print(r.data.shape)
