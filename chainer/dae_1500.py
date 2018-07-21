import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy as np
import cupy as xp
from chainer import report
from chainer import Variable
from sklearn.metrics import mean_squared_error
# Network definition
class MLP(chainer.Chain):
  def __init__(self):
    super(MLP, self).__init__()
    with self.init_scope():
      # the size of the inputs to each layer will be inferred
      self.l1 = L.Linear(None, 1500)  # n_in -> n_units
      self.l2 = L.Linear(None, 1500)  # n_units -> n_units
      self.l3 = L.Linear(None, 1500)  # n_units -> n_units
      self.l4 = L.Linear(None, 227)   # n_units -> n_out
  def __call__(self, h):
    h1 = F.relu(self.l1(h))
    h2 = F.relu(self.l2(h1))
    h3 = F.relu(self.l3(h2))
    if is_predict:
      return np.hstack([h1.data, h2.data, h3.data])
    h4 = self.l4(h3) 
    return h4

from chainer.datasets import TupleDataset
import pandas as pd
import chainer.serializers
import sys
import swap_noise
is_predict = None
if '--train' in sys.argv:
  df = pd.read_csv('vars/one_hot_all.csv')
  df = df.set_index('id')
  df = df.drop(['target'], axis=1)
  EPOCHS = 2
  DECAY  = 0.995
  BATCH_SIZE = 128
  INIT_LR = 0.003
  model = L.Classifier(MLP(), lossfun=F.mean_squared_error)
  OPTIMIZER = chainer.optimizers.SGD(lr=INIT_LR)
  OPTIMIZER.setup(model)
  for cycle in range(300):
    noise = swap_noise.noise(df.values).astype(np.float32)
    train = TupleDataset(noise, df.values.astype(np.float32))
    test  = TupleDataset(noise[-10000:].astype(np.float32), df[-10000:].values.astype(np.float32))
    # iteration, which will be used by the PrintReport extension below.
    model.compute_accuracy = False
    chainer.backends.cuda.get_device_from_id(1).use()
    model.to_gpu()  # Copy the model to the GPU
    print(f'cycle {cycle-1:09d}')
    train_iter = chainer.iterators.SerialIterator(train , BATCH_SIZE, repeat=True)
    test_iter  = chainer.iterators.SerialIterator(test,  BATCH_SIZE, repeat=False, shuffle=False)
    updater = training.updaters.StandardUpdater(train_iter, OPTIMIZER, device=1)
    trainer = training.Trainer(updater, (EPOCHS, 'epoch'), out='outputs')
    trainer.extend(extensions.Evaluator(test_iter, model, device=1))
    trainer.extend(extensions.dump_graph('main/loss'))
    frequency = EPOCHS
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'elapsed_time', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    def lr_drop(trainer):
      trainer.updater.get_optimizer('main').lr *= DECAY
      print('now learning rate', trainer.updater.get_optimizer('main').lr)
    def save_model(trainer):
      chainer.serializers.save_hdf5(f'snapshot_15000_model_h5', model)
       
    trainer.extend(lr_drop, trigger=(1, 'epoch'))
    trainer.extend(save_model, trigger=(10, 'epoch'))
    trainer.run()
    model.to_cpu()  # Copy the model to the CPU
    mse1 = mean_squared_error( df[-10000:].values.astype(np.float32),  model.predictor(  noise[-10000:].astype(np.float32) ).data )
    mse2 = mean_squared_error( df[-10000:].values.astype(np.float32),  model.predictor( df[-10000:].values.astype(np.float32) ).data )
    print('mse1', mse1)
    print('mse2', mse2)
    chainer.serializers.save_hdf5(f'models/model_{cycle:09d}_{mse1:0.09f}_{mse2:0.09f}.h5', model)
  chainer.serializers.save_hdf5(f'models/model_1500.h5', model)

if '--predict' in sys.argv:
  tdf = pd.read_csv('../michael/vars/one_hot_train.csv')
  Tdf = pd.read_csv('../michael/vars/one_hot_test.csv')
  df = pd.concat([tdf, Tdf], axis=0)
  df = df.set_index('id')
  df = df.drop(['target'], axis=1)
  #train = TupleDataset(df.values.astype(np.float32), df.values.astype(np.float32))
  model = L.Classifier(MLP(), lossfun=F.mean_squared_error)
  chainer.serializers.load_hdf5('models/model_000000199_0.007169580_0.001018013.h5', model)

  chainer.backends.cuda.get_device_from_id(0).use()
  model.to_cpu()  # Copy the model to the CPU

  BATCH_SIZE = 512
  #eval_iter = chainer.iterators.SerialIterator(train , BATCH_SIZE)
  
  print( df.values.shape )
  height, width = df.values.shape
  #for ev in eval_iter:
  #  model.predictor(Variable(ev))
  is_predict = True
  
  args = [(k, split) for k, split in enumerate(np.split(df.values.astype(np.float32), list(range(0, height, 10000)) + [height] ))]

  for k, split in args:
    r = model.predictor( split ).data
    if r.shape[0] == 0:
      continue
    np.save(f'dumps/{k:04d}', r) 
    print(r.shape)
