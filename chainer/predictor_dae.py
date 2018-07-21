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
      self.l1 = L.Linear(None, 4500)  # n_in -> n_units
      self.l2 = L.Linear(None, 1000)  # n_in -> n_units
      self.l3 = L.Linear(None, 1)   # n_units -> n_out
  def __call__(self, h):
    h1 = F.relu(self.l1(h))
    h2 = self.l2(h1) 
    h3 = self.l3(h2) 
    return h3

from chainer.datasets import TupleDataset
import pandas as pd
import chainer.serializers
import sys, glob
import swap_noise
is_predict = None

if '--train' in sys.argv:
  tdf    = pd.read_csv('../input/train.csv')
  tdf    = tdf.set_index('id')
  target = tdf['target'].values
  target = target.reshape(len(target), 1)
  adf    = np.vstack( [np.load(fn) for fn in sorted(glob.glob('dumps/*.npy'))[:60] ] )
  adf    = adf[:len(target)]
  
  print(tdf.shape)
  print(adf.shape)
  EPOCHS     = 200
  DECAY      = 0.995
  BATCH_SIZE = 128
  INIT_LR    = 0.003
  model      = L.Classifier(MLP(), lossfun=F.sigmoid_cross_entropy)
  OPTIMIZER  = chainer.optimizers.SGD(lr=INIT_LR)
  OPTIMIZER.setup(model)
  
  train = TupleDataset(adf, target)
  test  = TupleDataset(adf, target)
  # iteration, which will be used by the PrintReport extension below.
  model.compute_accuracy = False
  chainer.backends.cuda.get_device_from_id(0).use()
  model.to_gpu()  # Copy the model to the GPU
  train_iter = chainer.iterators.SerialIterator(train , BATCH_SIZE, repeat=True)
  test_iter  = chainer.iterators.SerialIterator(test,  BATCH_SIZE, repeat=False, shuffle=False)
  updater    = training.updaters.StandardUpdater(train_iter, OPTIMIZER, device=0)
  trainer    = training.Trainer(updater, (EPOCHS, 'epoch'), out='outputs')
  trainer.extend(extensions.Evaluator(test_iter, model, device=0))
  trainer.extend(extensions.dump_graph('main/loss'))
  frequency  = EPOCHS
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
  chainer.serializers.save_hdf5(f'models/predictor_model_4500.h5', model)
