from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb
from sklearn.cross_validation import KFold

import numpy as np
def get_oof(clf, x_train, y, x_test):
    NFOLDS=5
    SEED=71
    kf = KFold(len(x_train), n_folds=NFOLDS, shuffle=True, random_state=SEED)
    oof_train = np.zeros((len(x_train),))
    oof_test = np.zeros((len(x_test),))
    oof_test_skf = np.empty((NFOLDS, len(x_test)))
    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        # 'max_depth': 15,
        'num_leaves': 10,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.75,
        'bagging_freq': 4,
        'learning_rate': 0.016,
        #'max_bin':1023,
        'verbose': 0
    }
    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        y_te = y[test_index]
        x_te = x_train[test_index]
        lgtrain = lgb.Dataset(x_tr, y_tr)
                    #feature_name=x_train.columns.tolist())
        lgvalid = lgb.Dataset(x_te, y_te)
                    #feature_name=x_train.columns.tolist())
                    #categorical_feature = categorical)
        lgb_clf = lgb.train(
            lgbm_params,
            lgtrain,
            num_boost_round=20000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=['train','valid'],
            early_stopping_rounds=50,
            verbose_eval=50
        )
        oof_train[test_index] = lgb_clf.predict(x_te)
        oof_test_skf[i, :]    = lgb_clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train, oof_test
train = np.load('vars/flatten_train.npy')
test = np.load('vars/flatten_test.npy')

import pandas as pd

df = pd.read_csv('../input/train.csv')

y  = df['target']

oof_train, oof_test = get_oof(None, train,  y, test)

print("Modeling Stage")
preds = np.concatenate([oof_test])
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = pred
sub.to_csv('sub_et2.csv', index=False)

