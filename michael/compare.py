import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
from subprocess import check_output
print("Train")
params = {
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'learning_rate': 0.03,
        'metric': 'binary_logloss',
        'metric': 'auc',
        'min_data_in_bin': 3,
        'max_depth': 10,
        'objective': 'binary',
        'verbose': -1,
        'num_leaves': 108,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 223,
        'num_rounds': 1000,
        'verbose_eval':10,
         }
from sklearn.cross_validation import KFold
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
        'metric': ['auc', 'binary_logloss'],
        # 'max_depth': 15,
        'num_leaves': 31,
        'min_deal_in_leaf':1500,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'lambda_l1':1.0,
        'lambda_l2':1.0,
        'bagging_freq': 4,
        'learning_rate': 0.01,
        'max_bin':255,
        'verbose': 0
    }
    losses = []
    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train.iloc[train_index]
        y_tr = y[train_index]
        y_te = y[test_index]
        x_te = x_train.iloc[test_index]
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
            early_stopping_rounds=700,
            verbose_eval=50
        )
        oof_train[test_index] = lgb_clf.predict(x_te)
        oof_test_skf[i, :]    = lgb_clf.predict(x_test)
        del (lgvalid, lgtrain)
        gc.collect()

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train, oof_test

tdf = pd.read_csv('../input/train.csv')
Tdf = pd.read_csv('../input/test.csv')
y = tdf['target'].values
tdf = tdf.drop(['id', 'target'], axis=1)
Tdf = Tdf.drop(['id'], axis=1)
oof_train, oof_test = get_oof(None, tdf,  y, Tdf)

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
train_auc = roc_auc_score(y, oof_train)
train_logloss = log_loss(y, oof_train)
print('train auc', train_auc)
print('train logloss', train_logloss)

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds
sub.to_csv('sub_lgb.csv', index=False)
