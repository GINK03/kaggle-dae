# 

## gini coefficient

https://www.kaggle.com/batzner/gini-coefficient-an-intuitive-explanation

TODO：あとで解説
GINIの導出とAUC最大化について

## pythonのscikit-learn互換IFでのアンサンブル学習をクラスにまとめてうまくやる例

```python
class Ensemble(object):
  def __init__(self, n_splits, stacker, base_models):
    self.n_splits = n_splits
    self.stacker = stacker
    self.base_models = base_models

  def fit_predict(self, X, y, T):
    X = np.array(X)
    y = np.array(y)
    T = np.array(T)

    folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

    S_train = np.zeros((X.shape[0], len(self.base_models)))
    S_test = np.zeros((T.shape[0], len(self.base_models)))
    for i, clf in enumerate(self.base_models):
      S_test_i = np.zeros((T.shape[0], self.n_splits))
      for j, (train_idx, test_idx) in enumerate(folds):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_holdout = X[test_idx]

        print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_holdout)[:,1]
        S_train[test_idx, i] = y_pred
        S_test_i[:, j] = clf.predict_proba(T)[:,1]
      S_test[:, i] = S_test_i.mean(axis=1)

    results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
    print("Stacker score: %.5f" % (results.mean()))

    self.stacker.fit(S_train, y)
    res = self.stacker.predict_proba(S_test)[:,1]
    return res
```

