#!/usr/bin/env python
# coding: utf-8


## 読み込み

## 学習＆一時保存
### cv or 一括学習を選択

def quadratic_weighted_kappa(y_true, y_pred):
    if isinstance(y_pred, xgb.QuantileDMatrix):
        # XGB
        y_true, y_pred = y_pred, y_true

        y_true = (y_true.get_label() + a).round()
        y_pred = (y_pred + a).clip(1, 6).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return 'QWK', qwk

    else:
        # For lgb
        y_true = y_true + a
        y_pred = (y_pred + a).clip(1, 6).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return 'QWK', qwk, True

def qwk_obj(y_true, y_pred):

    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    f = 1/2*np.sum((preds-labels)**2)
    g = 1/2*np.sum((preds-a)**2+b)
    df = preds - labels
    dg = preds - a
    grad = (df/g - f*dg/g**2)*len(labels)
    hess = np.ones(len(labels))
    return grad, hess

model_params = {
    'lgbm': {
        'objective': qwk_obj,
        'metrics': 'None',
        'learning_rate': 0.05,
        'max_depth': 5,
        'num_leaves': 10,
        'colsample_bytree': 0.3,
        'reg_alpha': 2.,
        'reg_lambda': 0.1,
        'n_estimators': 700,
        'random_state': 42,
        'extra_trees': True,
        'class_weight': 'balanced',
        'device': 'gpu' if CUDA_AVAILABLE else 'cpu',
        'verbosity': - 1
    }

    'xgb': {
        'objective': qwk_obj,  # qwk_objは事前に定義されている関数を指定
        'metrics': 'None',
        'learning_rate': 0.1,
        'max_depth': 5,
        'num_leaves': 10,
        'colsample_bytree': 0.5,
        'reg_alpha': 1.0,
        'reg_lambda': 0.1,
        'n_estimators': 1024,
        'random_state': 42,
        'extra_trees': True,
        'class_weight': 'balanced',
        'tree_method': "hist",
        'device': "gpu" if CUDA_AVAILABLE else "cpu"
    }
}

class Trainer:
    def __init__(self, config, model_params):
        self.config = config
        self.model_params = model_params
        self.light = None
        self.xgb_regressor = None

    def train(self, X_train, y_train, X_valid_fold, y_valid):
        
        light = lgb.LGBMRegressor(**self.model_params['lgbm'])
        self.light = light.fit(X_train_fold,
                                y_train_fold,
                                eval_names=['train', 'valid'],
                                eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
                                eval_metric=quadratic_weighted_kappa,
                                callbacks=callbacks
        )

        xgb_regressor = xgb.XGBRegressor(**self.model_params['xgb'])
        xgb_callbacks = [
            xgb.callback.EvaluationMonitor(period=25),
            xgb.callback.EarlyStopping(75, metric_name="QWK", maximize=True, save_best=True)
        ]
        self.xgb_regressor = xgb_regressor.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=xgb_callbacks)

    def save_weight(self):

        self.light.save_model(f'lgbm_model_fold_{i}.json')
        self.xgb_regressor.save_model(f'xgb_model_fold_{i}.json')

    def load_weight(self):
        
        self.light.load_model(f'lgbm_model_fold_{i}.json')
        self.xgb_regressor.load_model(f'xgb_regressor_model_fold_{i}.json')
    
    def predict(self, X):

        predicted = None
        predicted = 0.76*self.light.predict(X)
        predicted += 0.24*self.xgb_regressor.predict(X)

        return predicted
