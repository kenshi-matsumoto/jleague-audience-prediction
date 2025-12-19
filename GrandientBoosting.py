import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd

import seaborn as sns
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error
)

import lightgbm as lgb
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

def heat_map(cm):
    cm_matrix = pd.DataFrame(data=cm,columns=['Actual Positive:1','Actual Negative:0'],
                            index=['Predict Positive:1','Predict Negative:0'])

    # ヒートマップで描画
    plt.figure(figsize=(3, 3))
    heatmap = sns.heatmap(cm_matrix, annot=True, fmt='d',cmap='coolwarm')
    # heatmap.set_title('混同行列',fontsize=20)
    heatmap.set_xticks([0.5,1.5])
    heatmap.set_xlabel('予測',fontsize=15)
    heatmap.set_xticklabels(['Negative:0','Positive:1'],fontsize=12)
    heatmap.set_ylabel('真値',fontsize=15)
    heatmap.set_yticks([0.5,1.5])
    heatmap.set_yticklabels(['Negative:0','Positive:1'],fontsize=12)

train_data = pd.read_csv("train_preprocessed.csv")
test_data = pd.read_csv("test_preprocessed.csv")

b_train = train_data['attendance']
A_train = train_data.drop('attendance', axis=1)
A_train = A_train.drop('id', axis=1)
X_test = test_data.drop('id', axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(A_train, b_train, test_size=0.3, random_state=0)

categorical_features = ['weather_encoded', 'holiday', 'home_team_encoded', 'away_team_encoded', 'address_encoded']

def objective(trial):
    params = {
        'objective': 'regression',
        'lambda_l1'         : trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2'         : trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'learning_rate': 0.01,
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_bin' : trial.suggest_int('max_bin', 64, 500),
    }

    lgb_train = lgb.Dataset(X_train, y_train,
                           categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train,
                           categorical_feature=categorical_features)

    model = lgb.train(params, lgb_train,
                     valid_sets=[lgb_train, lgb_eval],
                     num_boost_round=10000,
                     callbacks=[lgb.early_stopping(stopping_rounds=10,
                                                verbose=True),
                             lgb.log_evaluation(10)])

    y_pred_valid = model.predict(X_valid,
                                 num_intertion=model.best_iteration)

    score = mean_squared_error(y_valid, y_pred_valid)
    return score

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=40)

pre_params = {
    'objective': 'regression',
    'learning_rate': 0.01
}

params = {**pre_params, **study.best_params}

lgb_train = lgb.Dataset(X_train, y_train,
                        categorical_feature=categorical_features)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train,
                       categorical_feature=categorical_features)

model = lgb.train(params, lgb_train,
                  valid_sets=[lgb_train, lgb_eval],
                  num_boost_round=10000,
                  callbacks=[lgb.early_stopping(stopping_rounds=10,
                                                verbose=True),
                             lgb.log_evaluation(10)])

y_pred = model.predict(X_test, num_iteration=model.best_iteration)

y_ans = pd.Series(y_pred)

result = test_data['id']
result = pd.concat([result, y_ans], axis=1, join='inner')

result_file = "predict.csv"
result.to_csv(result_file, index=False, header=False)
