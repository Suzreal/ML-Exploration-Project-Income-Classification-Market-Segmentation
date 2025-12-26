import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

df = pd.read_csv('cleaned_data.csv')

y = df['label']
w = df['weight']

X = df.drop(columns=['label','weight'])

# build the test set
test_mask = X['year'] == 95

X_test = X.loc[test_mask].drop(columns=['year'])
y_test = y.loc[test_mask]
w_test = w.loc[test_mask]

# build the train test
trainval_mask = X['year'] == 94

X_94 = X.loc[trainval_mask].drop(columns=["year"])
y_94 = y.loc[trainval_mask]
w_94 = w.loc[trainval_mask]

X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X_94, y_94, w_94,test_size=0.2,
                                                                  random_state=42,stratify=y_94)



numerical_col = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_col = [col for col in X_train.columns if col not in numerical_col]

numerical_tran = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))])

cate_tran = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',  sparse_output=True))
])

preprocess = ColumnTransformer(transformers=[
    ('num',numerical_tran,numerical_col),
    ('cat',cate_tran,categorical_col),],
    remainder='drop')


candidates = [
    {'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'max_depth': 6, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'max_depth': 6, 'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'max_depth': 8, 'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.8},
]

best = None

for params in candidates:
    xgb = XGBClassifier(n_estimators=800,learning_rate=0.05,reg_lambda=1.0,objective='binary:logistic',
        eval_metric='logloss',tree_method='hist',random_state=42,n_jobs=-1,**params)

    model = Pipeline(steps=[
        ('preprocess', preprocess),
        ('xgb', xgb)
    ])

    model.fit(X_train, y_train, xgb__sample_weight=w_train)
    p = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        f1s.append(f1_score(y_val, pred, sample_weight=w_val, zero_division=0))

    idx = int(np.argmax(f1s))
    t_best = float(thresholds[idx])
    f1_best = float(f1s[idx])

    if (best is None) or (f1_best > best['f1']):
        best = {'model': model, 'params': params, 'threshold': t_best, 'f1': f1_best}

best_xgb_model = best['model']
best_xgb_threshold = best['threshold']

# refit best model on full 1994 pool
best_xgb_model.fit(X_94, y_94, xgb__sample_weight=w_94)

# run the same test evaluation again
proba_test = best_xgb_model.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= best_xgb_threshold).astype(int)

print('\n=== FINAL TEST (1995) METRICS after refit on all 1994 ===')
print('threshold:', best_xgb_threshold)
print('accuracy :', accuracy_score(y_test, pred_test, sample_weight=w_test))
print('precision:', precision_score(y_test, pred_test, sample_weight=w_test, zero_division=0))
print('recall   :', recall_score(y_test, pred_test, sample_weight=w_test, zero_division=0))
print('f1       :', f1_score(y_test, pred_test, sample_weight=w_test, zero_division=0))
print('roc_auc  :', roc_auc_score(y_test, proba_test, sample_weight=w_test))
print('\nWeighted confusion matrix [[TN, FP],[FN, TP]]:')
print(confusion_matrix(y_test, pred_test, sample_weight=w_test))
