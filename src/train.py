import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.pipeline_utils import get_features, get_preprocessor, load_and_split_data

le = LabelEncoder()
X, Y = load_and_split_data('telco_churn.csv')
Y = le.fit_transform(Y)

num_cols, ord_cols, nom_cols = get_features()
preprocessor = get_preprocessor(num_cols, ord_cols, nom_cols)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
model_scores = {}

model_ada = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(n_estimators=100))
])
model_ada.fit(X_train, Y_train)
param_grid_ada = {'classifier__n_estimators': [100, 150, 200]}
grid_ada = GridSearchCV(model_ada, param_grid_ada, cv=5, scoring='f1')
grid_ada.fit(X_train, Y_train)
Y_predict = grid_ada.predict(X_test)
model_scores['AdaBoost'] = {'F1-Score': f1_score(Y_test, Y_predict), 'Recall': recall_score(Y_test, Y_predict)}

scale_pos_weight = len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1])
model_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight))
])
model_xgb.fit(X_train, Y_train)
param_grid_xgb = {'classifier__n_estimators': [50, 100]}
grid_xgb = GridSearchCV(model_xgb, param_grid_xgb, cv=5, scoring='f1')
grid_xgb.fit(X_train, Y_train)
Y_predict = grid_xgb.predict(X_test)
model_scores['XGBoost'] = {'F1-Score': f1_score(Y_test, Y_predict), 'Recall': recall_score(Y_test, Y_predict)}

model_cb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', CatBoostClassifier(n_estimators=100, auto_class_weights='Balanced', verbose=False))
])
model_cb.fit(X_train, Y_train)
param_grid_cb = {'classifier__n_estimators': [200, 250, 300]}
grid_cb = GridSearchCV(model_cb, param_grid_cb, cv=5, scoring='f1')
grid_cb.fit(X_train, Y_train)
Y_predict = grid_cb.predict(X_test)
model_scores['CatBoost'] = {'F1-Score': f1_score(Y_test, Y_predict), 'Recall': recall_score(Y_test, Y_predict)}

model_knn = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])
model_knn.fit(X_train, Y_train)
param_grid_knn = {'classifier__n_neighbors': [7, 9, 11]}
grid_knn = GridSearchCV(model_knn, param_grid_knn, cv=5, scoring='f1')
grid_knn.fit(X_train, Y_train)
Y_predict = grid_knn.predict(X_test)
model_scores['KNeighbors'] = {'F1-Score': f1_score(Y_test, Y_predict), 'Recall': recall_score(Y_test, Y_predict)}

model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(max_depth=5, class_weight='balanced', random_state=42))
])
model_rf.fit(X_train, Y_train)
param_grid_rf = {'classifier__n_estimators': [150, 200, 250], 'classifier__max_depth': [5, 6, 7]}
grid_rf = GridSearchCV(model_rf, param_grid_rf, cv=5, scoring='f1')
grid_rf.fit(X_train, Y_train)
Y_predict = grid_rf.predict(X_test)
model_scores['Random Forest'] = {'F1-Score': f1_score(Y_test, Y_predict), 'Recall': recall_score(Y_test, Y_predict)}

scores_df = pd.DataFrame.from_dict(model_scores, orient='index').reset_index()
scores_df.rename(columns={'index': 'Model'}, inplace=True)
scores_df = scores_df.sort_values(by='F1-Score', ascending=False)
scores_df.to_csv('model_scores.csv', index=False)

best_model = grid_rf.best_estimator_
with open('telco_churn_pipeline.pkl', 'wb') as f:
    pickle.dump(best_model, f)