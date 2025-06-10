import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, roc_curve, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split


SEED = 42
np.random.seed(SEED)


pd.set_option('display.max_columns', 20)

plt.figure(dpi=1200)
plt.style.use('seaborn-v0_8-notebook')
sns.set_theme("notebook", rc={"figure.dpi": 1200, 'figure.figsize': (10, 8)})





# Daten importieren
df_train = pd.read_csv('min_max_train.csv', sep=',')


# Daten in Trainigs- und Testdaten aufteilen
x = df_train.iloc[:,1:]
y = df_train.iloc[:,0]

main_set = np.random.choice(x.shape[0], int(x.shape[0]*0.8), replace=False)
test_set = np.delete(np.arange(x.shape[0]), main_set)

# Trainingsdaten
x_train = x.iloc[main_set,:]
y_train = y.iloc[main_set]

# Testdaten
x_test = x.iloc[test_set,:]
y_test = y.iloc[test_set]


# Modele auf Trainigsdaten und Testdaten testen
df_models_acc = pd.DataFrame(columns=['KMEANS',
                                      'TREE',
                                      'FOREST',
                                      'LOGISTIC',
                                      'XGB'], 
                             index=['ACC'])


# KMEANS
kmeans = KMeans(n_clusters=2, 
                n_init=200, 
                max_iter=2000, 
                random_state=SEED).fit(x_train, y_train)

kmeans_pred = kmeans.predict(x_test)
kmeans_acc = accuracy_score(y_test, kmeans_pred)


df_models_acc['KMEANS'] = kmeans_acc

# Decision Tree
dtree = DecisionTreeClassifier(max_depth=200, 
                               min_samples_split=5, 
                               random_state=SEED).fit(x_train, y_train)

dtree_pred = dtree.predict(x_test)
dtree_acc = accuracy_score(y_test, dtree_pred)

df_models_acc['TREE'] = dtree_acc


# RandomForest
rf = RandomForestClassifier(n_estimators=150,
                            max_depth=200,
                            min_samples_split=5,
                            random_state=SEED).fit(x_train, y_train)
rf_pred = rf.predict(x_test)
rf_acc = accuracy_score(y_test, rf_pred)

df_models_acc['FOREST'] = rf_acc


# Logistic Regression
lr = LogisticRegression(solver='newton-cholesky').fit(x_train, y_train)
lr_pred = lr.predict(x_test)
lr_acc = accuracy_score(y_test, lr_pred)

df_models_acc['LOGISTIC'] = lr_acc


# XGBoost
rf_boost = XGBClassifier(learning_rate=0.1,
                         n_estimators=150,
                         max_depth=3,       
                         objective='binary:logistic',  
                         random_state=SEED).fit(x_train, y_train)
rf_boost_pred = rf_boost.predict(x_test)
rf_boost_acc = accuracy_score(y_test, rf_boost_pred)

df_models_acc['XGB'] = rf_boost_acc


print(df_models_acc)

# =============================================================================
# Gute Modelle (TREE, FOREST, LOGISTIC, XGB) auf Testdaten anwenden 
# und rohe Performance ohne Hyperparameter Tuning bestimmen
# =============================================================================
df_test = pd.read_csv('min_max_test.csv', sep=',')

def test_raw_models():

    def submit_to_csv(file_name, survived, ids=df_test.PassengerId):
        df_submit = pd.DataFrame()
        df_submit['PassengerId'] = ids
        df_submit['Survived'] = survived.astype(dtype=int)
    
        file_name = file_name + '.csv'
        df_submit.to_csv(file_name, sep=',', index=False)
    
    
    test_data = df_test.drop('PassengerId', axis=1)
    
    x_true_train = df_train.iloc[:,1:]
    y_true_train = df_train.iloc[:,0]
    
    # TREE auf Testdaten anwenden
    dtree = DecisionTreeClassifier(max_depth=200, 
                                   min_samples_split=5, 
                                   random_state=SEED).fit(x_true_train, y_true_train)
    
    dtree_pred_test = dtree.predict(test_data)
    
    submit_to_csv('dtree_submit', dtree_pred_test)
    
    
    # FOREST auf Testdaten anwenden
    rf = RandomForestClassifier(n_estimators=150,
                                max_depth=200,
                                min_samples_split=5,
                                random_state=SEED).fit(x_true_train, y_true_train)
    
    rf_pred_test = rf.predict(test_data)
    
    submit_to_csv('forest_submit', rf_pred_test)
    
    # LOGISTIC auf Testdaten anwenden
    lr = LogisticRegression(solver='newton-cholesky').fit(x_true_train, y_true_train)
    lr_pred_test = lr.predict(test_data)
    
    submit_to_csv('lr_submit', lr_pred_test)
    
    
    # XGB auf Testdaten anwenden
    rf_boost = XGBClassifier(learning_rate=0.1,
                             n_estimators=150,
                             max_depth=3,       
                             objective='binary:logistic',  
                             random_state=SEED).fit(x_true_train, y_true_train)
    rf_boost_pred_test = rf_boost.predict(test_data)
    
    submit_to_csv('xgb_submit', rf_boost_pred_test)



# Gute Modelle (FOREST, LOGISTIC, XGB) mit Hyperparameter Tuning optimieren

# Funktion zum Submitten / Test- und Trainingsdaten
def submit_to_csv(file_name, survived, ids=df_test.PassengerId):
    df_submit = pd.DataFrame()
    df_submit['PassengerId'] = ids
    df_submit['Survived'] = survived.astype(dtype=int)

    file_name = file_name + '.csv'
    df_submit.to_csv(file_name, sep=',', index=False)

test_data = df_test.drop('PassengerId', axis=1)

x_true_train = df_train.iloc[:,1:]
y_true_train = df_train.iloc[:,0]


# FOREST optimieren
rf_opti_rs = RandomForestClassifier()

param_dist_rf = {'n_estimators': [50, 100, 125, 150, 175, 200, 225],
                 'criterion': ['gini', 'entropy', 'log_loss'],
                 'max_depth': [None, 10, 20, 30, 40, 50],
                 'min_samples_split': [2, 5, 10, 15, 17, 20],
                 'min_samples_leaf': [2, 5, 10, 15, 17, 20],
                 'max_features': [None, 'sqrt', 'log2'],
                 'max_leaf_nodes':[50, 100, 150, 200, 250, 300, 350],
                 'bootstrap': [True, False]} 


rf_rs = RandomizedSearchCV(estimator= rf_opti_rs, 
                           param_distributions= param_dist_rf,
                           n_iter=250,
                           n_jobs=-1,
                           cv=3).fit(x_train, y_train)



# RandomForest mit Parameter aus RandomizedSearch bauen
rf_rs_final = RandomForestClassifier().set_params(**rf_rs.best_params_)

# RandomForest trainieren und Testdaten submiten
rf_rs_final.fit(x_true_train, y_true_train)
rf_rs_final_pred = rf_rs_final.predict(test_data)

submit_to_csv('rf_rs_submit', rf_rs_final_pred)

# Parameter aus RandomizedSearch mit GridSearch weiter optimieren

param_dist_rf_grid = {'n_estimators': [50, 55, 60, 65, 70, 75, 80],
                      'criterion': ['log_loss'],
                      'max_depth': [5, 10, 15],
                      'min_samples_split': [10, 15, 17],
                      'min_samples_leaf': [10, 15, 17],
                      'max_features': ['log2'],
                      'max_leaf_nodes': [225, 250, 275],
                      'bootstrap': [True]}

rf_opti_gs = RandomForestClassifier()

rf_gs = GridSearchCV(estimator= rf_opti_gs, 
                     param_grid= param_dist_rf_grid,
                     n_jobs= -1).fit(x_train, y_train)

rf_gs_final = RandomForestClassifier().set_params(**rf_gs.best_params_)

# RandomForest trainieren und Testdaten submiten
rf_gs_final.fit(x_true_train, y_true_train)
rf_gs_final_pred = rf_gs_final.predict(test_data)

submit_to_csv('rf_gs_submit', rf_gs_final_pred)

# logistic regression optimieren
lr_opti_rs = LogisticRegression()

param_dist_lr_rs = {'penalty': ['l2'],
                    'tol': np.arange(0.001, 0.1, 0.001),
                    'C': np.arange(0.001, 0.1, 0.001),
                    'fit_intercept': [True, False],
                    'class_weight': [None, 'balanced'],
                    'solver': ['newton-cg', 'newton-cholesky', 'sag', 'saga'],
                    'max_iter': np.arange(1000, 3000, 10)}

lr_rs = RandomizedSearchCV(estimator= lr_opti_rs, 
                           param_distributions= param_dist_lr_rs,
                           n_iter= 1000,
                           n_jobs= -1, 
                           cv= 3).fit(x_train, y_train)

print(lr_rs.best_params_, lr_rs.best_score_)

# LogisticRegression mit Parametern aus rs bauen
lr_rs_final = LogisticRegression().set_params(**lr_rs.best_params_)

# LogisticRegression trainieren und Testdaten submiten
lr_rs_final.fit(x_true_train, y_true_train)
lr_rs_final_pred = lr_rs_final.predict(test_data)

submit_to_csv('lr_rs_submit', lr_rs_final_pred)


# Parameter aus RandomizedSearch mit GridSearch weiter optimieren
lr_opti_gs = LogisticRegression()

param_dist_lr_grid = {'penalty': ['l2'],
                    'tol': np.arange(0.05, 0.061, 0.00012),
                    'C': np.arange(0.068, 0.071, 0.00012),
                    'fit_intercept': [True],
                    'class_weight': [None],
                    'solver': ['sag'],
                    'max_iter': np.arange(1700, 1800, 10)}


lr_gs = GridSearchCV(estimator= lr_opti_gs, 
                     param_grid= param_dist_lr_grid,
                     n_jobs= -1,
                     cv= 3).fit(x_train, y_train)


lr_gs_final = LogisticRegression().set_params(**lr_gs.best_params_)

# Finales Model  trainieren und submitten
lr_gs_final.fit(x_true_train, y_true_train)
lr_gs_final_pred = lr_gs_final.predict(test_data)

submit_to_csv('lr_gs_submit', lr_gs_final_pred)
