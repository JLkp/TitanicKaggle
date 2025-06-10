import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

SEED = 42
np.random.seed(SEED)

pd.set_option('display.max_columns', 20)

plt.figure(dpi=1200)
sns.set_theme("notebook", rc={"figure.dpi": 1200, 'figure.figsize': (10, 8)})


df_train = pd.read_csv("train_normalized.csv", sep=',')
df_test = pd.read_csv("test_normalized.csv", sep=',')

data_to_predict = df_test.drop("PassengerId", axis=1)


x_normalized = df_train.iloc[:,2:]
y_normalized = df_train.iloc[:,1]

x_train, x_test, y_train, y_test = train_test_split(x_normalized, y_normalized, 
                                                    test_size=0.2,
                                                    random_state=SEED)

# Model mit GridSearchCV aufbauen
param_grid = {'n_estimators': np.arange(100,800,100),
              'max_depth': [3,5,7,9,11],
              'criterion': ['gini', 'entropy', 'log_loss'],
              'min_samples_split': [2,5,10],
              'max_features': ['sqrt', 'log2']
              }
    
model = GridSearchCV(estimator=RandomForestClassifier(), 
                     param_grid=param_grid,
                     refit=True, 
                     cv=4, 
                     n_jobs=4).fit(x_normalized, y_normalized)
    

print(model.best_params_)

best_params = model.best_params_

model = RandomForestClassifier().set_params(**best_params).fit(x_normalized, 
                                                               y_normalized)
    
y_pred = model.predict(x_test)
model_report = classification_report(y_test, y_pred=y_pred)
print(model_report)

def submit(model_):
    test_result = model_.predict(data_to_predict)
    result = pd.DataFrame()
    result['PassengerId'] = df_test.PassengerId
    result['Survived'] = test_result.astype(dtype=int)
    result.to_csv('SUBMIT_TEST.csv', sep=',', index=False)

# submit(model)   
