This repository keeps all the data and code I worked with to compete in the Kaggle Competion "Titanic - Machine Learning from Disaster". 


The best model has a accuracy of 77.5% which is roughly better than 57% of all models that have been commited to the competition. 
Rough despriction of my approach to the problem:
1. I analysed and explored the given dataset.
2. I filled missing data to get a bigger dataset and normalized the numeric features.
3. I trained different machine learning models (Kmean Clustering, Decision Tree, Random Forest, Logistic Regression, Random Forest from xgboost) and compared their accuracy.
4. Next I selected the models that had the best accuracy and tuned their parameters with hyperparameter tuning.
5. As the last step I choose a logistic regression because it had the best accuracy after hyperparameter tuning and submitted the prediction of this model for the test data.
