# Kaggle Titanic

Play with titanic survive data.

Compared many different model using same features and data.

* Predict using only 'gender': 0.76555
* Predict using decision tree classifier: 0.78468 **(best)**
* Predict using svc: 0.77033 **(overfitted)**
* predict using random forest classifier: 0.76555 **(overfitted)**
* predict using logistic regression: 0.72249 **(overfitted)**
* predict using ensemble above 5 models: 0.76076
* predict using feed foward network that have only one hidden layer with 16 hidden units: 0.74641



In traditional models, I think there should be feature engineering. 

And almost every models are very easily overfitted which means training accuracy was much more higher than validation accuracy.

It was really hard to train feed foward network(very easily stuck in local minima). I guess learning rate became much too big after 100~200 epochs.