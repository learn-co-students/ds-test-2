# Rubric

This rubric will be used to assess students' performance on the Module 4 assessment.

## Calculus, Cost Function, and Gradient Descent
Point Value: 5

#### 1)  

Point Value: 1.25

0.75 points: identifies as cost curve  

0.5 points: states that the general goal of training a machine learning model is to minimize the cost curve

 *Learning Goal(s): Describe how minima and maxima are related to machine learning and optimization. Describe a cost curve and what it means to move along it*

#### 2)

Point Value: 1.25  

0.5 points: identifies the m value of 0.03

0.75 points: identifies that the Residual sum of squares is being updated as the *m* value is changing for the linear model. Indicates that a higher value on RSS means higher error or vice versa.

 *Learning Goal(s): Describe how to use an RSS curve to find the optimal parameters for a linear regression model. Describe a cost curve and what it means to move along it*

#### 3)

Point Value: 1

1 point: mentions that the distance between each step is getting smaller because the gradient of the cost curve is become less steep

 *Learning Goal(s): Define step sizes in the context of gradient descent*

#### 4)

Point Value: 1.5

1 point: defines learning rate as a value ranging from 0.0 - 1.0 that effects the step size in gradient descent to help ensure an optimum minimum is discovered

0.5 points: correctly describes the effect of large and small learning rates on step size

*Learning Goal(s): Define a learning rate, and its relationship to step size when performing gradient descent*  



## Extensions to Linear Models
Point Value: 5

#### 1)

Point Value: 1.5

0.75 points: uses PolynomialFeatures model to fit and transform the X_train and X_test

0.25 points: fits a linear regression model for each degree polynomial

0.25 points: calculates the mean_square error

0.25 points: returns 2 lists of train and test errors

 *Learning Goal(s): Use sklearn's built-in capabilities to create polynomial features*

#### 2)

Point Value: 1

0.5 points: identifies the correct number of degrees for the polynomial

0.5 points: defines bias/variance correctly as it relates to increased complexity of model

 *Learning Goal(s): Describe the bias/variance tradeoff in machine learning. Discuss how bias and variance are related to over and underfitting*  

#### 3)

Point Value: 1

0.25 points: identifies technique for overfitting

0.25 points: explains how the overfitting technique works

0.25 points: identifies technique for underfitting

0.25 points: explains how the underfitting technique works

 *Learning Goal(s): Identify when it is appropriate to use certain methods of feature selection. Discuss how bias and variance are related to overfitting and underfitting*

#### 4)

Point Value: 1.5

0.5 points: performs standardization on the data prior to fitting the regularized regression model

0.75 points: fits model with either Lasso, Ridge or ElasticNet regression

0.25 points: return the root mean square error of the model


 *Learning Goal(s): Describe why standardization is necessary before Ridge and Lasso regression. Use Lasso and Ridge regression with sci-kit learn*

## Introduction to Logistic Regression
Point Value: 5

#### 1)
Point Value: 1

0.34 points : calculates precision  

0.33 points : calculates recall  

0.33 points : calculates F1 score

*Learning Goal(s): Interpret a confusion matrix to assess performance of a model*

#### 2)
Point Value: 1

0.5 points: identifies valid real-life example when precision is more important than recall

0.5 points: correctly uses description of Type II errors or False Negatives in their description

 *Learning Goal(s): Define precision and recall.*

#### 3)

Point Value: 1

0.5 points: identify the correct best ROC curve

0.5 points: explain that it is best because it has the largest AUC

 *Learning Goal(s): Explain how ROC and AUC are used to evaluate and choose models*

#### 4)

Point Value: 0.5

0.5 points: identifies the problem as class imbalance

 *Learning Goal(s): Describe why class imbalance can lead to problems in machine learning*

#### 5)

Point Value: 1.5

0.75 points: uses SMOTE, undersampling, or oversampling to address

0.25 points: correctly transforms just training data (not test data)

0.25 points: fits a logistic regression model to the resampled data

0.25 points: makes predictions with test data and evaluates accuracy and AUC

 *Learning Goal(s): Use sampling techniques to address class imbalance problem within a dataset*

## Working with Time Series Data
Point Value: 5

#### 1)

Point Value: 1

0.5 points: transform the date feature to datetime

0.5 points: set the datetime to the index

 *Learning Goal(s): Load time series data into a pandas dataframe and perform time series indexing*

#### 2)

Point Value: 1

0.5 points: resample on the monthly basis

0.5 points: take the mean of the resampled data

 *Learning Goal(s): Change the granularity of a time series*

#### 3)

Point Value: 1

0.5 points: obtain the rolling mean and rolling standard deviation

0.5 points: correctly determines whether or not the stock time series is stationary


 *Learning Goal(s): Create visualizations of transformed time series as a visual aid to determine if stationarity has been achieved. Use rolling statistics an a check for stationarity*

#### 4)

Point Value: 1

0.5 points: perform the Dickey-Fuller test using the statsmodels method

0.5 points: interpret the p-value of the Dickey-Fuller test to indicate that the data is not stationary

 *Learning Goal(s): Use the Dickey Fuller Test and conclude whether or not a dataset is exhibiting stationarity*

#### 5)

Point Value: 1

0.5 points: uses time series indexing to select the maximum value for a given year

0.5 points: returns a dictionary with the correct year and max value data

 *Learning Goal(s): Perform time series indexing*
