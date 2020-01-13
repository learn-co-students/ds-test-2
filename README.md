
<h1 align="center">Module 3 Code Challenge</h1>

## Overview

This assessment is designed to test your understanding of the Mod 3 materials. It covers:

* Calculus, Cost Function, and Gradient Descent
* Introduction to Logistic Regression
* Decision Trees
* Ensemble Models 


Read the instructions carefully. You will be asked both to write code and respond to a few short answer questions.

### Note on the short answer questions

For the short answer questions please use your own words. The expectation is that you have not copied and pasted from an external source, even if you consult another source to help craft your response. While the short answer questions are not necessarily being assessed on grammatical correctness or sentence structure, do your best to communicate yourself clearly.

---
## Calculus, Cost Function, and Gradient Descent [Suggested Time: 25 min]
---

![best fit line](visuals/best_fit_line.png)

The best fit line that goes through the scatterplot up above can be generalized in the following equation: $$y = mx + b$$

Of all the possible lines, we can prove why that particular line was chosen using the plot down below:

![](visuals/cost_curve.png)

where RSS is defined as the residual sum of squares:

$$ 
\begin{align}
RSS &= \sum_{i=1}^n(actual - expected)^2 \\
&= \sum_{i=1}^n(y_i - \hat{y})^2 \\
&= \sum_{i=1}^n(y_i - (mx_i + b))^2
\end{align}
$$ 

### 1. What is a more generalized name for the RSS curve above? How is it related to machine learning models?


```python
# Your answer here
```


```python
# __SOLUTION__
# The residual sum of squares curve above is a specific example of a cost curve. 
# When training machine learning models, the goal is to minimize the cost curve.
```

### 2. Would you rather choose a $m$ value of 0.08 or 0.05 from the RSS curve up above?   What is the relation between the position on the cost curve, the error, and the slope of the line?


```python
# Your answer here
```


```python
# __SOLUTION__
# It would be better to have a value of 0.05 rather than 0.08 in the cost curve above. 
# The reason for this is that the RSS is lower for the value of 0.05. 
# As m changes values from 0.00 to 0.10 the Residual Sum of Squares is changing.
# The higher the value of the RSS, the worse the model is performing.
```

![](visuals/gd.png)

### 3. Using the gradient descent visual from above, explain why the distance between each step is getting smaller as more steps occur with gradient descent.


```python
# Your answer here
```


```python
# __SOLUTION__
# The distance between the steps is getting smaller because the slope gradually 
# becomes less and less steep as you get closer to finding the minimum.
```

### 4. What is the purpose of a learning rate in gradient descent? Explain how a very small and a very large learning rate would affect the gradient descent.


```python
# Your answer here
```


```python
# __SOLUTION__
# Learning rate is a number that is multiplied by each step that 
# is taken during gradient descent. If the learning rate is smaller, the step sizes will become smaller. 
# If the learning rate is larger, the step sizes will be larger. 
# Learning rate is present in gradient descent to help ensure that an optimal minimum on the cost curve is discovered.
```


```python
# Your answer here
```


```python
# __SOLUTION__ 
# Regularization adjusts feature weights depending on their magnitude.
# Feature weights themselves depend on both the feature importance and the magnitude of the input variable.
# Therefore, it's important to control for the magnitude of the input variable by scaling all features the same.
```

---
## Introduction to Logistic Regression [Suggested Time: 25 min]
---

<!---
# load data
ads_df = pd.read_csv("raw_data/social_network_ads.csv")

# one hot encode categorical feature
def is_female(x):
    """Returns 1 if Female; else 0"""
    if x == "Female":
        return 1
    else:
        return 0
        
ads_df["Female"] = ads_df["Gender"].apply(is_female)
ads_df.drop(["User ID", "Gender"], axis=1, inplace=True)
ads_df.head()

# separate features and target
X = ads_df.drop("Purchased", axis=1)
y = ads_df["Purchased"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=19)

# preprocessing
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

# save preprocessed train/test split objects
pickle.dump(X_train, open("write_data/social_network_ads/X_train_scaled.pkl", "wb"))
pickle.dump(X_test, open("write_data/social_network_ads/X_test_scaled.pkl", "wb"))
pickle.dump(y_train, open("write_data/social_network_ads/y_train.pkl", "wb"))
pickle.dump(y_test, open("write_data/social_network_ads/y_test.pkl", "wb"))

# build model
model = LogisticRegression(C=1e5, solver="lbfgs")
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

from sklearn.metrics import confusion_matrix

# create confusion matrix
# tn, fp, fn, tp
cnf_matrix = confusion_matrix(y_test, y_test_pred)
cnf_matrix

# build confusion matrix plot
plt.imshow(cnf_matrix,  cmap=plt.cm.Blues) #Create the basic matrix.

# Add title and Axis Labels
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Add appropriate Axis Scales
class_names = set(y_test) #Get class labels to add to matrix
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Add Labels to Each Cell
thresh = cnf_matrix.max() / 2. #Used for text coloring below
#Here we iterate through the confusion matrix and append labels to our visualization.
for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

# Add a Side Bar Legend Showing Colors
plt.colorbar()

# Add padding
plt.tight_layout()
plt.savefig("visuals/cnf_matrix.png",
            dpi=150,
            bbox_inches="tight")
--->

![cnf matrix](visuals/cnf_matrix.png)

### 1. Using the confusion matrix up above, calculate precision, recall, and F-1 score.


```python
# // your code here //
```


```python
# __SOLUTION__ 
precision = 30/(30+4)
recall = 30 / (30 + 12)
F1 = 2 * (precision * recall) / (precision + recall)

print("precision: {}".format(precision))
print("recall: {}".format(recall))
print("F1: {}".format(F1))
```

### 2.  What is a real life example of when you would care more about recall than precision? Make sure to include information about errors in your explanation.


```python
# Your answer here
```


```python
# __SOLUTION__
# We would care more about recall than precision in cases where a Type II error (a False Negative) would 
# have serious consequences. An example of this would be a medical test that determines if someone has a serious disease.
# A higher recall would mean that we would have a higher chance of identifying all people who ACTUALLY had the serious disease.
```

<!---
# save preprocessed train/test split objects
X_train = pickle.load(open("write_data/social_network_ads/X_train_scaled.pkl", "rb"))
X_test = pickle.load(open("write_data/social_network_ads/X_test_scaled.pkl", "rb"))
y_train = pickle.load(open("write_data/social_network_ads/y_train.pkl", "rb"))
y_test = pickle.load(open("write_data/social_network_ads/y_test.pkl", "rb"))

# build model
model = LogisticRegression(C=1e5, solver="lbfgs")
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

labels = ["Age", "Estimated Salary", "Female", "All Features"]
colors = sns.color_palette("Set2")
plt.figure(figsize=(10, 8))
# add one ROC curve per feature
for feature in range(3):
    # female feature is one hot encoded so it produces an ROC point rather than a curve
    # for this reason, female will not be included in the plot at all since it is
    # disingeneuous to call it a curve.
    if feature == 2:
        pass
    else:
        X_train_feat = X_train[:, feature].reshape(-1, 1)
        X_test_feat = X_test[:, feature].reshape(-1, 1)
        logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='lbfgs')
        model_log = logreg.fit(X_train_feat, y_train)
        y_score = model_log.decision_function(X_test_feat)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        lw = 2
        plt.plot(fpr, tpr, color=colors[feature],
                 lw=lw, label=labels[feature])

# add one ROC curve with all the features
model_log = logreg.fit(X_train, y_train)
y_score = model_log.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
lw = 2
plt.plot(fpr, tpr, color=colors[3], lw=lw, label=labels[3])

# create foundation of the plot
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.yticks([i / 20.0 for i in range(21)])
plt.xticks([i / 20.0 for i in range(21)])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("visuals/many_roc.png",
            dpi=150,
            bbox_inches="tight")
--->

### 3. Pick the best ROC curve from this graph and explain your choice. 

*Note: each ROC curve represents one model, each labeled with the feature(s) inside each model*.

<img src = "visuals/many_roc.png" width = "700">



```python
# Your answer here
```


```python
# __SOLUTION__
# The best ROC curve in this graph is for the one that contains all features (the pink one). 
# This is because it has the largest area under the curve. The ROC curve is created by obtaining
# the ratio of the True Positive Rate to the False Positive Rate over all thresholds of a classification model.
```

<!---
# sorting by 'Purchased' and then dropping the last 130 records
dropped_df = ads_df.sort_values(by="Purchased")[:-130]
dropped_df.reset_index(inplace=True)
pickle.dump(dropped_df, open("write_data/sample_network_data.pkl", "wb"))
--->


```python
network_df = pickle.load(open("write_data/sample_network_data.pkl", "rb"))

# partion features and target 
X = network_df.drop("Purchased", axis=1)
y = network_df["Purchased"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2019)

# scale features
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

# build classifier
model = LogisticRegression(C=1e5, solver="lbfgs")
model.fit(X_train,y_train)
y_test_pred = model.predict(X_test)

# get the accuracy score
print(f"The original classifier has an accuracy score of {round(accuracy_score(y_test, y_test_pred), 3)}.")

# get the area under the curve from an ROC curve
y_score = model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
auc = round(roc_auc_score(y_test, y_score), 3)
print(f"The original classifier has an area under the ROC curve of {auc}.")
```


```python
# __SOLUTION__ 
network_df = pickle.load(open("write_data/sample_network_data.pkl", "rb"))

# partion features and target 
X = network_df.drop("Purchased", axis=1)
y = network_df["Purchased"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2019)

# scale features
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

# build classifier
model = LogisticRegression(C=1e5, solver="lbfgs")
model.fit(X_train,y_train)
y_test_pred = model.predict(X_test)

# get the accuracy score
print(f"The original classifier has an accuracy score of {round(accuracy_score(y_test, y_test_pred), 3)}.")

# get the area under the curve from an ROC curve
y_score = model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
auc = round(roc_auc_score(y_test, y_score), 3)
print(f"The original classifier has an area under the ROC curve of {auc}.")
```

### 4. The model above has an accuracy score that might be too good to believe. Using `y.value_counts()`, explain how `y` is affecting the accuracy score.


```python
y.value_counts()
```


```python
# __SOLUTION__ 
y.value_counts()
```


```python
# // your answer here //
```


```python
# __SOLUTION__

# y.value_counts() indicates that we have a class imbalance. When we have class imbalance our model only learns to
# predict one class and is not penalized for doing so, because it is still getting the right answer most of the time. 
```

### 5. What methods would you use to address the issues mentioned up above in question 4? 



```python
# // your answer here //

```


```python
# __SOLUTION__

#Any one of these is an acceptable answer : 

# Class imbalance could be rectified using SMOTE to generate additional synthetic data points for the minority class so
# that we have equal (or almost equal) number of data points in each class. 

# Class imbalance could be rectified using oversampling to sample (with replacement) from the minority class until we
# have equal samples from both classes. 
```

## Decision Trees [Suggested Time: 15 min]

### Concepts 
You're given a dataset of **30** elements, 15 of which belong to a positive class (denoted by *`+`* ) and 15 of which do not (denoted by `-`). These elements are described by two attributes, A and B, that can each have either one of two values, true or false. 

The diagrams below show the result of splitting the dataset by attribute: the diagram on the left hand side shows that if we split by Attribute A there are 13 items of the positive class and 2 of the negative class in one branch and 2 of the positive and 13 of the negative in the other branch. The right hand side shows that if we split the data by Attribute B there are 8 items of the positive class and 7 of the negative class in one branch and 7 of the positive and 8 of the negative in the other branch.

<img src="visuals/decision_stump.png">

**1.1) Which one of the two attributes resulted in the best split of the original data? How do you select the best attribute to split a tree at each node?** _(Hint: Mention splitting criteria)_


```python
# Your answer here 
```


```python
# __SOLUTION__
# Attribute A generates the best split for the data. 
# The best attribute to split a tree at each node is selected by considering 
# the attribute that creates the purest child nodes. Gini impurity and information 
# gain are two criteria that can be used to measure the quality of a split.
```

### Decision Trees for Regression 

In this section, you will use decision trees to fit a regression model to the Combined Cycle Power Plant dataset. 

This dataset is from the UCI ML Dataset Repository, and has been included in the `data` folder of this repository as an Excel `.xlsx` file, `Folds5x2_pp.xlsx`. 

The features of this dataset consist of hourly average ambient variables taken from various sensors located around a power plant that record the ambient variables every second.  
- Temperature (AT) 
- Ambient Pressure (AP) 
- Relative Humidity (RH)
- Exhaust Vacuum (V) 

The target to predict is the net hourly electrical energy output (PE). 

The features and target variables are not normalized.

In the cells below, we import `pandas` and `numpy` for you, and we load the data into a pandas DataFrame. We also include code to inspect the first five rows and get the shape of the DataFrame.


```python
import pandas as pd 
import numpy as np 

# Load the data
filename = 'raw_data/Folds5x2_pp.xlsx'
df = pd.read_excel(filename)
```


```python
# __SOLUTION__
import pandas as pd 
import numpy as np 

# Load the data
filename = 'raw_data/Folds5x2_pp.xlsx'
df = pd.read_excel(filename)
```


```python
# Inspect the first five rows of the dataframe
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AT</th>
      <th>V</th>
      <th>AP</th>
      <th>RH</th>
      <th>PE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.96</td>
      <td>41.76</td>
      <td>1024.07</td>
      <td>73.17</td>
      <td>463.26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25.18</td>
      <td>62.96</td>
      <td>1020.04</td>
      <td>59.08</td>
      <td>444.37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.11</td>
      <td>39.40</td>
      <td>1012.16</td>
      <td>92.14</td>
      <td>488.56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.86</td>
      <td>57.32</td>
      <td>1010.24</td>
      <td>76.64</td>
      <td>446.48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.82</td>
      <td>37.50</td>
      <td>1009.23</td>
      <td>96.62</td>
      <td>473.90</td>
    </tr>
  </tbody>
</table>
</div>




```python
# __SOLUTION__
# Inspect the first five rows of the dataframe
df.head()
```


```python
# Get the shape of the dataframe 
df.shape
```


```python
# __SOLUTION__
# Get the shape of the dataframe 
df.shape
```

Before fitting any models, you need to create training and testing splits for the data.

Below, we split the data into features and target ('PE') for you. 


```python
X = df[df.columns.difference(['PE'])]
y = df['PE']
```


```python
# __SOLUTION__
X = df[df.columns.difference(['PE'])]
y = df['PE']
```

**1.2) Split the data into training and test sets. Create training and test sets with `test_size=0.5` and `random_state=1`.** 


```python
# Your code here. Replace None with appropriate code. 

X_train, X_test, y_train, y_test = None
```


```python
# __SOLUTION__
# Include relevant imports 
from sklearn.model_selection import train_test_split

# Create training and test sets with test_size=0.5 and random_state=1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
```

**1.3) Fit a vanilla decision tree regression model with scikit-learn to the training data.** Set `random_state = 1` for reproducibility. **Evaluate the model on the test data.** 


```python
# Your code here 
```


```python
# __SOLUTION__
# Bring in necessary imports 
from sklearn.tree import DecisionTreeRegressor

# Fit the model to the training data 
dt = DecisionTreeRegressor(random_state=1)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
```

**1.4) Obtain the mean squared error, mean absolute error, and coefficient of determination (r2 score) of the predictions on the test set.** _Hint: Look at the `sklearn.metrics` module._


```python
# Your code here. Replace None with appropriate code. 

print("Mean Squared Error:", None)
print("Mean Absolute Error:", None)
print("R-squared:", None)
```


```python
# __SOLUTION__
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))
```

    Mean Squared Error: 22.21041691053512
    Mean Absolute Error: 3.223405100334449
    R-squared: 0.9250580726905822


Hint: MSE = 22.21041691053512

### Hyperparameter Tuning of Decision Trees for Regression

For this next section feel free to refer to the scikit learn documentation on [decision tree regressors](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

**1.5) Add hyperparameters to a a new decision tree and fit it to our training data and evaluate the model with the test data.**


```python
# Your code here 
```


```python
# __SOLUTION__ 
# Evaluate the model on test data 
dt_tuned = DecisionTreeRegressor(
    random_state=1,
    max_depth=3,
    min_samples_leaf=2,
)
dt_tuned.fit(X_train,y_train)
y_pred_tuned = dt_tuned.predict(X_test)
```

**1.6) Obtain the mean squared error, mean absolute error, and coefficient of determination (r2 score) of the predictions on the test set. Did this improve your previous model? (It's ok if it didn't)**


```python
# Your answer and explanation here
```


```python
# __SOLUTION__

# Example: adjusting the max depth changes how many splits can happen on a single branch.
# Setting this to three helped improve the model and reduced overfitting.

print("Mean Squared Error:", mean_squared_error(y_test, y_pred_tuned))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_tuned))
print("R-squared:", r2_score(y_test, y_pred_tuned))
```

    Mean Squared Error: 26.619897534087755
    Mean Absolute Error: 4.01892182530331
    R-squared: 0.9101796947792777


## Ensemble Methods [Suggested Time: 15 min]

### Random Forests and Hyperparameter Tuning using GridSearchCV

In this section, you will perform hyperparameter tuning for a Random Forest classifier using GridSearchCV. You will use `scikit-learn`'s wine dataset to classify wines into one of three different classes. 

After finding the best estimator, you will interpret the best model's feature importances. 

In the cells below, we have loaded the relevant imports and the wine data for you. 


```python
# Relevant imports 
from sklearn.datasets import load_wine

# Load the data 
wine = load_wine()
X, y = load_wine(return_X_y=True)
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'target'
df = pd.concat([X, y.to_frame()], axis=1)
```


```python
# __SOLUTION__
# Relevant imports 
from sklearn.datasets import load_wine

# Load the data 
wine = load_wine()
X, y = load_wine(return_X_y=True)
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'target'
df = pd.concat([X, y.to_frame()], axis=1)
```

In the cells below, we inspect the first five rows of the dataframe and compute the dataframe's shape.


```python
df.head()
```


```python
# __SOLUTION__
df.head()
```


```python
df.shape
```


```python
# __SOLUTION__
df.shape
```

We also get descriptive statistics for the dataset features, and obtain the distribution of classes in the dataset. 


```python
X.describe()
```


```python
# __SOLUTION__
X.describe()
```


```python
y.value_counts().sort_index()
```


```python
# __SOLUTION__
y.value_counts().sort_index()
```

You will now perform hyper-parameter tuning for a Random Forest classifier.

**4.1) Construct a `param_grid` dictionary to pass to `GridSearchCV` when instantiating the object. Choose at least 3 hyper-parameters to tune and 3 values for each.** 


```python
# Replace None with relevant code 
param_grid = None
```


```python
# __SOLUTION__
#this is only an example (student's answers will likely be different)
param_grid = { 
    'n_estimators': [5,10,15,20],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6],
    'criterion' :['gini', 'entropy']}
```

Now that you have created the `param_grid` dictionary of hyperparameters, let's continue performing hyperparameter optimization of a Random Forest Classifier. 

In the cell below, we include the relevant imports for you.


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
```


```python
# __SOLUTION__
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
```

**4.2) Create an instance of a Random Forest classifier estimator; call it `rfc`.** Make sure to set `random_state=42` for reproducibility. 


```python
# Replace None with appropriate code
rfc = None
```


```python
# __SOLUTION__
rfc = RandomForestClassifier(random_state=42)
```

**4.3) Create an instance of an `GridSearchCV` object and fit it to the data.** Call the instance `cv_rfc`. 

* Use the random forest classification estimator you instantiated in the cell above, the parameter grid dictionary constructed, and make sure to perform 5-fold cross validation. 
* The fitting process should take 10 - 15 seconds to complete. 


```python
# Replace None with appropriate code 
cv_rfc = None 

cv_rfc.fit(None, None)
```


```python
# __SOLUTION__
# Create an instance of a `GridSearchCV` object with the appropriate params. 
cv_rfc = GridSearchCV(estimator=rfc, 
                      param_grid=param_grid, 
                      cv = 5)

# Fit it to the data
cv_rfc.fit(X, y)
```

**4.4) What are the best training parameters found by GridSearchCV?** 

_Hint: Explore the documentation for GridSearchCV._ 


```python
# Replace None with appropriate code 
None 
```


```python
# __SOLUTION__
cv_rfc.best_params_
```

In the cell below, we create a variable `best_model` that holds the best model found by the grid search.


```python
best_model = cv_rfc.best_estimator_
```


```python
# __SOLUTION__
best_model = cv_rfc.best_estimator_
```

Next, we give you a function that creates a horizontal bar plot to visualize the feature importances of a model, sorted in descending order. 


```python
import matplotlib.pyplot as plt 
%matplotlib inline 

def create_plot_of_feature_importances(model, X):
    ''' 
    Inputs: 
    
    model: A trained ensemble model instance
    X: a dataframe of the features used to train the model
    '''
    
    feat_importances = model.feature_importances_

    features_and_importances = zip(X.columns, feat_importances)
    features_and_importances = sorted(features_and_importances, 
                                     key = lambda x: x[1], reverse=True)
    
    features = [i[0] for i in features_and_importances]
    importances = [i[1] for i in features_and_importances]
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.gca().invert_yaxis()
    plt.title('Feature Importances')
    plt.xlabel('importance')
```


```python
# __SOLUTION__
import matplotlib.pyplot as plt 
%matplotlib inline 

def create_plot_of_feature_importances(model, X):
    ''' 
    Inputs: 
    
    model: A trained ensemble model instance
    X: a dataframe of the features used to train the model
    '''
    
    feat_importances = model.feature_importances_

    features_and_importances = zip(X.columns, feat_importances)
    features_and_importances = sorted(features_and_importances, 
                                     key = lambda x: x[1], reverse=True)
    
    features = [i[0] for i in features_and_importances]
    importances = [i[1] for i in features_and_importances]
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.gca().invert_yaxis()
    plt.title('Feature Importances')
    plt.xlabel('importance')
```

**4.5) Create a plot of the best model's feature importances.** 

_Hint: To create the plot, pass the appropriate parameters to the function above._


```python
# Your code here.
```


```python
# __SOLUTION__
create_plot_of_feature_importances(best_model, X)
```

**4.6) What are this model's top 3 features in order of descending importance?**


```python
# Your answer here 
```


```python
# __SOLUTION__
# flavanoids, color_intensity, alcohol
# Note: this may vary depending how the student tuned the model
```
