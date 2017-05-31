# BigDataAnalytics-HW2

  ## Gradient Boosting 參數組合分析
 


```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

data = pd.read_csv('LargeTrain.csv')
train = pd.DataFrame(data)
label = 'Class'
feature = [x for x in train.columns if x not in [label]]
```
test  n_estimators as the feature

```python
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,
max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1,n_jobs=4,iid=False, cv=5)

grid1.fit(data[feature],data[label])
grid1.grid_scores_, grid1.best_params_, grid1.best_score_
```
test  max_features  as the feature

```python
param_test2 = {'max_features':range(100, 1801,100)}
grid2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth =9,
min_samples_split=200,subsample=0.8, random_state=10,min_samples_leaf=60)
,param_grid = param_test2,n_jobs=4,iid=False, cv=5 )
grid2.fit(data[feature],data[label])
grid2.grid_scores_, grid2.best_params_, grid2.best_score_
```

test  min_samples_leaf  as the feature

```python
param_test3 = {'min_samples_leaf':range(30,71,10)}
grid3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth =9,
min_samples_split=200,max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test3,n_jobs=4,iid=False, cv=5)
grid3.fit(data[feature],data[label])
grid3.grid_scores_, grid3.best_params_, grid3.best_score_
```
test  max_depth & min_samples_split as the feature

```python
param_test4 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}

grid4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_features='sqrt',
subsample=0.8, random_state=10), 
param_grid = param_test4,n_jobs=4,iid=False, cv=5)
grid4.fit(data[feature],data[label])
grid4.grid_scores_, grid4.best_params_, grid4.best_score_
```

test  subsample as the feature

```python
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
grid5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth =9,
min_samples_split=200, random_state=10,min_samples_leaf=60,max_features=1800)
,param_grid = param_test5,n_jobs=4,iid=False, cv=5 )
grid5.fit(data[feature],data[label])
grid5.grid_scores_, grid5.best_params_, grid5.best_score_
```

  * 最佳組合的參數組合為

    n_estimators = 80
    max_depth = 10
    min_samples_split = 200
    min_samples_leaf = 50
    max_features = 1900
    subsample = 0.8

  ## Confusion Matrix 驗證 
 ```python
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm , classes , normalize=False , title='Confusion matrix' , cmap=plt.cm.Blues):
    plt.imshow(cm , interpolation='nearest' , cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks , classes , rotation=45)
    plt.yticks(tick_marks , classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    else:
        print('Confusion matrix , without normalization')
    print(cm)
    
    thresh = cm.max()/2.
    for i , j in itertools.product(range(cm.shape[0]) , range(cm.shape[1])):
        plt.text(j , i , cm[i,j] , horizontalalignment='center' , color='white' if cm[i,j]>thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    
data = pd.read_csv('LargeTrain.csv')
target = 'Class'
train = [x for x in data.columns if x!= target]
class_name = ['Class' + str(x) for x in range(1,10)]
X = data[train]
y = data[target]

X_train , X_test , y_train , y_test = train_test_split(X, y , random_state=0)
clf = GradientBoostingClassifier(n_estimators=80,max_depth=11,min_samples_split=200,min_samples_leaf=40,subsample=0.9,max_features=19)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

cnf_matrix = confusion_matrix(y_test , y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_name,title='Confusion matrix , without normalization')
plt.show  
```
  ![alt text](https://github.com/KuanChiChiu/Big_Data_Analytics_HW2/blob/master/GB.jpg)  
    
    
  ## Xgboost 參數組合分析
 


```python
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics  
from sklearn.grid_search import GridSearchCV   
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

data = pd.read_csv('LargeTrain1.csv')
train = pd.DataFrame(data)
label = 'Class'
feature = [x for x in train.columns if x not in [label]]
```
test  max_depth and min_child_weight as the feature

```python
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, max_depth=5,min_child_weight=1,
 gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'multi:softmax', scale_pos_weight=1, seed=27),
 param_grid = param_test1 , n_jobs=4 , iid=False , cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```
test  gamma  as the feature

```python
param_test2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, 
  max_depth=9,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
  objective= 'multi:softmax', scale_pos_weight=1,seed=27), 
  param_grid = param_test2, n_jobs=4 , iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
```

test  subsample ＆ colsample_bytree  as the feature

```python
param_test3 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, max_depth=9,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', scale_pos_weight=1,seed=27), 
 param_grid = param_test3, n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
```
test  reg_alpha as the feature

```python
param_test4 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, max_depth=9,
 min_child_weight=1, gamma=0, subsample=0.9 , colsample_bytree=0.6 ,
 objective= 'multi:softmax', scale_pos_weight=1,seed=27), 
 param_grid = param_test4 ,n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
```


  * 最佳組合的參數組合為

    max_depth=9
    min_child_weight=1
    gamma=0
    subsample=0.9
    colsample_bytree =0.6
    reg_alpha=0.1
  ## Confusion Matrix 驗證 
 ```python
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm , classes , normalize=False , title='Confusion matrix' , cmap=plt.cm.Blues):
    plt.imshow(cm , interpolation='nearest' , cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks , classes , rotation=45)
    plt.yticks(tick_marks , classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    else:
        print('Confusion matrix , without normalization')
    print(cm)
    
    thresh = cm.max()/2.
    for i , j in itertools.product(range(cm.shape[0]) , range(cm.shape[1])):
        plt.text(j , i , cm[i,j] , horizontalalignment='center' , color='white' if cm[i,j]>thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    
data = pd.read_csv('LargeTrain.csv')
target = 'Class'
train = [x for x in data.columns if x!= target]
class_name = ['Class' + str(x) for x in range(1,10)]
X = data[train]
y = data[target]

X_train , X_test , y_train , y_test = train_test_split(X, y , random_state=0)
clf = XGBClassifier(max_depth=9,min_child_weight=1,gamma=0,subsample=0.9,colsample_bytree=0.6,reg_alpha=0.1)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

cnf_matrix = confusion_matrix(y_test , y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_name,normalize=True,title='Normalized confusion matrix')
plt.show
```
  ![alt text](https://github.com/KuanChiChiu/Big_Data_Analytics_HW2/blob/master/Xgb.jpg)    
