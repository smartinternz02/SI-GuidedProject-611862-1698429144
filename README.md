# SI-GuidedProject-611862-1698429144

### CODE (MODEL BUILDING AND DATA PREPROCESSING)

#### Importing needed libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

#### Defining the Column names for the dataset
```
col_names = ['class', 'lymphatics', 'block of affere', 'bl. of lymph. c', 'bl. of lymph. s', 'by pass', 'extravasates', 'regeneration of', 'early uptake in', 'lym.nodes dimin', 'lym.nodes enlar', 'changes in lym.', 'defect in node', 'changes in node', 'changes in stru', 'special forms', 'dislocation of', 'exclusion of no', 'no. of nodes in']
```

#### Retrieving Data from a third party website
```
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data",names=col_names)
```

#### Analysing data
```
df.head()
```
<img width="730" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/f170eb01-fd3b-4447-be31-47dfd1b9ef7d">

```
df.tail()
```
<img width="720" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/18bc7635-7e9f-48c4-81c2-766e127d0905">

```
df.info()
```
<img width="297" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/842507a1-719e-4fe5-82f0-ee42c611b014">

```
df.isnull().any()
```
<img width="191" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/a66b2107-a00a-4dd6-a207-441341aa84ac">

```
df.isnull().sum()
```
<img width="151" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/1260f1be-64fa-411a-8279-817aaf89a3cc">

```
df.describe()
```
<img width="729" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/c767cf16-c67d-4e6f-a784-afb32a41a7ce">

```
cor=df.corr()
plt.figure(figsize=(15,20))
sns.heatmap(cor,annot=True)
```
![image](https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/1499e204-61af-40d0-a351-1d5a290e76b5)

```
sns.distplot(df["by pass"],color='g')
```
![image](https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/4cf01209-e212-42c0-8415-c05f6f1ab994)

```
sns.countplot(data=df,y="by pass",hue="class")
```
![image](https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/9ac38691-4360-40af-9971-5c9f99da138c)

```
df.shape
```
(148, 19)

```
sns.boxplot(df["block of affere"])
```
![image](https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/884da9e5-d495-4716-9f3e-40d66b667854)

```
sns.boxplot(df["bl. of lymph. c"])
```
![image](https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/3bb8b848-2808-4a82-97d3-515e4f9bb64f)

```
plt.figure(figsize=(8,5))
sns.boxplot(df["bl. of lymph. s"])
```
![image](https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/937bda4e-2e7b-470d-8410-cce39e93ae3e)

```
x=df.iloc[:,1:]
x.head()
```
<img width="727" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/7fd4eae1-f0f6-4486-989c-69affc0ebda6">

```
y=df["class"]
y
```
<img width="275" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/e3fac50b-1fdc-47ec-a243-c306d1f6c445">

```
x.shape
```
(148, 18)

```
y.shape
```
(148,)

```
y.nunique()
```
4

```
y.unique()
```
array([3, 2, 4, 1], dtype=int64)

```
x.info()
```
<img width="296" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/d072fb17-eaf4-4014-88f7-e6fa0d7300f5">


#### Feature Scaling
```
#feature scaling
from sklearn.preprocessing import MinMaxScaler
ms=MinMaxScaler()
x_scaled=pd.DataFrame(ms.fit_transform(x),columns=x.columns)
x_scaled
```
<img width="725" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/dc65c39b-6977-4bfe-833b-5777d72502cc">

#### Splitting the Data into train,test
```
#Splitting Data into Train and Test.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
```
((118, 18), (30, 18), (118,), (30,))

```
x_train.head()
```
<img width="723" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/98b1ff32-8032-4f9e-95e0-c2fe75a534e2">

#### Model Initialization
```
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
params={
    'max_depth':[9,10,11],
    'min_samples_leaf':[2,3],
    'n_estimators':[90,95,100,110],
    'max_features':[2,3,4,5]
}
```
```
from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(estimator=rf,
                        param_grid=params,
                        cv=2,
                        verbose=1,
                        scoring="accuracy")
```
```
grid_search.fit(x_train,y_train)
```
<img width="397" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/e6913dd3-6498-4159-963b-cee7da806869">

```
grid_search.best_score_
```
0.7966101694915254

```
rf_best=grid_search.best_estimator_
rf_best
```
<img width="451" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/52ce35ea-8c37-49ec-87de-0fe4a57f2791">

```
rf_classify=RandomForestClassifier(random_state=42,
                                   n_jobs=-1,
                                   max_depth=9,
                                   min_samples_split=2,
                                   max_features='sqrt',
                                   n_estimators=90,
                                   bootstrap=True)
```
```
rf_classify.fit(x_train,y_train)
```
<img width="485" alt="image" src="https://github.com/smartinternz02/SI-GuidedProject-611862-1698429144/assets/105509542/d1078278-6dcf-4251-88bc-c6563782e6d9">

```
from sklearn.metrics import accuracy_score
prediction=rf_classify.predict(x_test)
```
```
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
confusion_matrix(y_test,prediction)
```
array([[11,  1,  0],
       [ 2, 15,  0],
       [ 1,  0,  0]], dtype=int64)


```
accuracy_score(y_test,prediction)
```
0.8666666666666667

```
print(classification_report(y_test,prediction))
```
       precision    recall  f1-score   support

           2       0.79      0.92      0.85        12
           3       0.94      0.88      0.91        17
           4       0.00      0.00      0.00         1

    accuracy                           0.87        30
   macro avg       0.57      0.60      0.59        30
weighted avg       0.85      0.87      0.85        30


#### Saving the model and scalar
```
import pickle

pickle.dump(rf_classify,open('saved_model.pkl','wb'))
pickle.dump(ms,open('ms_saved.pkl','wb'))
```

##### By following the above code and order, you can easily build an ML model.

### Thankyou!















