#%%
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
# %matplotlib inline


# %%
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%206/cbb.csv')
df.head()

#%%
df.shape

#%%
df['windex'] = np.where(df.WAB > 7, 'True', 'False')

#%%
df1 = df.loc[df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
df1.head()

#%%
df1['POSTSEASON'].value_counts()

#%%
# notice: installing seaborn might takes a few minutes
# !conda install -c anaconda seaborn -y

#%%
import seaborn as sns

bins = np.linspace(df1.BARTHAG.min(), df1.BARTHAG.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=6)
g.map(plt.hist, 'BARTHAG', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

#%%
bins = np.linspace(df1.ADJOE.min(), df1.ADJOE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJOE', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

#%%
bins = np.linspace(df1.ADJDE.min(), df1.ADJDE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJDE', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

#%%
df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize=True)

#%%
df1['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
df1.head()

#%%
X = df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
X[0:5]

#%%
y = df1['POSTSEASON'].values
y[0:5]

#%%
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

#%%
# We split the X into train and test to find the best k
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Validation set:', X_val.shape,  y_val.shape)

#%%
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
k = 5
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(X_val)
yhat[0:5]
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_val, yhat))

#%%
for k in range(15):
    neigh = KNeighborsClassifier(n_neighbors = k+1).fit(X_train,y_train)
    yhat = neigh.predict(X_val)
    print(f'Train set Accuracy for k = {k+1}: {metrics.accuracy_score(y_val, neigh.predict(X_val))}')

#%%
from sklearn.tree import DecisionTreeClassifier
prevAcc = 1
for depth in range(10):
    finalFourTree = DecisionTreeClassifier(criterion="entropy", max_depth = depth+1 )
    finalFourTree.fit(X_train,y_train)
    finalFourTree
    predTree = finalFourTree.predict(X_val)
    print(f'DecisionTree Accuracy for max depth {depth+1}: {metrics.accuracy_score(y_val, predTree)}')
    if depth > 1 and prevAcc > metrics.accuracy_score(y_val, predTree):
        print(f'Minimum max_depth for improvement : {depth}')
        break

#%%
from sklearn import svm
accuracies = []
models = ['linear','poly','rbf','sigmoid']
for model in models:
    clf = svm.SVC(kernel=model)
    clf.fit(X_train, y_train) 

    yhat = clf.predict(X_val)
    from sklearn import metrics
    accuracies.append(metrics.accuracy_score(y_val, yhat))
import operator
index, value = max(enumerate(accuracies), key=operator.itemgetter(1))
print(f'The best kernel was {models[index]} with an accuracy of {accuracies[index]}')
my_svm = svm.SVC(kernel=models[index])

#%%
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(X_val)
yhat_prob = LR.predict_proba(X_val)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_val, yhat)
print(f'Accuracy of logistic regression with C=0.01 is : {accuracy}')


#%%
from sklearn.metrics import f1_score
# for f1_score please set the average parameter to 'micro'
from sklearn.metrics import log_loss
def jaccard_index(predictions, true):
    if (len(predictions) == len(true)):
        intersect = 0;
        for x,y in zip(predictions, true):
            if (x == y):
                intersect += 1
        return intersect / (len(predictions) + len(true) - intersect)
    else:
        return -1

# %%
test_df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0120ENv3/Dataset/ML0101EN_EDX_skill_up/basketball_train.csv',error_bad_lines=False)
test_df.head()

test_df['windex'] = np.where(test_df.WAB > 7, 'True', 'False')
test_df1 = test_df[test_df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
test_Feature = test_df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
test_Feature['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
test_X=test_Feature
test_X= preprocessing.StandardScaler().fit(test_X).transform(test_X)
test_X[0:5]

#%%
test_y = test_df1['POSTSEASON'].values
test_y[0:5]

neigh = KNeighborsClassifier(n_neighbors = 5).fit(X_train,y_train)
yhat = neigh.predict(X_val)
accuracy = metrics.accuracy_score(y_val, yhat)
f1 = f1_score(y_val, yhat, average='micro')
jaccard = jaccard_index(y_val, yhat)
print(f'accuracy {accuracy}, f1 score {f1}, and jaccard score {jaccard}')

#%%
finalFourTree = DecisionTreeClassifier(criterion="entropy", max_depth = 2 )
finalFourTree.fit(X_train,y_train)
yhat = finalFourTree.predict(X_val)
accuracy = metrics.accuracy_score(y_val, yhat)
f1 = f1_score(y_val, yhat, average='micro')
jaccard = jaccard_index(y_val, yhat)
print(f'accuracy {accuracy}, f1 score {f1}, and jaccard score {jaccard}')

#%%
clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train) 
yhat = clf.predict(X_val)
accuracy = metrics.accuracy_score(y_val, yhat)
f1 = f1_score(y_val, yhat, average='micro')
jaccard = jaccard_index(y_val, yhat)
print(f'accuracy {accuracy}, f1 score {f1}, and jaccard score {jaccard}')

#%%
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(X_val)
accuracy = metrics.accuracy_score(y_val, yhat)
f1 = f1_score(y_val, yhat, average='micro')
jaccard = jaccard_index(y_val, yhat)
print(f'accuracy {accuracy}, f1 score {f1}, and jaccard score {jaccard}')
from sklearn.metrics import log_loss
yhat_prob = LR.predict_proba(X_val)
logloss=log_loss(y_val, yhat_prob)
print(f'logloss is {logloss}')

#%%
# | Algorithm          | Accuracy | Jaccard | F1-score | LogLoss |
# | ------------------ | -------- | ------- | -------- | ------- |
# | KNN                | 0.6666666666666666        | 0.5       | 0.6666666666666666        | NA      |
# | Decision Tree      | 0.6666666666666666        | 0.5       | 0.6666666666666666        | NA      |
# | SVM                | 0.6666666666666666        | 0.5       | 0.6666666666666666        | NA      |
# | LogisticRegression | 0.5833333333333334        | 0.4117647058823529       | 0.5833333333333334        | 1.095461062326229       |

