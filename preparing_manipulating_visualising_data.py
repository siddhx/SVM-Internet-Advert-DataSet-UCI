
# coding: utf-8

# Internet Advertisements Data Set from the UC Irvine Machine Learning Repository (http://archive.ics.uci.edu).
# Machine learning for web
import pandas as pd
import numpy as np


# Advertisement Dataset for prediciting click-trough rate
# by Scalar Vector Machine (SVM)
df = pd.read_csv('./ad-dataset/ad.data', header=None)


df = df.replace({'?': np.NAN})
df = df.replace({'  ?': np.NAN})
df = df.replace({'   ?': np.NAN})
df = df.replace({'    ?': np.NAN})
df = df.replace({'     ?': np.NAN})
df = df.fillna(-1)


df.head()


# Each ad. label has been transformed into 1
adindices = df[df.columns[-1]] == 'ad.'
df.loc[adindices,df.columns[-1]] = 1

# while the nonad. values have been replaced by 0.
nonadindices = df[df.columns[-1]] == 'nonad.'
df.loc[nonadindices,df.columns[-1]] = 0

# All the columns (features) need to be numeric and float types
# (using the astype function and the to_numeric function through a lambda function).
df[df.columns[-1]] = df[df.columns[-1]].astype(float)
df.apply(lambda x: pd.to_numeric(x))


dataset = df.values[:, :]
np.random.shuffle(dataset)

# The -1 notation indicates the last column of the array is not considered.
data = dataset[:, :-1]
labels = dataset[:, -1].astype(float)

# we split the data into two sets: a training set (80%) and a test set (20%):
ntrainrows = int(len(data)*.8)
train = data[:ntrainrows, :]
trainlabels = labels[:ntrainrows]
test = data[ntrainrows:, :]
testlabels = labels[ntrainrows:]


from sklearn.svm import SVC
clf = SVC(gamma=0.001, C=100.)
clf.fit(train, trainlabels)


score = clf.score(test, testlabels)
print('score:', score)



