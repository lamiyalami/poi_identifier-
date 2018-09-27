#!/usr/bin/python
##best solution
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.figure import Figure
sns.set(style="ticks")

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments', 'bonus','total_stock_value', 'deferred_income','exercised_stock_options', 'long_term_incentive','director_fees','poi_tot']  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

##### Task 2: Remove outliers
data_dict.pop("TOTAL",0)

### Task 3: Create new feature(s)
from_poi=[]
to_poi=[]
shar=[]
for k,v in data_dict.items():
    for k1,v1 in v.items():
        if k1=="from_poi_to_this_person":
	    if v1=="NaN":
	        from_poi.append(0)
	    else:
		from_poi.append(v1)
	if k1=="from_this_person_to_poi":
	    if v1=="NaN":
		to_poi.append(0)
	    else:
		to_poi.append(v1)
	if k1=="shared_receipt_with_poi":
	    if v1=="NaN":
		shar.append(0)
	    else:
		shar.append(v1)
tot=[x + y + z for x, y, z in zip(from_poi,to_poi,shar)]

for k,v in data_dict.items():
    for l in tot:
	v["poi_tot"]=l

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features=scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()

# Provided to give you a starting point. Try a variety of classifiers.
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#Scatter Matrix
names = ['salary', 'total_payments', 'bonus','total_stock_value', 'deferred_income','exercised_stock_options', 'long_term_incentive','director_fees','poi_tot']
df = pd.DataFrame(data=features, columns=names)
sns_plot=sns.pairplot(df)
plt.show()

###PCA 
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
pca.fit(features_train)
features_train=pca.transform(features_train)
features_test=pca.transform(features_test)
 
#Classification and Prediction
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

#Evaluation
print accuracy_score(pred,labels_test)
print precision_score(labels_test,pred),recall_score(labels_test,pred)
print confusion_matrix(labels_test,pred)
print classification_report(labels_test,pred)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)#!/usr/bin/python
##best solution
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.figure import Figure
sns.set(style="ticks")

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments', 'bonus','total_stock_value', 'deferred_income','exercised_stock_options', 'long_term_incentive','director_fees','poi_tot']  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

##### Task 2: Remove outliers
data_dict.pop("TOTAL",0)

### Task 3: Create new feature(s)
from_poi=[]
to_poi=[]
shar=[]
for k,v in data_dict.items():
    for k1,v1 in v.items():
        if k1=="from_poi_to_this_person":
	    if v1=="NaN":
	        from_poi.append(0)
	    else:
		from_poi.append(v1)
	if k1=="from_this_person_to_poi":
	    if v1=="NaN":
		to_poi.append(0)
	    else:
		to_poi.append(v1)
	if k1=="shared_receipt_with_poi":
	    if v1=="NaN":
		shar.append(0)
	    else:
		shar.append(v1)
tot=[x + y + z for x, y, z in zip(from_poi,to_poi,shar)]

for k,v in data_dict.items():
    for l in tot:
	v["poi_tot"]=l

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features=scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()

# Provided to give you a starting point. Try a variety of classifiers.
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#Scatter Matrix
names = ['salary', 'total_payments', 'bonus','total_stock_value', 'deferred_income','exercised_stock_options', 'long_term_incentive','director_fees','poi_tot']
df = pd.DataFrame(data=features, columns=names)
sns_plot=sns.pairplot(df)
plt.show()

###PCA 
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
pca.fit(features_train)
features_train=pca.transform(features_train)
features_test=pca.transform(features_test)
 
#Classification and Prediction
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

#Evaluation
print accuracy_score(pred,labels_test)
print precision_score(labels_test,pred),recall_score(labels_test,pred)
print confusion_matrix(labels_test,pred)
print classification_report(labels_test,pred)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
