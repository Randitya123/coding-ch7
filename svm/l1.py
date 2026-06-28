from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd

cancer_dict=datasets.load_breast_cancer()
print(cancer_dict.keys())
cancer_data=pd.DataFrame(cancer_dict.data)
cancer_data.columns=cancer_dict.feature_names
cancer_data["isCancer"]=cancer_dict.target

print(cancer_data.info())
print(cancer_data.head())