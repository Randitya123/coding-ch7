import pandas as pd 
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

traingdata = pd.read_csv("train.txt", delimiter=";", names=["label", "text"])
testingdata = pd.read_csv("test.txt", delimiter=";", names=["label", "text"])
print(traingdata.head())
print(testingdata.head())

print(traingdata.head["label"].value_counts())


def customencoder(data):
    data.replace(to_replace="suprise",value=1,inplace=True)
    data.replace(to_replace="love",value=1,inplace=True)
    data.replace(to_replace="joy",value=0,inplace=True)
    data.replace(to_replace="sadness",value=1,inplace=True)
    data.replace(to_replace="fear",value=1,inplace=True)
    data.replace(to_replace="anger",value=1,inplace=True)

customencoder(traingdata["label"])
lm = WordNetLemmatizer()
def texttransformation(data):
    corpus = []
    for sentence in data:
        newitem=re.sub("[^a-zA-Z]", " ", str(sentence))
        newitem=newitem.lower()
        newitem=newitem.split()
        newitem=[lm.lemmatize(word) for word in newitem if word not in set(stopwords.words("english"))]
        corpus.append(" ".join(str(x) for x in newitem))
    return corpus

corpus=texttransformation(traingdata)
