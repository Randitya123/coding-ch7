import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
nltk.download('stopwords')

ps=PorterStemmer

text1=input("Enter scentence 1: ")
text2=input("Enter scentence 2: ")

words1=text1.split()
words2=text2.split()

fwords1=[]
fwords2=[]

for w in words1:
    if w.lower() not in stopwords.words('english'):
        fwords1.append(w)
for w in words2:
    if w.lower() not in stopwords.words('english'):
        fwords2.append(w)

print(fwords1)
print("/n")
print(fwords2)