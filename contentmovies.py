import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies=pd.read_csv("movies_metadata.csv")
print(movies.head())
print(movies.info())
tfidf= TfidfVectorizer(stop_words="english")

movies['overview']=movies['overview'].fillna('')
tfidfmatrix=tfidf.fit_transform(movies['overview'])
print(tfidfmatrix.shape)
print(tfidf.get_feature_names_out()[0:40000])
cosinesimilarity=linear_kernel(tfidfmatrix,tfidfmatrix)
indices=pd.Series(movies.index,index=movies['title']).drop_duplicates()
def getrecommendation(title,cos_sim=cosinesimilarity):
    idx=indices[title]
    simscore=list(enumerate(cos_sim[idx]))
    simscore=sorted(simscore,key=lambda x:x[1],reverse=True)
    simscore=simscore[1:11]
    movieindices=[i[0]for i in simscore]
    movietitles=movies["title"].iloc[movieindices]
    return movietitles

print(getrecommendation("The Dark Knight Rises"))