import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("fifa2026_player_performance_clean.csv")
data.columns=("player_name", "position", "age", "competition", "minutes_played", "goals", "assists", "shots_on_target", "pass_accuracy_pct", "tackles", "distance_covered_km", "match_rating", "yellow_card", "red_card", "fouls_committed", "injured_flag")
print(data.head())
print(data.describe())
print(data.info())

g = data["goals"].quantile(0.20)
a = data["assists"].quantile(0.15)
s = data["shots_on_target"].quantile(0.10)
p = data["pass_accuracy_pct"].quantile(0.15)
t = data["tackles"].quantile(0.10)
d = data["distance_covered_km"].quantile(0.05)
m = data["match_rating"].quantile(0.25)

def wrating(x, g=g, a=a, s=s, p=p, t=t, d=d, m=m):
    goals = x["goals"]
    assists = x["assists"]
    shots_on_target = x["shots_on_target"]
    pass_accuracy_pct = x["pass_accuracy_pct"]
    tackles = x["tackles"]
    distance_covered_km = x["distance_covered_km"]
    match_rating = x["match_rating"]

    wr = (goals/(goals+g) + assists/(assists+a) + shots_on_target/(shots_on_target+s) + pass_accuracy_pct/(pass_accuracy_pct+p) + tackles/(tackles+t) + distance_covered_km/(distance_covered_km+d) + match_rating/(match_rating+m)) / 7
    return wr

data["score"]=data.apply(wrating,axis=1)

print(data[['player_name','position','score']].head(10))
