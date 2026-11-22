import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("fifa2026_player_performance_clean.csv")
data.columns=("player_name", "position", "age", "competition", "minutes_played", "goals", "assists", "shots_on_target", "pass_accuracy_pct", "tackles", "distance_covered_km", "match_rating", "yellow_card", "red_card", "fouls_committed", "injured_flag")
print(data.head())
print(data.describe())
print(data.info())