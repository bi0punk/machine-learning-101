import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
import seaborn as sns




df = pd.read_csv('titanic/train.csv')
print(df.info())



df = df.drop(['PassengerId', 'Ticket', 'Cabin'],axis =1)
print(df.head(10))