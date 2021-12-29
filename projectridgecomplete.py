import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

data = pd.read_csv("bestsellnormtrim.csv", delimiter=",")

def replaceGenre(number, data):
    
    newData = data.copy()
    
    #(H, 2019)
    
    newData.loc[data['Genre'] == 'Fiction', ['Genre']] = number
    newData.loc[data['Genre'] == 'Non Fiction', ['Genre']] = number * -1
    
    return newData

#(Rustagi, 2020)

def normalizeFirstThree(data):
    data['Reviews'] = (data['Reviews'] - data['Reviews'].mean()) / data['Reviews'].std()
    data['Price'] = (data['Price'] - data['Price'].mean()) / data['Price'].std()
    data['Year'] = (data['Year'] - data['Year'].mean()) / data['Year'].std()
    
#(Rustagi, 2020)
    
def normalizeGenre(data):
    data['Genre'] = (data['Genre'] - data['Genre'].mean()) / data['Genre'].std()
    
normalizeFirstThree(data)
newData = replaceGenre(5, data)
normalizeGenre(newData)

#(176coding, 2016)

X = newData.iloc[:, 0:4].values
y = newData.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)


alphas = np.linspace(0, 120, 1000)
train_scores = []
test_scores = []

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    train_scores.append(ridge.score(X_train, y_train))
    test_scores.append(ridge.score(X_test, y_test))
    
plt.plot(alphas, train_scores, label='Complete Set Ridge Training Score')
plt.plot(alphas, test_scores, label='Complete Set Ridge Testing Score')
plt.legend()
plt.figure()

alphas = np.linspace(0, 80, 1000)

rr_scores = [cross_val_score(Ridge(alpha=a), X, y, cv=5).mean() for a in alphas]

plt.plot(alphas, rr_scores, label='Complete Set Ridge 5-fold CV Score')
plt.legend()


