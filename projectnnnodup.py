import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from itertools import combinations

data = pd.read_csv("bestsellnoduptrim.csv", delimiter=",")

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

sizes = range(1, 21)
networks = combinations(sizes, 5)
max_score = -2
max_network = (1, 1, 1, 1, 1)

for network in networks:
    mlp = MLPRegressor(hidden_layer_sizes=network, random_state=1, max_iter=5000000000, activation='logistic', solver='lbfgs')
    mlp.fit(X_train, y_train)
    test_score = mlp.score(X_test, y_test)
    if (test_score > max_score):
        max_score = test_score
        max_network = network
    

print(max_network)
alphas = np.linspace(0, 10, 100)
train_scores = []
test_scores = []

for a in alphas:
    mlp = MLPRegressor(hidden_layer_sizes=max_network, random_state=1, activation='logistic', alpha=a, max_iter=5000000000, solver='lbfgs')
    mlp.fit(X_train, y_train)
    train_scores.append(mlp.score(X_train, y_train))
    test_scores.append(mlp.score(X_test, y_test))
    
plt.plot(alphas, train_scores, label='Complete Set NN Training Score')
plt.plot(alphas, test_scores, label='Complete Set NN Testing Score')
plt.legend()
plt.show()
plt.figure()


max_score = -2
max_network = (1, 1, 1, 1, 1)
networks = combinations(sizes, 5)

for network in networks:
    mlp = MLPRegressor(hidden_layer_sizes=network, random_state=1, max_iter=5000000000, activation='logistic', solver='lbfgs')
    cv_score = cross_val_score(mlp, X, y).mean()
    if (cv_score > max_score):
        max_score = cv_score
        max_network = network
    

print(max_network)
alphas = np.linspace(0, 10, 100)
cv_scores = []

for a in alphas:
    mlp = MLPRegressor(hidden_layer_sizes=max_network, random_state=1, activation='logistic', alpha=a, max_iter=5000000000, solver='lbfgs')
    cv_scores.append(cross_val_score(mlp, X, y).mean())
    
plt.plot(alphas, cv_scores, label='Complete Set NN 5-Fold CV Score')
plt.legend()
plt.show()


    
    


