import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

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

rbf_tts_scores = []
rbf_tts_gammas = np.linspace(0.001, 10.0, 100)
rbf_tts_cs = np.linspace(0.001, 10.0, 100)

for gamma in rbf_tts_gammas:
    max_score = -2
    max_c = -2
    
    for c in rbf_tts_cs:
        svr = SVR(gamma=gamma, C=c)
        svr.fit(X_train, y_train)
        score = svr.score(X_test, y_test)
        if score > max_score:
            max_score = score
            max_c = c
            
    rbf_tts_scores.append([gamma, max_c, max_score])
    
poly_tts_scores = []
poly_tts_gammas = np.linspace(0.001, 1, 5)
poly_tts_cs = np.linspace(0.001, 1, 5)
poly_tts_degrees = range(1, 6)

for degree in poly_tts_degrees:
    for gamma in poly_tts_gammas:
        max_score = -2
        max_c = -2
        
        for c in poly_tts_cs:
            svr = SVR(kernel='poly', gamma=gamma, C=c, degree=degree)
            svr.fit(X_train, y_train)
            score = svr.score(X_test, y_test)
            if score > max_score:
                max_score = score
                max_c = c
                
        poly_tts_scores.append([degree, gamma, max_c, max_score])
        
sigmoid_tts_scores = []
sigmoid_tts_gammas = np.linspace(0.0001, 0.1, 100)
sigmoid_tts_cs = np.linspace(0.001, 5.0, 100)

for gamma in sigmoid_tts_gammas:
    max_score = -2
    max_c = -2
    
    for c in sigmoid_tts_cs:
        svr = SVR(kernel='sigmoid', gamma=gamma, C=c)
        svr.fit(X_train, y_train)
        score = svr.score(X_test, y_test)
        if score > max_score:
            max_score = score
            max_c = c
            
    sigmoid_tts_scores.append([gamma, max_c, max_score])

linear_tts_train_scores = []
linear_tts_test_scores = []
linear_tts_cs = np.linspace(0.001, 0.25, 100)

for c in linear_tts_cs:
    linsvr = LinearSVR(C=c)
    linsvr.fit(X_train, y_train)
    linear_tts_train_scores.append(linsvr.score(X_train, y_train))
    linear_tts_test_scores.append(linsvr.score(X_test, y_test))
    
plt.plot(linear_tts_cs, linear_tts_train_scores, label='Complete Set LinearSVR Training Score')
plt.plot(linear_tts_cs, linear_tts_test_scores, label='Complete Set LinearSVR Testing Score')
plt.legend()
plt.figure()

rbf_cv_scores = []
rbf_cv_gammas = np.linspace(0.001, 0.1, 100)
rbf_cv_cs = np.linspace(0.001, 10.0, 100)

for gamma in rbf_cv_gammas:
    max_score = -2
    max_c = -2
    
    for c in rbf_cv_cs:
        score = cross_val_score(SVR(gamma=gamma, C=c), X, y).mean()
        if score > max_score:
            max_score = score
            max_c = c
            
    rbf_cv_scores.append([gamma, max_c, max_score])


poly_cv_scores = []
poly_cv_gammas = np.linspace(0.001, 2, 100)
poly_cv_cs = np.linspace(0.001, 1, 100)
poly_cv_degrees = range(1, 2)

for degree in poly_cv_degrees:
    for gamma in poly_cv_gammas:
        max_score = -2
        max_c = -2
        
        for c in poly_cv_cs:
            score = cross_val_score(SVR(kernel='poly', gamma=gamma, C=c, degree=degree), X, y).mean()
            if score > max_score:
                max_score = score
                max_c = c
                
        poly_cv_scores.append([degree, gamma, max_c, max_score])

sigmoid_cv_scores = []
sigmoid_cv_gammas = np.linspace(0.001, 5.0, 100)
sigmoid_cv_cs = np.linspace(0.001, 5.0, 100)

for gamma in sigmoid_cv_gammas:
    max_score = -2
    max_c = -2
    
    for c in sigmoid_cv_cs:
        score = cross_val_score(SVR(kernel='sigmoid', gamma=gamma, C=c), X, y).mean()
        if score > max_score:
            max_score = score
            max_c = c
            
    sigmoid_cv_scores.append([gamma, max_c, max_score])
    
linear_cv_scores = []
linear_cv_cs = np.linspace(0.001, 0.13, 100)

for c in linear_cv_cs:
    linear_cv_scores.append(cross_val_score(LinearSVR(C=c), X, y).mean())
    
plt.plot(linear_cv_cs, linear_cv_scores, label='Complete Set LinearSVR CV Score')
plt.legend()

