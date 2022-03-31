# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 17:23:45 2021

@author: Rold
"""
from preprocessing import preprocessing
import numpy as np
from model import train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



file_x = "FullData_FIFA17.csv"
file_y = "Poste.txt"

"""Pré-traitement des données"""

"""IMPORTER X ET Y DANS DES NUMPY ARRAY ET PAS DES LISTES"""
prepro = preprocessing(file_x,file_y)
X = prepro.X
Yold = prepro.Y

Y = np.zeros(len(Yold))

for i in range(0,len(Yold)):
    if Yold[i]=="Gardien":
        Y[i]=0
    if Yold[i]=="Defenseur":
        Y[i]=1
    if Yold[i]=="Milieu":
        Y[i]=2
    if Yold[i]=="Attaquant":
        Y[i]=3
  

""" Centrer et réduire nos données X (Standardisation) """        
sts = StandardScaler()

X = sts.fit_transform(X)  
        


""" Données Test et de Validation """

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2)



""" Entraînement et Validation du modèle """
model, losses, f1_valid, f1_val, losss = train(X_train, y_train, X_valid, y_valid, batch_size=100, epochs=200, lr = 1e-3)


" Affichage résultats"

 
"""Phase Test"""
plt.plot(losss)
plt.title("Evolution fonction de perte")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

"""Phase Validation"""

plt.plot(f1_val)
plt.title("f1 score validation set")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.show()




