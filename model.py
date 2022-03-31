# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 17:23:45 2021

@author: Rold
"""
import torch 
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
import torch.utils.data as data


class Model(nn.Module):
    
    def __init__(self,dim_in,dim_inter,dim_out):
        """Initialiser le modèle
        Param: 
            dim_in : nombre caractéristiques d'un joueur
            dim_out : nombre de postes possibles d'un joueur'
        """
         
        """Appel du constructeur de la classe ancêtre"""
        super(Model,self).__init__()
        
         
        """Initialiser les fonctions de pré-activation et d'activation 
        (Couches du modèle) """
        self.preactivation = nn.Linear(dim_in,dim_inter,bias=True)
        self.c1 = nn.Linear(dim_inter,dim_out,bias=True)
        self.c1_dropout= nn.Dropout(p=0.5)
        self.activation = nn.LogSoftmax(dim=-1)
 


        """Application de notre modèle"""
    def forward(self,X):
        a = self.preactivation(X)
        c = self.c1(a)
        d = self.c1_dropout(c)
        z = self.activation(d)
        return z


        
""" Entraîner et tester notre modèle """

def train(Xtrain,Ytrain,Xvalid,Yvalid,batch_size,epochs,lr):


    model = Model(Xtrain.shape[1], 16 , 4)
    
    losses = []
    f1_valid = [] 
    
    f1_val = []
    losss = []
    
    
    """ Transformer ndarray en torch.tensor """
    t_train_X = torch.tensor(Xtrain,dtype=torch.float)
    t_train_Y = torch.tensor(Ytrain,dtype=torch.long)    

    t_valid_X = torch.tensor(Xvalid,dtype=torch.float)    
    t_valid_Y = torch.tensor(Yvalid,dtype=torch.long)

    """Regrouper Partie Test et Partie Validation"""
    train_set = data.TensorDataset(t_train_X, t_train_Y)
    valid_set = data.TensorDataset(t_valid_X, t_valid_Y)
    
    """Permettre le chargement des données par le modèle"""
    train_loader = data.DataLoader(train_set, batch_size=batch_size)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size)    
    
    """Initialisaer l'optimiseur des paramètres"""
    optim = torch.optim.SGD(model.parameters(),lr=lr)
    
    """Initialiser la fonction de perte"""
    criterion = nn.NLLLoss()

    for e in range (epochs):
    
        """PARTIE TEST"""
    
        model.train()   
        
        
        for batch_ndx, (trX,trY) in enumerate(train_loader):            

            """Réinitiliser les gradients"""  
            optim.zero_grad()
            
            """Calculer la prediction de notre modèle"""
            ypred = model(trX)
            
            """Caluculer la perte"""
            loss = criterion(ypred,trY)

            """La collecter"""
            losses.append(loss)
                  
            """Calcul du gradient"""
            loss.backward()
            
            """Mise à jour des poids et biais"""
            optim.step()
            
               
        """PARTIE VALIDATION """
            
        model.eval()    
        for batch_ndx, (vlX,vlY) in enumerate(valid_loader): 
            
            yvalid_pred = model(vlX)

            yvalid_pred = np.argmax(yvalid_pred.detach().numpy(), axis=1)            
            score = f1_score(vlY,yvalid_pred, average="micro")
            f1_valid.append(score)
        
        f1_val.append(f1_valid[-1])
        losss.append(losses[-1])
        print("Epoch {0}/{1} : Loss = {2:.2f}\tF1_score = {3:.2f}".format(e, epochs, losses[-1], f1_valid[-1]))

    return model,losses, f1_valid, f1_val, losss
            
    
    

    