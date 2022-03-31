# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 17:23:45 2021

@author: Rold
"""

import numpy as np

class preprocessing:
    
    def __init__(self,fileX,fileY):
        

        self.num_x = 0
        self.num_y = 0
        
        """Comptage du nombre d'individus"""
        ligne_x = open(fileX,"r",encoding="utf8")
        for i in ligne_x.readlines():
            
            self.num_x = self.num_x + 1

        
        ligne_x.close()
        
     
        
        ligne_y = open(fileY,"r",encoding="utf8")
        for i in ligne_y.readlines():
            self.num_y = self.num_y + 1

        """Initialisation de X  """

        self.X = np.zeros((self.num_x,34))
        self.Y = []   


        if (self.num_x != self.num_y):
            print("Fichiers incomplets")
        else:
            lect_x = open(fileX,"r",encoding="utf8")
            
            it = 0
            for i in lect_x.readlines():
                
                l = i.split(";")
             
                for j in range(0,len(l)):
 
                   self.X[it][j]=int(l[j])

                it = it + 1

            """Initialisation de Y """
         
            self.y_out(fileY)   
        
        
            ligne_y.close()
        

    """Traiter nos données sur les positions des joueurs""" 
    def y_out(self,fileY): 
         lect_y = open(fileY,"r",encoding="utf8")
         for i in range(0,self.num_y):
             l = lect_y.readline().strip()
             self.Y.append(l)

         lect_y.close()    

         
        
        
        