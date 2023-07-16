# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:04:04 2021

@author: argdi
"""

import numpy as np
from scipy.special import gamma
import random

class configuration_model:
    def __init__(self, seedSim, N1, N2, kRound1, kRound2, kRound12, kRound21, graph):
        np.random.seed(seedSim)
        #initialize the desired degrees of each node as integers
        #in configuration model, every pair of nodes that has available links has the same probability of being formulated
        self.graph = graph
        self.N1 = N1
        self.N2 = N2
        self.kRound1 = np.array(np.rint(kRound1), dtype=int)
        self.kRound2 = np.array(np.rint(kRound2), dtype=int)
        self.kRound12 = np.array(np.rint(kRound12), dtype=int)
        self.kRound21 = np.array(np.rint(kRound21), dtype=int)
        
    def createEdges1(self):
        #we formulate the links of the first group
        
        end = False
        while end==False:
            for i in range(len(self.kRound1)):
                if self.kRound1[i] > 0:
                    #tempJ includes all the nodes that have not completed their expected connectivity degree
                    tempJ = [node for node in range(self.N1) if self.kRound1[node] > 0 and node!=i]
                    if len(tempJ) < 1: #when there are not available nodes stop the procedure
                        end = True
                        break
                    j = np.random.choice(tempJ) #pick a random available node
                    self.graph.add_edge(i, j) 
                    
                    self.kRound1[i] -= 1 #we reduce the available links
                    self.kRound1[j] -= 1
                    
                
    def createEdges2(self):                
                
        #same procedure as in createEdges1. Here we formulate the links of the second group
        end = False
        while end==False:
            for i in range(len(self.kRound2)):
                if self.kRound2[i] > 0:
                    tempJ = [node for node in range(self.N2) if self.kRound2[node] > 0 and node!=i] 
                    if len(tempJ) < 1:
                        end = True
                        break
                    j = np.random.choice(tempJ)
                    self.graph.add_edge(i, j)
                    
                    self.kRound2[i] -= 1
                    self.kRound2[j] -= 1
                    
                    
    
    def createEdges12(self):
        #formulate the interlinks between the two groups
        i = 0 
        endBool = False
        while True:
            for i in range(self.N1 + self.N2): #loop through the nodes of the whole graph
                #when the node belongs to the first group, we check if the nodes of the second group have available connections with the first group
                if i < self.N1:
                    if self.kRound12[i] > 0:
                        tempJ = np.where(self.kRound21>0)[0] 
                        if len(tempJ) < 1:
                            endBool = True
                            break
                        j = np.random.choice(tempJ)
                        self.graph.add_edge(i, self.N1 + j)
                        self.kRound12[i] -= 1
                        self.kRound21[j] -= 1
                #if the node belongs to the second group, we check if the nodes of the second group have available connections with the first group
                if i > self.N1-1:
                    if self.kRound21[i - self.N1] > 0:
                        tempJ = np.where(self.kRound12>0)[0]
                        if len(tempJ) < 1:
                            endBool = True
                            break
                        j = np.random.choice(tempJ)
                        self.graph.add_edge(i, j)
                        self.kRound12[j] -= 1
                        self.kRound21[i-self.N1] -= 1
            if endBool:
                break
            
    def run(self):

        #we create the links
        self.createEdges1()
        self.createEdges2()
        self.createEdges12()
        
        return [self.graph, self.kRound1, self.kRound2, self.kRound12, self.kRound21]
    

class prefAttachDegree:
    def __init__(self, seedSim, N1, N2, kRound1, kRound2, kRound12, kRound21, graph):
        #initialize the desired degrees of each node as integers
        #in the model of this class, each node has a higher probability to be connected to a node with a higher degree
        np.random.seed(seedSim)
        self.graph = graph
        self.N1 = N1
        self.N2 = N2
        self.kRound1 = np.array(np.rint(kRound1), dtype=int)
        self.kRound2 = np.array(np.rint(kRound2), dtype=int)
        self.kRound12 = np.array(np.rint(kRound12), dtype=int)
        self.kRound21 = np.array(np.rint(kRound21), dtype=int)
        
        #completed links 
        self.k11Completed = np.zeros(len(kRound1), dtype=int)
        self.k22Completed = np.zeros(len(kRound2), dtype=int)
        self.k12Completed = np.zeros(len(kRound12), dtype=int)
        self.k21Completed = np.zeros(len(kRound21), dtype=int)
        
        
    def createEdges1(self):
        
        #we formulate the links of the first group
        end = False
        while end==False:
            for i in range(len(self.kRound1)): #loop through the nodes of the group
                if self.kRound1[i] > 0:
                    #tempJ includes all the nodes that have not completed their expected connectivity degree
                    tempJ = [node for node in range(self.N1) if self.kRound1[node] > 0 and node!=i] 
                    if len(tempJ) < 1: #if there are no available links stop the procedure
                        end = True
                        break
                    
                    #the connection probability will be proportional to the degree of the node j
                    j = np.random.choice(tempJ, p = self.kRound1[tempJ]/sum(self.kRound1[tempJ]))
                    self.graph.add_edge(i, j)
                    
                    #vector of the completed links 
                    self.k11Completed[i] += 1
                    self.k11Completed[j] += 1
                    
                    #when the completed links are equal to the desired ones make self.kRound1[i] = 0
                    if self.k11Completed[i] + 0.1 >= self.kRound1[i]:
                        self.kRound1[i] = 0 
                    
                    if self.k11Completed[j] + 0.1 >= self.kRound1[j]:
                        self.kRound1[j] = 0
            
                
    def createEdges2(self):
        #same procedure as in createEdges1. Here we formulate the links of the second group
        end = False
        while end==False:
            for i in range(len(self.kRound2)):
                if self.kRound2[i] > 0:
                    tempJ = [node for node in range(self.N2) if self.kRound2[node] > 0 and node!=i] #indices in k arr
                    if len(tempJ) < 1:
                        end = True
                        break
                    j = np.random.choice(tempJ, p = self.kRound2[tempJ]/sum(self.kRound2[tempJ]))
                    self.graph.add_edge(i, j)
                    
                    self.k22Completed[i] += 1
                    self.k22Completed[j] += 1
                    
                    if self.k22Completed[i] + 0.1 >= self.kRound2[i]:
                        self.kRound2[i] = 0 
                    
                    if self.k22Completed[j] + 0.1 >= self.kRound2[j]:
                        self.kRound2[j] = 0
    
    def createEdges12(self):
        i = 0 
        endBool = False
        while True:
            for i in range(self.N1 + self.N2): #loop through all the nodes of the graph
                if i < self.N1:
                    if self.kRound12[i] > 0:
                    
                        tempJ = np.where(self.kRound21>0)[0]
                        if len(tempJ) < 1:
                            endBool = True
                            break
                        #the connection probability will be proportional to the degree of the node j
                        j = np.random.choice(tempJ, p = self.kRound21[tempJ]/sum(self.kRound21[tempJ])) 
                        self.graph.add_edge(i, self.N1 + j)                        
                        self.k12Completed[i] += 1
                        self.k21Completed[j] += 1
                        
                        if self.k12Completed[i] + 1 >= self.kRound12[i]:
                            self.kRound12[i] = 0 
                        
                        if self.k21Completed[j] + 1 >= self.kRound21[j]:
                            self.kRound21[j] = 0
                        
                        
                
                if i > self.N1-1:
                    if self.kRound21[i - self.N1] > 0:
                        tempJ = np.where(self.kRound12>0)[0]
                        if len(tempJ) < 1:
                            endBool = True
                            break
                        #the connection probability will be proportional to the degree of the node j
                        j = np.random.choice(tempJ, p = self.kRound12[tempJ]/sum(self.kRound12[tempJ]))
                        self.graph.add_edge(i, j)
                        
                        self.k12Completed[j] += 1
                        self.k21Completed[i - self.N1] += 1
                        
                        if self.k12Completed[j] + 1 >= self.kRound12[j]:
                            self.kRound12[j] = 0 
                        
                        if self.k21Completed[i - self.N1] + 1 >= self.kRound21[i - self.N1]:
                            self.kRound21[i - self.N1] = 0
                        
                        
            if endBool:
                break
            
    def run(self):
        self.createEdges1()
        self.createEdges2()
        self.createEdges12()
        
        return [self.graph, self.k11Completed, self.k22Completed, self.k12Completed, self.k21Completed]



#implementation of the lognormal distribution
class Lognormal():
    def mu(self, lM, sLi):
        return (np.log(lM*lM/np.sqrt(sLi*sLi + lM*lM)))
    
    def sigma(self, lM, sLi):
        return (np.log(1 + sLi*sLi/(lM*lM)))
    
    def Afun(self, sigma1, sigma2, rho):
        return (1/(2*np.pi * sigma1 * sigma2 * np.sqrt(1 - rho*rho)))
        
    
    def P(self, lM1, lM2, sL1, sL2, rho, l1, l2):
        sigma1, sigma2 = self.sigma(lM1, sL1), self.sigma(lM2, sL2)
        mu1, mu2 = self.mu(lM1, sL1), self.mu(lM2, sL2)
    
        return (self.Afun(sigma1, sigma2, rho)*np.exp( -((np.log(l1) - mu1)**2)/ (2*sigma1*sigma1 * (1 - rho*rho)) -((np.log(l2) - mu2 )**2)/ (2*sigma2 *sigma2 * (1 - rho*rho))
                        + (np.log(l1) - mu1) * (np.log(l2) - mu2 )*rho / (sigma1 * sigma2 *(1 - rho*rho)) )/(l1*l2))
    
    
#implementation of the weibull distribution
class Weibull():
    def a(self, lM, sL, di):
        return ( di * sL*sL/lM )
    
    def b(self, lM, sL):
        return ( lM*lM/(sL*sL) - 1 )
    
    def B(self, lM1, lM2, sL1, sL2, d1, d2):
        return ( d1 * d2 / (self.a(lM1, sL1, d1) * self.a(lM2, sL2, d2) * gamma( (1 + self.b(lM1, sL1)) / d1 ) *
                            gamma( (1 + self.b(lM2, sL2)) / d2 ) ) )

    def P(self, l1, l2, lM1, lM2, sL1, sL2, d1, d2):
        return ( self.B(lM1, lM2, sL1, sL2, d1, d2) * (l1/self.a(lM1, sL1, d1))**self.b(lM1, sL1) 
                * np.exp( -(l1/self.a(lM1, sL1, d1))**d1 ) * (l2/self.a(lM2, sL2, d2))**self.b(lM2, sL2)
                * np.exp( -(l2/self.a(lM2, sL2, d2))**d2 ))



def GibbsSampling(seedNum, N1, N2, lM1, lM12, lM2, lM21, sL1, sL2, rho):
    
    #Gibbs sampling is a random sampling method on a joint distribution
    #we sample from a joint distribution having one of its parameters equal to a value
    #lM1, lM2, lM12, lM21 are the mean values of the distributions
    #sL1, sL2 the standard deviations
    #rho the correlation 
    l1Temp = np.arange(0.01, 50, 0.1) 
    l2Temp = np.arange(0.01, 50, 0.1)
    e = 0.1
    samples = {'l1':[lM1 + e], 'l12':[lM12 - e], 'l2': [lM2 + e], 'l21': [lM21 - e]} #the initial samples will be close to the mean values
    
    
    np.random.seed(seedNum)
    #for the first group
    for i in range(N1-1):
        cur_y = samples['l12'][-1] #for the initial value of l2
        #we create a distribution
        p1 = [Lognormal().P(lM1, lM12, sL1, sL2, rho, l1 , cur_y) for l1 in l1Temp]
        
        while True:
            ind = np.random.randint(0, len(l1Temp)) #random index between 0 and len(l1Temp) - 1
            rand = random.random()*max(p1) + min(p1) #random number between min(p1), max(p1)
            if(rand < p1[ind]): #if rand < value of the distribution of a random index ind, keep l1Temp[ind]
                newx = l1Temp[ind]
                break
        
        #for l1 = newx, perform the same procedure
        p2 = [Lognormal().P(lM1, lM12, sL1, sL2, rho, newx, l2) for l2 in l2Temp]
        while True:
            ind = np.random.randint(0, len(l2Temp))
            rand = random.random()*max(p2) + min(p2)
            if(rand < p2[ind]):
                newy = l2Temp[ind]
                break
        
        samples['l1'].append(newx)
        samples['l12'].append(newy)
        if i%500==0:
            print(i)
    
    #for the second group
    for i in range(N2-1):
        cur_y = samples['l21'][-1]
        p1 = [Lognormal().P(lM2, lM21, sL2, sL1, rho, l2 , cur_y) for l2 in l2Temp]
        
        while True:
            ind = np.random.randint(0, len(l2Temp))
            rand = random.random()*max(p1) + min(p1)
            if(rand < p1[ind]):
                newx = l2Temp[ind]
                break
        
        
        p2 = [Lognormal().P(lM2, lM21, sL2, sL1, rho, newx, l1) for l1 in l1Temp]
        while True:
            ind = np.random.randint(0, len(l1Temp))
            rand = random.random()*max(p2) + min(p2)
            if(rand < p2[ind]):
                newy = l1Temp[ind]
                break
        
        samples['l2'].append(newx)
        samples['l21'].append(newy)
        if i%500==0:
            print(i)

    k1Round = np.array(samples['l1'])
    k2Round = np.array(samples['l2'])
    k12Round = np.array(samples['l12'])
    k21Round = np.array(samples['l21'])*(N1*lM12/(N2*lM21))

    return [k1Round, k2Round, k12Round, k21Round]
    
    
    
    
    