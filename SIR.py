# -*- coding: utf-8 -*-


"""
Created on Tue Apr 12 18:02:47 2022

@author: argdi
"""
import networkx as nx
import numpy as np
import sys

class SIR():
    def __init__(self, seedSim, tau, gamma):
        #initializes the network and the lists needed for the simulation
        
        self.N1 = 16000 #population of the first group
        self.N2 = 4000 #population of the second group
        self.N = self.N1 + self.N2
        strategy = 'C'
        
        #load an edgelist created by distrVersion
        self.graph = nx.read_edgelist(str(int(self.N1/1000)) + '_' + str(int(self.N2/1000)) + 'k' + strategy + 'model' + str(int(seedSim/100) + 1) + '.txt', nodetype=int)
        for i in range(self.N): #if a node is missing from the edgelist (has 0 links) add it to the graph
            if i not in list(self.graph.nodes()):
                self.graph.add_node(i)
        
        
    
        np.random.seed(seedSim)
        #self.graph = graph
        self.tau = tau #transmisibillity 
        self.gamma = gamma #recovery rate
        
        self.tMax = 200
        #initialize the time series of S, I, R
        self.I, self.S, self.R = np.zeros(self.tMax), np.zeros(self.tMax), np.zeros(self.tMax)
        self.nei = [list(self.graph.neighbors(i)) for i in range(self.N)] #neighbors of each node
        
        #the first infected will have more than 2 degree connectivity so that the infection
        #will not die out immediately
        sickTemp = [i for i in range(self.N1) if len(self.nei[i])>2]
        self.I_node, self.R_node = np.array([np.random.choice(sickTemp)]), np.array([])
        
        self.S_node = np.array([i for i in range(self.N) if i not in self.I_node])
        self.I[0] = 1
        self.S[0] = self.N - 1


        #time series of the compartmentments S, I, R, in each group
        self.Sgroups = np.zeros((2, 200))
        self.Igroups = np.zeros((2, 200))
        self.Rgroups = np.zeros((2, 200))
        
        self.Sgroups[0,0], self.Sgroups[1,0] = self.N1 - 1, self.N2
        self.Igroups[0,0], self.Igroups[1,0] = 1, 0 #first infected in group 1

        
        
    def simulation(self):     
        #Stochastic Monte Carlo simulation of the SIR model

        t = 1 #time of the simulation
        newInum = [1]
        saveD = {}
        while t < self.tMax: 
            eventNum = [1,2] #array which defines the order of the events
            self.Sgroups[0, t], self.Sgroups[1, t] = self.Sgroups[0, t-1], self.Sgroups[1, t-1]  
            self.Igroups[0, t], self.Igroups[1, t] = self.Igroups[0, t-1], self.Igroups[1, t-1]
            self.Rgroups[0, t], self.Rgroups[1, t] = self.Rgroups[0, t-1], self.Rgroups[1, t-1]
            np.random.shuffle(eventNum) #the order of the events will be random
            #Infection
            newI = np.array([], int)
            
            for event in eventNum:
                if abs(event - 1) < 0.2:
                    #infection
                    #loop through each S - I link
                    for node in self.I_node:
                        for node2 in self.nei[node]: 
                            if node2 in self.S_node and node2 not in newI:    
                                #if a random number 0-1 is smaller than transmissibility 
                                #we move the node it from S to a temporary list newI            
                                if (np.random.random() < self.tau): 
                                    
                                
                                    newI = np.append(newI, node2) 
                                    tpl = np.where(self.S_node==node2)
                                    if len(tpl) != 1:
                                        sys.exit("error")
                                    inf = tpl[0][0]
                                    self.S_node = np.delete(self.S_node, inf) 
                                    #then we find its group
                                    if node2 < self.N1:
                                        self.Sgroups[0, t] -=  1
                                        self.Igroups[0, t] += 1
                                    
                                    else:
                                        self.Sgroups[1, t] -= 1
                                        self.Igroups[1, t] += 1
                                    
                
                
                #create a random number 0-1 for each I
                #if the random number is smaller than recovery rate
                #move the I node to R
                if abs(event - 2) < 0.2:
                    randList = np.random.random(len(self.I_node))
                    temp = np.where(randList < self.gamma)
                    nodes = self.I_node[temp[0]]
                    self.I_node = np.delete(self.I_node, temp[0])
                    self.R_node = np.append(self.R_node, nodes)
                    
                    #then we find its group
                    for ind in nodes:
                        if ind < self.N1:
                            self.Igroups[0, t] -= 1
                            self.Rgroups[0, t] += 1
                        else:
                            self.Igroups[1, t] -= 1
                            self.Rgroups[1, t] += 1
                
                
            self.I_node = np.append(self.I_node, newI)
            newInum.append(len(newI))
            self.I[t] = len(self.I_node)
            self.S[t] = len(self.S_node)             
            self.R[t] = len(self.R_node)
            
            print(t)        
            t += 1
        saveD = {'s1': self.Sgroups[0, :], 's2': self.Sgroups[1, :], 'i1': self.Igroups[0, :], 'i2': self.Igroups[1, :], 'r1': self.Rgroups[0, :], 'r2': self.Rgroups[1, :]}
        return [self.S, self.I, self.R, saveD]





    
    
    
    
    
    
    



