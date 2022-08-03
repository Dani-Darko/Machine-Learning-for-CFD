#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the rest of the NeuralNetwork.py file, just to run the other with no problems.

@author: dani
"""

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim
 
# Neural Network module for the SVD
def NNSVD(hidden_sizes, D_in, D_out):
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
                    
            # Inputs to hidden layer linear transformation
            self.hidden1 = torch.nn.Linear(D_in, hidden_sizes)
            # Hidden layer 
            self.hidden2 = torch.nn.Linear(hidden_sizes, hidden_sizes)
            # hidden layer to hidden output
            self.output = torch.nn.Linear(hidden_sizes, D_out)
            # Standard weight parameters between -0.5 and 0.5
            self.hidden1.weight.data = torch.rand(hidden_sizes, D_in)*0.5
            self.hidden2.weight.data = torch.rand(hidden_sizes, hidden_sizes)*0.5
            
            # Define sigmoid activation
            self.sigmoid = torch.nn.Sigmoid()
            self.GELU = torch.nn.Tanh()
            
        def forward(self, x):
            # Pass the input tensor through each of our operations
            a1 = self.hidden1(x)
            a1 = self.sigmoid(a1)
            a1 = self.hidden2(a1)
            a1 = self.sigmoid(a1)
            g = self.output(a1)
            
            return g
        
    network = Network()
    return network

# Neural Network module for the DoF
def NNDoF(hidden_sizes, D_in, D_out):
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
                    
            # Inputs to hidden layer linear transformation
            self.hidden1 = torch.nn.Linear(D_in, hidden_sizes)
            # Hidden layer 
            self.hidden2 = torch.nn.Linear(hidden_sizes, hidden_sizes)
            # hidden layer to hidden output
            self.output = torch.nn.Linear(hidden_sizes, D_out)
            # Standard weight parameters between -0.5 and 0.5
            self.hidden1.weight.data = torch.rand(hidden_sizes, D_in)*0.5
            self.hidden2.weight.data = torch.rand(hidden_sizes, hidden_sizes)*0.5
            
            # Define sigmoid activation
            self.sigmoid = torch.nn.Sigmoid()
            self.GELU = torch.nn.Tanh()
            
        def forward(self, x):
            # Pass the input tensor through each of our operations
            a1 = self.hidden1(x)
            a1 = self.sigmoid(a1)
            a1 = self.hidden2(a1)
            a1 = self.sigmoid(a1)
            g = self.output(a1)
            
            return g
        
    network = Network()
    return network