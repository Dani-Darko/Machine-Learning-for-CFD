"""
1. PredictionWL.py takes the data from a random set of points to predict
    with all of the trained machine learning algorithms the quantities of
    interest. It does so directly with QoI(), but indirectly through the
    data points with the rest of the models.

@author: Daniela Segura
"""
# Neural networks libraries
import torch

# Numpy libraries for array and matrix operations
import numpy as np
from torch import optim

from joblib import load

# Call for all the subfunctions needed from the rest of the scripts
import NNmodules

def PredsDoF(newx, stanBCname, hidden_sizes, D_in, D_out, xsd, xmean, std, mean):
    network = NNmodules.NNDoF(hidden_sizes, D_in, D_out)
    optimizer = optim.Rprop(network.parameters())
    checkpoint_path='../SVDdata/'+stanBCname+'/DoF-outData-NN.pt'
    checkpoint= torch.load(checkpoint_path)
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    network.eval()
    preds = network(newx.float())
    preds = preds.detach().numpy()


    for mj in range(0,len(preds[0,:]),1):
        for tri in range(0,len(preds),1):
            preds[tri,mj] = preds[tri,mj] * std + mean
    
    return preds

def PredsSVD(newx, stanBCname, hidden_sizes, D_in, D_out, xsd, xmean, std, mean):
    network = NNmodules.NNSVD(hidden_sizes, D_in, D_out)
    optimizer = optim.Rprop(network.parameters())
    checkpoint_path='../SVDdata/'+stanBCname+'/SVD-outData-NN.pt'
    checkpoint= torch.load(checkpoint_path)
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    network.eval()
    Tr = network(torch.from_numpy(np.float32(newx))).detach().numpy()
    preds = np.empty(Tr.shape)
    # For SVD NN, transform the matrix back to the variable's dimension
    Trsd = np.loadtxt('../SVDdata/'+stanBCname+'/TrstdmeanData.txt')[0]
    Trmean = np.loadtxt('../SVDdata/'+stanBCname+'/TrstdmeanData.txt')[1]
    for mj in range(0,D_out,1):
        for tri in range(0,len(Tr),1):
            preds[tri,mj] = (Tr[tri,mj] * Trsd+Trmean)
    VTBC = torch.from_numpy(np.loadtxt('../SVDdata/'+stanBCname+'/VT_Data.csv', delimiter=","))
    preds = preds.dot(VTBC)
    
    for mj in range(0,len(preds[0,:]),1):
        for tri in range(0,len(preds),1):
            preds[tri,mj] = preds[tri,mj] * std + mean
    
     # Apreds = yhat.dot(VT)
     # PredsSigma = np.linalg.lstsq(U, yhat)
     # PredsTr = PredsSigma[0]
     
    return preds

def PredsQoI(newx, QoIname, hidden_sizes, D_in, D_out, xmean, xsd):
    network = NNmodules.NNSVD(hidden_sizes, D_in, D_out)
    optimizer = optim.Rprop(network.parameters())
    checkpoint_path='../SVDdata/QoI/QoI-'+QoIname+'-NN.pt'
    checkpoint= torch.load(checkpoint_path)
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    network.eval()
    predsQ = network(newx.float())
    predsQ = predsQ.detach().numpy()
    
    Qstd = np.loadtxt('../SVDdata/QoI/Q-'+QoIname+'-stdmeanData.txt')[0]
    Qmean = np.loadtxt('../SVDdata/QoI/Q-'+QoIname+'-stdmeanData.txt')[1]
    for i in range(0,len(predsQ),1):
        predsQ[i] = (predsQ[i] * Qstd+Qmean)
    
    Q0 = predsQ[0]
    preds = predsQ/Q0
    return predsQ, preds

def PredsRBF(newx, stanBCname, xsd, xmean, modes, std, mean):
    # Call for the interpolation parameters for the Variable of Interest
    rbfi1 = load("../SVDdata/"+stanBCname+"/RBF-data.joblib")
    Tr = rbfi1(torch.from_numpy(np.float32(newx)))
    preds = np.empty(Tr.shape)
    # For SVD NN, transform the matrix back to the variable's dimension
    Trsd = np.loadtxt('../SVDdata/'+stanBCname+f'/TrstdmeanData.txt')[0]
    Trmean = np.loadtxt('../SVDdata/'+stanBCname+f'/TrstdmeanData.txt')[1]
    for mj in range(0,modes,1):
        for tri in range(0,len(Tr),1):
            preds[tri,mj] = (Tr[tri,mj] * Trsd+Trmean)
    VTBC = torch.from_numpy(np.loadtxt('../SVDdata/'+stanBCname+'/VT_Data.csv', delimiter=","))
    preds = preds.dot(VTBC)
    
    for mj in range(0,len(preds[0,:]),1):
        for tri in range(0,len(preds),1):
            preds[tri,mj] = preds[tri,mj] * std + mean
    
    return preds

def PredsKriging(newx, stanBCname, xsd, xmean, modes, std, mean):
    # Call for the interpolation parameters for the Variable of Interest
    gpr = load("../SVDdata/"+stanBCname+"/GPR-data.joblib")
    preds, Tr_std = gpr.predict(newx, return_std=True)
    return preds