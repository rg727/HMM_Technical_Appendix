# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:06:28 2023

@author: rg727
"""
from SALib.sample import latin
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as ss
from random import random
from functions import fitmodel




#Create larger selection (100*105 years) of synthetic traces for users: 
    
zero_data = np.zeros(shape=(105,100))
total_samples = pd.DataFrame(zero_data)
    
# Number of years for alternative trace
n_years = 105

# Import historical data that it used to fit HMM model 
AnnualQ_h = pd.read_csv('uc_historical.csv')

# Fit the model and pull out relevant parameters and samples 
logQ = np.log(AnnualQ_h)


#Stationary Synthetic Sample Size=100: 

for i in range(100):
    hidden_states, mus, sigmas, P, logProb, samples, model = fitmodel.fitHMM(logQ, n_years)
    total_samples.iloc[:,i]=np.array(samples[0])

total_samples.to_csv("synthetic_stationary_small_sample_100.csv",index=False)



#Stationary Synthetic Sample Size=1000:
zero_data = np.zeros(shape=(105,1000))
total_samples_large = pd.DataFrame(zero_data)
    
# Number of years for alternative trace
n_years = 105

# Import historical data that it used to fit HMM model 
AnnualQ_h = pd.read_csv('uc_historical.csv')

# Fit the model and pull out relevant parameters and samples 
logQ = np.log(AnnualQ_h)


for i in range(1000):
    hidden_states, mus, sigmas, P, logProb, samples, model = fitmodel.fitHMM(logQ, n_years)
    total_samples_large.iloc[:,i]=np.array(samples[0])

total_samples_large.to_csv("synthetic_stationary_large_sample_1000.csv",index=False)


#Non-Stationary Generator Sample Size=1000:
# Create problem structure with parameters that we want to sample 
problem = {
    'num_vars': 6,
    'names': ['wet_mu', 'dry_mu', 'wet_std','dry_std','dry_tp',"wet_tp"],
    'bounds': [[0.98, 1.02],
               [0.98, 1.02],
               [0.75,1.25],
               [0.75,1.25],
               [-0.3,0.3],
               [-0.3,0.3]]
}

# generate 1000 parameterizations
n_samples = 10000

# set random seed for reproducibility
seed_value = 123

# Generate our samples 
LHsamples = latin.sample(problem, n_samples, seed_value)

# Number of years for alternative trace
n_years = 105

# Import historical data that it used to fit HMM model 
AnnualQ_h = pd.read_csv('uc_historical.csv')

zero_data = np.zeros(shape=(105,10000))
total_samples_large_nonstationary = pd.DataFrame(zero_data)

for y in range(10000):
    
    # Create empty arrays to store the new Gaussian HMM parameters for each SOW
    Pnew = np.empty([2, 2])
    piNew = np.empty([2])
    musNew_HMM = np.empty([2])
    sigmasNew_HMM = np.empty([2])
    logAnnualQ_s = np.empty([n_years]) 
    
    # Calculate new transition matrix and stationary distribution of SOW at last node
    # as well as new means and standard deviations

    Pnew[0, 0] = max(0.0,min(1.0, P[0, 0] + LHsamples[y][4]))
    Pnew[1, 1] = max(0.0,min(1.0, P[1, 1] + LHsamples[y][5]))
    Pnew[0, 1] = 1 - Pnew[0, 0]
    Pnew[1, 0] = 1 - Pnew[1, 1]
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(Pnew))
    one_eigval = np.argmin(np.abs(eigenvals - 1))
    piNew = np.divide(np.dot(np.transpose(Pnew), eigenvecs[:, one_eigval]),
                      np.sum(np.dot(np.transpose(Pnew), eigenvecs[:, one_eigval])))

    musNew_HMM[0] = mus[0] * LHsamples[y][1]
    musNew_HMM[1] = mus[1] * LHsamples[y][0]
    sigmasNew_HMM[0] = sigmas[0] * LHsamples[y][3]
    sigmasNew_HMM[1] = sigmas[1] * LHsamples[y][2]

    # Generate first state and log-space annual flow at last node
    states = np.empty([n_years])
    if random() <= piNew[0]:
        states[0] = 0
        logAnnualQ_s[0] = ss.norm.rvs(musNew_HMM[0], sigmasNew_HMM[0])
    else:
        states[0] = 1
        logAnnualQ_s[0] = ss.norm.rvs(musNew_HMM[1], sigmasNew_HMM[1])

    # generate remaining state trajectory and log space flows at last node
    for j in range(1, n_years):
        if random() <= Pnew[int(states[j-1]), int(states[j-1])]:
            states[j] = states[j-1]
        else:
            states[j] = 1 - states[j-1]

        if states[j] == 0:
            logAnnualQ_s[j] = ss.norm.rvs(musNew_HMM[0], sigmasNew_HMM[0])
        else:
            logAnnualQ_s[j] = ss.norm.rvs(musNew_HMM[1], sigmasNew_HMM[1])

    # Convert log-space flows to real-space flows
    total_samples_large_nonstationary.iloc[:,y]=logAnnualQ_s

total_samples_large_nonstationary.to_csv("synthetic_nonstationry_large_sample_10000.csv",index=False)
    



#Plot FDCs (Stationary vs. Historic) 
for i in range(1000):
        if i==0:
           sample=total_samples_large_nonstationary.iloc[:,i]
           sample = np.sort(sample)
           exceedence = np.arange(1.,len(sample)+1) / (len(sample) +1)
           plt.plot(exceedence, sample,color="#005F73", label="Synthetic Non-Stationary (large sample)") 
        else:  
            sample=total_samples_large_nonstationary.iloc[:,i]
            sample = np.sort(sample)
            exceedence = np.arange(1.,len(sample)+1) / (len(sample) +1)
            plt.plot(exceedence, sample,color="#005F73")
            

for i in range(1000):
        if i==0:
           sample=total_samples_large.iloc[:,i]
           sample = np.sort(sample)
           exceedence = np.arange(1.,len(sample)+1) / (len(sample) +1)
           plt.plot(exceedence, sample,color="#183A2E", label="Synthetic Stationary (large sample)") 
        else:  
            sample=total_samples_large.iloc[:,i]
            sample = np.sort(sample)
            exceedence = np.arange(1.,len(sample)+1) / (len(sample) +1)
            plt.plot(exceedence, sample,color="#183A2E")
                
    
sort = np.sort(np.log(AnnualQ_h.iloc[:,0]))
exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
plt.plot(exceedence, sort,color="#EFE2BE",label="Historical")        
plt.xlabel("Non-Exceedance Probability",fontsize=14)
plt.ylabel('Log Annual Flow ($m^3$)',fontsize=14)
plt.legend(fontsize = 10)
plt.savefig("nonstationary_synthetic_FDC.png", format="png", dpi=300,bbox_inches='tight')




#Plot FDCs (Stationary vs. NonStationary vs. Historic)   
for i in range(1000):
        if i==0:
           sample=total_samples_large.iloc[:,i]
           sample = np.sort(sample)
           exceedence = np.arange(1.,len(sample)+1) / (len(sample) +1)
           plt.plot(exceedence, sample,color="#183A2E", label="Synthetic Stationary (large sample)") 
        else:  
            sample=total_samples_large.iloc[:,i]
            sample = np.sort(sample)
            exceedence = np.arange(1.,len(sample)+1) / (len(sample) +1)
            plt.plot(exceedence, sample,color="#183A2E")

                

 
for i in range(100):
        if i==0:
            sample=total_samples.iloc[:,i]
            sample = np.sort(sample)
            exceedence = np.arange(1.,len(sample)+1) / (len(sample) +1)
            plt.plot(exceedence, sample,color="#848E76",label="Synthetic Stationary (small sample)")
        else:
            sample=total_samples.iloc[:,i]
            sample = np.sort(sample)
            exceedence = np.arange(1.,len(sample)+1) / (len(sample) +1)
            plt.plot(exceedence, sample,color="#848E76")
    
sort = np.sort(np.log(AnnualQ_h.iloc[:,0]))
exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
plt.plot(exceedence, sort,color="#EFE2BE",label="Historical")        
plt.xlabel("Non-Exceedance Probability",fontsize=14)
plt.ylabel('Log Annual Flow ($m^3$)',fontsize=14)
plt.legend(fontsize = 10)
plt.savefig("stationary_synthetic_FDC.png", format="png", dpi=300,bbox_inches='tight')

