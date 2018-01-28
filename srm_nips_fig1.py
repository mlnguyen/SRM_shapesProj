
# This is a test script that does SRM on Sherlock data in PCC, following the 
# method described in Exp 1 of the NIPS paper. In brief, Sherlock subs are split
# in half, the SRM is trained on the first half of the data in each group, and
# then the second half is used for pattern similarity. I compare this to 
# pattern similarity without SRM, and the SRM improves similarity across ROIs.

#%% import useful stuff
import scipy.io
import numpy as np
import brainiak.funcalign.srm
import scipy.stats as stats
import matplotlib.pyplot as plt
import random


#%% Helper functions
def calc_ptnSim(data, iters=5):

    #allocate ptnSim space
    ptnSim = np.empty(iters)
    
    # get dims of data
    dims = data.shape

    for i in range(iters):
        
        # split data into two groups
        nsubs = data.shape[2]
        subs = list(range(nsubs))
        random.shuffle(subs)
        
        group1 = subs[0:round(nsubs/2)]
        group2 = subs[round(nsubs/2):nsubs]
        
        # average across subjects within group
        group1_avg = data[:,:,group1].mean(axis=2)
        group2_avg = data[:,:,group2].mean(axis=2)

        # correlate each column with each other
        r,p = stats.spearmanr(group1_avg, group2_avg)
        ptnSim[i] = np.diagonal(r[0:dims[1], dims[1]:dims[1]*2]).mean()
        
        print('iteration ' + str(i+1) + ': r = '+ str(ptnSim[i]))

    
    return ptnSim


#%% Benchmark: normal pattern sim

data = scipy.io.loadmat('/Users/Mai/Projects/shapesProject/data/from_matlab/srm_test/PCC.mat')
roidata = data['PCC']

# zscore data across time 
roidata_z = np.empty(roidata.shape)
for i in range(roidata.shape[2]):
    subdata = stats.zscore(roidata[:,:,i], axis=1, ddof=1)
    roidata_z[:,:,i] = subdata


r_orig = calc_ptnSim(roidata_z[:, 988:1976,:])

#%% SRM pattern sim

dims = roidata_z.shape

# split data into two groups
nsubs = dims[2]
subs = list(range(nsubs))
random.shuffle(subs)        
group1 = subs[0:round(nsubs/2)]
group2 = subs[round(nsubs/2):nsubs]

#split each group data into training data and test data
group1_train = roidata_z[:, 0:round(dims[1]/2), group1]
group1_test = roidata_z[:, round(dims[1]/2):dims[1], group1]

group2_train = roidata_z[:, 0:round(dims[1]/2), group2]
group2_test = roidata_z[:, round(dims[1]/2):dims[1], group2]

# transpose data so it is nSubs x nVox x nTRs
group1_train = np.transpose(group1_train, axes = [2,0,1])
group1_test = np.transpose(group1_test, axes = [2,0,1])

group2_train = np.transpose(group2_train, axes = [2,0,1])
group2_test = np.transpose(group2_test, axes = [2,0,1])

# set up srm
srm1 = brainiak.funcalign.srm.SRM(n_iter=10, features=50)
srm2 = brainiak.funcalign.srm.SRM(n_iter=10, features=50)

# fit srm on training data
srm1.fit(group1_train)
srm2.fit(group2_train)

# Register SRM's to each other. First get registration matrix Q
Am = srm2.s_.dot(srm1.s_.T)
pert = np.zeros((Am.shape)) 
np.fill_diagonal(pert,1)
Uq, sq, Vqt = np.linalg.svd(Am+0.001*pert,full_matrices=False)
Q = Uq.dot(Vqt)

# fit data to SRM
group1_shared = np.array(srm1.transform(group1_test))
group2_shared = np.array(srm2.transform(group2_test))

# register W from srm1 to W from srm2
w1 = np.array(srm1.w_)
nsubs = w1.shape[0]
nvox = w1.shape[1]
nfeat = w1.shape[2]
ntrs = group1_train.shape[2]
#
# register shared responses to one another
group1_shared_reg = np.empty(group1_shared.shape)
for s in range(nsubs):
    group1_shared_reg[s,:,:] = Q.dot(group1_shared[s,:,:])

## avg across subs
group1_shared_avg = group1_shared.mean(axis=0)
group2_shared_avg = group2_shared.mean(axis=0)
group1_shared_reg_avg = group1_shared_reg.mean(axis=0)
    
## correlate
trs = group1_shared.shape[2]
r,p = stats.spearmanr(group1_shared_reg_avg, group2_shared_avg)
ptnSim = np.diagonal(r[0:trs, trs:trs*2]).mean()


#%%
x_share = np.empty([nsubs, nvox, ntrs])
resid = np.empty([nsubs, nvox, ntrs])
w = np.array(srm1.w_)
for s in range(nsubs):
    x_share[s,:,:] = w[s,:,:].dot(np.array(srm1.s_))
    resid[s,:,:] = group1_test[s,:,:] - x_share[s,:,:]


print('ptnSim in original test data')
ptnSim_orig = calc_ptnSim(np.transpose(group1_test, axes=[2,1,0]))

print('ptnSim in shared SRM data in voxel space')
ptnSim_x_share = calc_ptnSim(np.transpose(x_share, axes=[2,1,0]))

print('ptnsim in residuals in voxel space')
resid = group1_test - x_share
ptnSim_resid = calc_ptnSim(np.transpose(resid, axes=[2,1,0]))









