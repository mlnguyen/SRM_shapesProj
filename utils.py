#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utils for shapes project SRM analysis (srm_shapesData.py)

import numpy as np
import scipy.stats as stats
import time
from scipy.fftpack import fft, ifft

#%% General utils ##############################

def find_goodVox(*args):
# args = tuple of datasets which dimensions nSubs x nVox x nTRs. Returns  
# boolean array fo dims nVox that indexes voxels that all subjects across all
# included datasets have.
        
    # initialize good voxel array
    good_vox = np.array([True] * args[0].shape[1])
    
    # find bad voxels in every dataset for every subject
    for dataSet in args:
        
        for sub in range(dataSet.shape[0]):
            # get subject data
            subdata = dataSet[sub,:,:]
    
            # find voxels that are nans
            nanVox = np.isnan(subdata[:,1])

            # find voxels that are empty
            emptyVox = np.where(~np.any(subdata, axis=1))        

            # set these voxels as bad in the good_vox array
            good_vox[nanVox] = False
            good_vox[emptyVox] = False
    
    # return vector of good voxels
    return good_vox


####
def get_lowerTri(mat, diag):
    ltri_inds = np.tril(np.ones(mat.shape), diag)
    data = mat[ltri_inds==True]
    return data


########
def print_results(roi_list, r, p, sig):
    
    for i, roi in enumerate(roi_list):
        r_str = str(round(r[i], 2))    
        p_str = str(round(p[i], 5))
    
        result = roi + ': r = ' + r_str + ', p = ' + str(p_str)   

        if sig[i]:
            result = result + '*'
    
        print(result)
        

####
def unmask_data(data, mask):
    
    # check if mask is flat. if not, flatten
    if mask.ndim == 2 or mask.ndim == 1:
        mask_flat = mask
    elif mask.ndim == 3:
        mask_flat = np.reshape(mask, mask.size, order='F')
    else:
        print('invalid mask dimension')
        return
    
    # check if number of good vox in mask = size of data
    if sum(mask_flat) != data.shape[0]:
        print('mask size is inconrrect')
        return
    
    # unmask data
    nvox = mask_flat.shape[0]
    ntrs = data.shape[1]
    data_unmasked = np.empty([nvox, ntrs])
    data_unmasked[mask_flat] = data
    
    return data_unmasked
    
    
    

######

def phaseScrambleTS(ts):
    """Returns a TS: original TS power is preserved; TS phase is shuffled."""
    # source: https://goo.gl/VyIsXC
    # ts is array with length time points

    fs = fft(ts)
    pow_fs = np.abs(fs) ** 2.
    phase_fs = np.angle(fs)
    phase_fsr = phase_fs.copy()
    if len(ts) % 2 == 0:
        phase_fsr_lh = phase_fsr[1:int(len(phase_fsr)/2)]
    else:
        phase_fsr_lh = phase_fsr[1:int(len(phase_fsr)/2 + .5)]
    
    
    np.random.shuffle(phase_fsr_lh)
    if len(ts) % 2 == 0:
        phase_fsr_rh = -phase_fsr_lh[::-1]
        phase_fsr = np.concatenate((np.array((phase_fsr[0],)), phase_fsr_lh,
                                    np.array((phase_fsr[len(phase_fsr)/2],)),
                                    phase_fsr_rh))
    else:
        phase_fsr_rh = -phase_fsr_lh[::-1]
        phase_fsr = np.concatenate((np.array((phase_fsr[0],)), phase_fsr_lh, phase_fsr_rh))
    fsrp = np.sqrt(pow_fs) * (np.cos(phase_fsr) + 1j * np.sin(phase_fsr))
    tsrp = ifft(fsrp)
    
    if not np.allclose(tsrp.imag, np.zeros(tsrp.shape)):
        max_imag = (np.abs(tsrp.imag)).max()
        imag_str = '\nNOTE: a non-negligible imaginary component was discarded.\n\tMax: {}'
        print(imag_str.format(max_imag))
    
    return tsrp.real



#%% Stats ######################################

def zscore(data, axis=1):
# args: data is nSubs x nVox x nTRs
    data_z = np.empty(data.shape)
    
    for i in range(data.shape[0]):
            subdata = stats.zscore(data[i,:,:], axis=axis, ddof=1)
            data_z[i,:,:] = subdata   
        
    return data_z


###
def stats_ptnSim_permute(data, stat, niters=1000):
# args: data is nSubs x nVox x ntRs. Zscored
    
    # reorder data
    data_swap = np.swapaxes(data, axis1=1, axis2=2)
    
    # initialize null
    nsubs = data_swap.shape[0]
    null = np.empty([nsubs, niters])

    
    for subN in range(nsubs):
        start = time.time()
        print('\ndoing permutations for sub' + str(subN+1) )
        
        # get subdata
        subdata = data_swap[subN,:,:] 

        # get average of others
        others = np.setdiff1d(list(range(nsubs)), subN)
        otherdata = np.nanmean(data_swap[others,:,:], axis=0)
                
        # shuffle and get null correlation
        for i in range(niters):
            if i%100 == 0:
                print('iteration ' + str(i))
            np.random.shuffle(otherdata)
            null[subN,i] = np.diagonal(corrmat_fast(subdata, otherdata)).mean()
        print('time elapsed: ' + str(time.time()-start))
    
    
    # get p value
    null_mean = null.mean(axis=0)   
    mu = null_mean.mean()
    var = null_mean.std()
    p = 1-stats.norm(mu,var).cdf(stat)
    
    return p, null_mean

###
def stats_timeSim_permute(data, stat, niters=1000):
    nsubs, nvox, ntrs = data.shape    
    r = np.empty([nsubs, niters])    
    data_mean = data.mean(axis=1)
    
    
    for sub in range(nsubs):
        print('scrambling sub ' + str(sub))
        
        # get subdata & average of others data
        subdata = data_mean[sub,:] 
        other_subs = np.setdiff1d(list(range(nsubs)), [sub])
        otherdata = np.nanmean(data_mean[other_subs,:], axis = 0)
    
        start = time.time()
        for i in range(niters):        
            # scramble otherdata
            scramdata = phaseScrambleTS(otherdata)
            
            # corr
            r[sub,i] = np.corrcoef(subdata, scramdata)[0,1]
    
        print('time elapsed: ' + str(time.time()-start))
        
    
    # get p value
    null = r.mean(axis=0)
    mu = null.mean()
    var = null.std()
    p = 1-stats.norm(mu,var).cdf(stat)
        
    return p, null


####
def stats_rsa_permute(mat1, mat2, stat, keep='lowerTri', iters=1000):

    null = np.empty([iters,1])
    shuff = np.array(range(len(mat1)))
    
    for i in range(iters):
        # shuffle order
        np.random.shuffle(shuff)
        
        # shuffle mat1
        shuf_mat1 = mat1[shuff, :]
        shuf_mat1 = shuf_mat1[:,shuff]        
        
        # get data keep
        if keep == 'lowerTri':             
            shuf_mat1_keep = get_lowerTri(shuf_mat1, -1)
            mat2_keep = get_lowerTri(mat2, -1)
        elif keep == 'full':
            shuf_mat1_keep = shuf_mat1.reshape(shuf_mat1.size)
            mat2_keep = mat2.reshape(mat2.size)
        else:
            print('invalid keep string')
            return
            
        #corr
        r,p = stats.spearmanr(shuf_mat1_keep, mat2_keep)
        null[i] = r
        
    # get p value
    mu = null.mean()
    var = null.std()
    p = 1-stats.norm(mu,var).cdf(stat)
    
    return p, null


####
def fdr_correct(pvals, q=.05):

    # sort p values
    pvals_sort = np.sort(pvals, kind='quicksort')

    # get threshold for each p value
    m = len(pvals)
    temp = np.divide(np.arange(1,len(pvals)+1), m)
    thresh = np.multiply(temp, q)

    # find significant pvalues
    pvals_sig = pvals_sort[pvals_sort <= thresh]
    
    # if there are significant p values, return thresh. Otherwise NaN
    if pvals_sig.size > 0:
        pcrit = pvals_sig[-1]
        inds_sig = pvals <= pcrit
    else:
        pcrit = np.nan
        inds_sig = [False] * len(pvals)
    
    return pcrit, inds_sig

#######
def corr_fast(mat1, mat2, norm=True):
# Vectorized method for correlating rows of one matrix with another. 
# Return array of correlations nRow x 1. If mats are already zscored, use norm
# = false. Otherwise, will calculate zscore, which slows fx considerably.

    # zscore
    if norm == True:
        mat1_z = stats.zscore(mat1, axis=1, ddof=1)
        mat2_z = stats.zscore(mat2, axis=1, ddof=1)
    else:
        mat1_z = mat1 
        mat2_z = mat2 
    
    # Calculate sum of products
    sum_vec = np.nansum(mat1_z.T* mat2_z.T, axis = 0)
                        
    # Calculate degrees of free
    dof = mat1_z.shape[1] - 1
    
    # Correlation coeff
    r = sum_vec/dof        
    return r

###
def corrmat_fast(A,B):
    # source: goo.gl/sNl8FTcontent_copyCopy short URL
    # vectorized method for calculating cormat between rows of two matrices

    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))


# %% Analyses #################################

def calc_ptnSim(data, kind='avg_others'):
# args: data is nSubs x nVox x ntRs. Zscored
    
    nsubs = data.shape[0]
    ntrs = data.shape[2]
    
    # initialize ptnsim
    if kind == 'avg_others':
        ptnSim = np.full(nsubs, np.nan)
    elif kind == 'pairwise':
        ptnSim = np.full([nsubs, nsubs], np.nan)
    else:
        print('invalid ptnsim type')
        return
    
    # pattern similarity    
    for subN in range(nsubs):   
        
        # get subdata
        subdata = data[subN,:,:]   
    
        # compare each subject to average of other subjects
        if kind == 'avg_others':                
            # get average of others
            others = np.setdiff1d(list(range(nsubs)), subN)
            otherdata = np.nanmean(data[others,:,:], axis=0)      
        
            # get pattern sim
            r = np.corrcoef(subdata.T,otherdata.T)
            cormat = r[0:ntrs, ntrs:ntrs*2]
            ptnSim[subN] = np.diagonal(cormat).mean()
        
        # compare each subject to every other subject
        elif kind == 'pairwise':          
            for subN2 in range(nsubs):
                
                # get other data
                otherdata = data[subN2,:,:]
                
                # get pattern sim
                cormat = corrmat_fast(subdata, otherdata)
                ptnSim[subN, subN2] = np.diagonal(cormat).mean()

    return ptnSim


####
def calc_tempSim(data, kind='avg_others'):
    
    nsubs = data.shape[0]
    
    # initialize tempSim
    if kind == 'avg_others':
        tempSim = np.full(nsubs, np.nan)
    elif kind == 'pairwise':
        tempSim = np.full([nsubs, nsubs], np.nan)
    else:
        print('invalid temporal sim type')
        return
    
    # average voxels 
    data_mean = data.mean(axis=1)
    
    # temporal similarity
    for subN in range(nsubs):
       
        # get subdata
        subdata = data_mean[subN,:]  

        # compare each subject with average of other subjects
        if kind == 'avg_others':
            # get average of others
            other_subs = np.setdiff1d(list(range(nsubs)), [subN])
            otherdata = np.nanmean(data_mean[other_subs,:], axis = 0)

            # correlations
            r = np.corrcoef(subdata, otherdata)
            tempSim[subN] = r[0,1]
            
        elif kind == 'pairwise':
            for subN2 in range(nsubs):
                otherdata = data_mean[subN2,:]
                r = np.corrcoef(subdata, otherdata)
                tempSim[subN, subN2] = r[0,1]

    return tempSim


####
def calc_isc(data1, data2=[], kind='within'):
# Calculates one-to-average others ISC. Data1 = nSubs x nVox x nTRs.

    r_subs = np.empty(data1.shape[0:2])
    
    # within group ISC
    if kind == 'within':
        print('calculating isc within group...')
        for i in range(len(data1)):
            
            # get subdata
            subdata = data1[i,:,:]
        
            # get average of others
            other_subs = np.setdiff1d(list(range(len(data1))), [i])
            otherdata = np.nanmean(data1[other_subs,:], axis = 0)
            
            # correlation
            r_subs[i,:] = corr_fast(subdata, otherdata)

    #between group isc
    elif kind == 'between':
        print('between group isc')
    
    # error
    else:
        print('invalid isc type')
        return
    
    # average across subjects
    isc = r_subs.mean(axis=0)
    
    return isc, r_subs


