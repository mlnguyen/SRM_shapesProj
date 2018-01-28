#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script does SRM, followed by pattern similarity, ISC, and RSA, on ROI data from the
# shapes movie project. Subjects in this project watched an animated shapes movie in the 
# style of Heider and Simmel. A subset of this subjects also watched a clip of the movie
# Catch Me If You Can. The goal of these analyses is to test if SRM improves pattern
# similarity or temporal similarity (ISC) for the shapes data. It also tests whether SRM
# improves RSA comparing neural similarity during the shapes movie to behavioral similarity.
# See https://www.biorxiv.org/content/early/2017/12/08/231019 for details, though SRM
# wasn't ultimately used in these analyses.
#
# SRM types:
#	1. shapesResid: fit SRM on shapes data, project back into subject space. Subtract 
#			shared response from subject response and do analyses on residuals. Theoretically
#			should enhance individual differences
#	2. catch2shapes: fit SRM on CMIYC, project shapes data into shared space. Do analysis 
#			in shared space.
#	3. shapesShared: fit SRM on shapes data, do analysis in shared space. Double dipping.
#
# Analyses:
#	1. ptn: pattern similarity following Chen et al. 2016
#	2. time: standard ISC
#	3. RSA: neural similarity (ptn or time) correlated with behavioral similarity as
#			measured by LSA


#%% Set up
import scipy.stats as stats
import numpy as np
import os
import utils
import brainiak.funcalign.srm
import matplotlib.pyplot as plt
import csv

smooth = 'smooth_20mm'
roi_list = ['A1', 'precun']
#roi_list = ['A1', 'V1', 'ltpj', 'rtpj', 'precun', 'mpfc', 'dlpfc', 'dmn']
datadir = '/Users/Mai/Projects/data/'+ smooth + '/roidata_all'
savedir = '/Users/Mai/Projects/analysis/rsa/' + smooth

# stats?
do_stats = ['time', 'rsa']

# features
k_list = [5, 10, 15, 20, 25, 50, 75, 100, 150]

# srm type
srm_type = 'shapesResid' #'catch2shapes' #, 'shapesShared', 'shapesResid']

#%% Load data based on srm type
if srm_type == 'catch2shapes':
    data = np.load(datadir + '/catch_shapes_n14.npz')
    train_data = data['catch_data'][()]
    test_data = data['shapes_data'][()]
    
    baseline_file = 'rsa_noSRM_n14.npy'
    lsafile = '/Users/Mai/Projects/data/lsa/lsa_grp1_kept.csv'
    nsubs = 14
    do_sim_types = ['ptn']    

elif srm_type == 'shapesShared' or srm_type == 'shapesResid':
    data = np.load(datadir + '/shapesMovie_n18.npz')
    train_data = data['shapes_data'][()]
    test_data = data['shapes_data'][()]
    
    baseline_file = 'rsa_noSRM_all.npy'
    lsafile = '/Users/Mai/Projects/data/lsa/lsa_grp1_all.csv'
    nsubs = 18
    do_sim_types = ['ptn', 'time']
    
    
# load lsa
lsa_file = open(lsafile)
lsa_reader = csv.reader(lsa_file)
lsa_mat = np.full([nsubs, nsubs], np.nan)
for i, row in enumerate(lsa_reader):
    lsa_mat[i,:] = row[1:]


#%% Initialize stuff

nfeats = len(k_list)
nrois = len(roi_list)

if 'ptn' in do_sim_types:
    ptn_sim = np.full([nrois, nfeats, nsubs, nsubs], np.nan)
    ptn_sim_avg = np.full([nrois, nfeats,  nsubs], np.nan)
    ptn_rsa = np.full([nrois, nfeats], np.nan)
    
    if 'ptn' in do_stats:
        ptn_sim_avg_p = np.full([nrois, nfeats], np.nan)
    if 'rsa' in do_stats:
        ptn_rsa_p = np.full([nrois, nfeats], np.nan)

if 'time' in do_sim_types:
    time_sim = np.full([nrois, nfeats, nsubs, nsubs], np.nan)
    time_sim_avg = np.full([nrois, nfeats,  nsubs], np.nan)
    time_rsa = np.full([nrois, nfeats], np.nan)

    if 'time' in do_stats:
        time_sim_avg_p = np.full([nrois, nfeats], np.nan)
    if 'rsa' in do_stats:
        time_rsa_p = np.full([nrois, nfeats], np.nan)

#%% Analysis

for roiN,roi in enumerate(roi_list): 
    
    for featN, feat in enumerate(k_list):        
        print('**** doing rsa in ' + roi + ' *****')
        print('number feats = ' + str(feat))
        
        #------ SRM -------------------------------------------------
        # get roidata & clean
        train = train_data[roi]
        test = test_data[roi]
        good_inds = utils.find_goodVox(train,test)
        train_good = train[:,good_inds,:]
        test_good = test[:,good_inds,:]
        
        # train srm on train data
        srm = brainiak.funcalign.srm.SRM(n_iter=10, features=feat)
        srm.fit(train_good)        

        # transform test data as appropriate
        if srm_type == 'catch2shapes':
            data = np.array(srm.transform(test_good))
       
        elif srm_type == 'shapesResid' or srm_type == 'shapesShared':
            data = np.empty(train_good.shape)
            resid = np.empty(train_good.shape)
            for s in range(nsubs):
                data[s,:,:] = srm.w_[s].dot(np.array(srm.s_))
                resid[s,:,:] = train_good[s,:,:] - data[s,:,:]
            
            if srm_type == 'shapesResid':
                data = resid
        else:
            print('invalid srm type')
            break;
        
        #------ Spatial sim ---------------------------------
        if 'ptn' in do_sim_types:
            ptn_sim[roiN,featN,:,:] = utils.calc_ptnSim(data, kind='pairwise')
            ptn_sim_avg[roiN,featN,:] = utils.calc_ptnSim(data, kind='avg_others')
            if 'ptn' in do_stats:
                ptn_sim_avg_p[roiN, featN], null = utils.stats_ptnSim_permute(data, 
                                                          ptn_sim_avg[roiN, featN,:].mean(), 
                                                          niters = 1000)

        #------ Temporal sim ---------------------------------      
        if 'time' in do_sim_types:
            time_sim[roiN,featN,:,:] = utils.calc_tempSim(data, kind='pairwise')
            time_sim_avg[roiN,featN,:] = utils.calc_tempSim(data, kind='avg_others')
            if 'time' in do_stats:
                time_sim_avg_p[roiN, featN], null = utils.stats_timeSim_permute(data, 
                                                          time_sim_avg[roiN, featN,:].mean(), 
                                                          niters = 1000)


        #------ RSA ---------------------------------      
        lsa_keep = utils.get_lowerTri(lsa_mat, -1)
        
        if 'ptn' in do_sim_types:
            ptn_keep = utils.get_lowerTri(ptn_sim[roiN,featN,:,:], -1)
            ptn_rsa[roiN, featN], p = stats.spearmanr(lsa_keep, ptn_keep)
            if 'rsa' in do_stats:
                ptn_rsa_p[roiN, featN], null = utils.stats_rsa_permute(lsa_mat, 
                                                                 ptn_sim[roiN, featN,:,:], 
                                                                 stat = ptn_rsa[roiN, featN], 
                                                                 keep='lowerTri') 
        if 'time' in do_sim_types:
            time_keep = utils.get_lowerTri(time_sim[roiN,featN,:,:], -1)            
            time_rsa[roiN, featN], p = stats.spearmanr(lsa_keep, time_keep)
            if 'rsa' in do_stats:
                time_rsa_p[roiN, featN], null = utils.stats_rsa_permute(lsa_mat, 
                                                                 time_sim[roiN, featN,:,:], 
                                                                 stat = ptn_rsa[roiN, featN], 
                                                                 keep='lowerTri') 

#%% Save

temp = {}

if 'ptn' in do_sim_types:
    temp['ptn_sim'] = ptn_sim 
    temp['ptn_sim_avgOthers'] = ptn_sim_avg 
    temp['ptn_rsa'] = ptn_rsa 
    
    if 'ptn' in do_stats:
        temp['ptn_sim_avg_p'] = ptn_sim_avg_p
    if 'rsa' in do_stats:
        temp['ptn_rsa_p'] = ptn_rsa_p


if 'time' in do_sim_types:
    temp['time_sim'] = time_sim
    temp['time_sim_avgOthers'] = time_sim_avg
    temp['time_rsa'] = time_rsa

    if 'time' in do_stats:
        temp['time_sim_avgOthers_p'] = time_sim_avg_p
    if 'rsa' in do_stats:
        temp['time_rsa_p'] = time_rsa_p

savename = os.path.join(savedir, 'rsa_srm_' + srm_type + '_n' + str(nsubs))
np.save(savename, temp)


#%% Plotting


# plot params
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '.3']
x = [160, 165, 170, 175, 180, 185, 190, 195, 200, 205]
size = 7; size_sig = 15

# load data
no_srm = np.load(os.path.join(savedir, baseline_file))[()]
data = temp

# Plot ptn sim: all ks
if 'ptn' in do_sim_types:
    for i in range(len(roi_list)):
       
        # plot SRM results. Large markers for sig
        plt.plot(k_list, data['ptn_sim_avgOthers'][i,:,:].mean(axis=1), color = colors[i], ls = '-', 
                 marker = '.', markersize = size, label = roi_list[i])    
    
        if 'ptn' in do_stats:
            for j in range(len(k_list)):
                pcrit, sig = utils.fdr_correct(data['ptn_sim_avgOthers_p'][:,j])
                if sig[i] == True:
                    plt.plot(k_list[j], data['ptn_sim_avgOthers'][i,j].mean(), color = colors[i],
                             marker = '.', markersize = size_sig)
        
        # plot baseline
        pcrit, sig = utils.fdr_correct(no_srm['ptn_sim_avgOthers_p'])
    
        if sig[i] == True:
            plt.plot(x[i], no_srm['ptn_sim_avgOthers'][i,:].mean(), color = colors[i], 
                         marker = '.', markersize = size_sig)
        else:
            plt.plot(x[i], no_srm['ptn_sim_avgOthers'][i,:].mean(), color = colors[i], 
                         marker = '.', markersize = size)            
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Pattern similarity with SRM (' + srm_type + ')')
    plt.show()

# Plot time sim: all ks
if 'time' in do_sim_types:
    for i in range(len(roi_list)):
       
        # plot temporal sim with SRM
        plt.plot(k_list, data['time_sim_avgOthers'][i,:,:].mean(axis=1), color = colors[i], ls = '-', 
                 marker = '.', markersize = size, label = roi_list[i])    
        
        if 'time' in do_stats:
            for j in range(len(k_list)):
                pcrit, sig = utils.fdr_correct(data['time_sim_avgOthers_p'][:,j])
                if sig[i] == True:
                    plt.plot(k_list[j], data['time_sim_avgOthers'][i,j].mean(), color = colors[i],
                             marker = '.', markersize = size_sig)
        
        # plot original
        pcrit, sig = utils.fdr_correct(no_srm['time_sim_avgOthers_p'])
        if sig[i] == True:
            plt.plot(x[i], no_srm['time_sim_avgOthers'][i,:].mean(), color = colors[i], 
                     marker = '.', markersize = size_sig)
        else:
            plt.plot(x[i], no_srm['time_sim_avgOthers'][i,:].mean(), color = colors[i], 
                     marker = '.', markersize = size)                     
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Temporal similarity with SRM (' + srm_type + ')')
    plt.show()


# plot spatial RSA
if 'ptn' in do_sim_types: 
    for i in range(len(roi_list)):
        
        # plot RSA with SRM
        plt.plot(k_list, data['ptn_rsa'][i,:], color = colors[i], ls = '-', 
                 marker = '.', markersize = size, label=roi_list[i])
        
        for j in range(len(k_list)):
            pcrit, sig = utils.fdr_correct(data['ptn_rsa_p'][:,j])
            if sig[i] == True:
                plt.plot(k_list[j], data['ptn_rsa'][i,j].mean(), color = colors[i],
                         marker = '.', markersize = size_sig)
        
        # plot orig data
        pcrit, sig = utils.fdr_correct(no_srm['ptn_rsa_p'])
        if sig[i] == True:
            plt.plot(x[i], no_srm['ptn_rsa'][i].mean(), color = colors[i], 
                     marker = '.', markersize = size_sig)
        else:
            plt.plot(x[i], no_srm['ptn_rsa'][i].mean(), color = colors[i], 
                     marker = '.', markersize = size)  
            
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('RSA over space')
    plt.show()


# plot temporal RSA
if 'time' in do_sim_types: 
    for i in range(len(roi_list)):
        
        # plot RSA with SRM
        plt.plot(k_list, data['time_rsa'][i,:], color = colors[i], ls = '-', 
                 marker = '.', markersize = size, label=roi_list[i])
        
        for j in range(len(sig)):
            pcrit, sig = utils.fdr_correct(data['time_rsa_p'][:,j])
            if sig[i] == True:
                plt.plot(k_list[j], data['time_rsa'][i,j].mean(), color = colors[i],
                         marker = '.', markersize = size_sig)
        
        # plot orig data
        pcrit, sig = utils.fdr_correct(no_srm['time_rsa_p'])
        if sig[i] == True:
            plt.plot(x[i], no_srm['time_rsa'][i].mean(), color = colors[i], 
                     marker = '.', markersize = size_sig)
        else:
            plt.plot(x[i], no_srm['time_rsa'][i].mean(), color = colors[i], 
                     marker = '.', markersize = size)  
            
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('RSA over time')
    plt.show()












