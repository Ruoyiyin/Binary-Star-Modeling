import os, shutil, sys, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.table import Table,join
from scipy import stats
from scipy.optimize import curve_fit
import itertools
from astropy.io import fits
#from gcreduce import gcutil 
from gcwork import starset
from gcwork import starTables
from gcwork import objects
from gcwork.polyfit import accel
from gcwork import realmedian
from gcwork import orbits
#from mpldatacursor import datacursor
import pdb
import emcee
import corner
import pymultinest
#from microlens.jlu import model_fitter
import json

#star = 'S0-67'
#star = 'S2-79'
#star = 'S1-44'
#star = 'S2-271'
#star = 'S3-303'

#star, period = 'S0-27', 10
#star, period = 'S3-237', 9

# real observed data
#root = "/g/lu/scratch/siyao/work/2_align/18_09_26/"
#t = Table.read(root+'points_3_c/'+star+'.points', format='ascii')
#time = t['col1']
#x_obs = t['col2']*-1.
#y_obs = t['col3']
#xe = t['col4']
#ye = t['col5']

## prediction from align
#tlin = Table.read(root + 'polyfit_4_trim/linear.txt', format='ascii')
#idx = np.where(tlin['name']==star)[0]
#tlin = tlin[idx]
#tacc = Table.read(root + 'polyfit_4_trim/accel.txt', format='ascii')
#idx = np.where(tacc['name']==star)[0]
#tacc= tacc[idx]

# calculate the acceleration upper limit at each projected distance
#cc = objects.Constants()

# keep record of the statistics number
#p = open('summary_{0}.txt'.format(star), 'a')
#p.close()

def pair_posterior(table, weights, outfile=None, title=None):
    """
    from: /u/mwhosek/Desktop/code/python/StarClusters/star_clusters/general_imf.py
    plot_posteriors1D
    Produces a matrix of plots. On the diagonals are the marginal
    posteriors of the parameters. On the off-diagonals are the
    marginal pairwise posteriors of the parameters.
    
    Parameters:
    -----------
    table: astropy table
        Contains 1 column for each parameter with samples.

    weights: array
        Weighting of each multinest result

    """
    params = list(table.keys())
    pcnt = len(params)

    fontsize = 10

    plt.figure(figsize = (50,50))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    # Margenilized 1D
    for ii in range(pcnt):
        ax = plt.subplot(pcnt, pcnt, ii*(pcnt+1)+1)
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
        n, bins, patch = plt.hist(table[params[ii]], normed=True,
                                 histtype='step', weights=weights, bins=50)
        plt.xlabel(params[ii], size=fontsize)
        plt.ylim(0, n.max()*1.1)

    # Bivariates
    for ii in range(pcnt - 1):
        for jj in range(ii+1, pcnt):
            ax = plt.subplot(pcnt, pcnt, ii*pcnt + jj+1)
            plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize)
            (H, x, y) = np.histogram2d(table[params[jj]], table[params[ii]],
                                       weights=weights, bins=50)
            xcenter = x[:-1] + (np.diff(x) / 2.0)
            ycenter = y[:-1] + (np.diff(y) / 2.0)
            
            plt.contourf(xcenter, ycenter, H.T, cmap=plt.cm.gist_yarg)

            plt.xlabel(params[jj], size=fontsize)
            plt.ylabel(params[ii], size=fontsize)
    if title != None:
        plt.suptitle(title)
    if outfile != None:
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()
    return


######################################
## Linear: vx, vy, x0, y0
######################################

def linear_multinest(star, prefix = 'try1', n_iter = 100, n_live = 1000, et = 0.3, run=True):
    root = "/g/lu/scratch/siyao/work/2_align/18_09_26/"
    t = Table.read(root+'points_3_c/'+star+'.points', format='ascii')
    time = t['col1']
    x_obs = t['col2']*-1.
    y_obs = t['col3']
    xe = t['col4']
    ye = t['col5']

## prediction from align
    tlin = Table.read(root + 'polyfit_4_trim/linear.txt', format='ascii')
    idx = np.where(tlin['name']==star)[0]
    tlin = tlin[idx]
    #tacc = Table.read(root + 'polyfit_4_trim/accel.txt', format='ascii')
    #idx = np.where(tacc['name']==star)[0]
    #tacc= tacc[idx]

# calculate the acceleration upper limit at each projected distance
    cc = objects.Constants()

# keep record of the statistics number
    p = open('summary_{0}.txt'.format(star), 'a')
    p.close()
    table_now_read = Table.read('linear_acc_comb_final.txt', format='ascii')
    def ln_prior(cube, ndim, nparams):
        cube[0] = cube[0]*tlin['vx']*4*1000. - tlin['vx']*1000.      # vx:    in mas/yr 
        cube[1] = cube[1]*tlin['vy']*4*1000. - tlin['vy']*1000.      # vy:    in mas/yr 
        cube[2] = cube[2]*tlin['x0']*4 - tlin['x0']                  # x0:    in arcsec 
        cube[3] = cube[3]*tlin['y0']*4 - tlin['y0']                  # y0:    in arcsec 
        return
    
    def ln_likelyhood(cube, ndim, nparams):
        # define model params
        vx = cube[0] * 0.001
        vy = cube[1] * 0.001
        x0 = cube[2]
        y0 = cube[3]
    
        # predicted x and y
        tf = 1990
        x = vx * (time-tf) + x0 
        y = vy * (time-tf) + y0
        ll = -0.5 * np.sum(((x_obs-x)/xe)**2 - np.log(1/xe**2)) - 0.5 * np.sum(((y_obs-y)/ye - np.log(1/ye**2))**2) 
        return ll
    
    parameters = ["vx", "vy", "x0", "y0"]
    n_params = len(parameters)

    # run the fitter 
    if run:
        pymultinest.run(ln_likelyhood, ln_prior, n_params, outputfiles_basename=star+ '_' + prefix+'_', resume = False, verbose = True, n_iter_before_update=n_iter,
                n_live_points=n_live, evidence_tolerance=et)
        json.dump(parameters, open(star + '_' + prefix + '_params.json', 'w')) # save parameter names
    a = pymultinest.Analyzer(outputfiles_basename=star + '_' + prefix + '_', n_params = n_params)
    print(a.get_best_fit())
    best_fit =  a.get_best_fit()
    
    ## make the plot
    ## plot the distribution of a posteriori possible models
    tab = Table.read( './'+ star + '_' + prefix + '_.txt', format='ascii')
    tab.rename_column('col1', 'weights')
    tab.rename_column('col2', 'logLike')
    tab.rename_column('col3', 'vx')
    tab.rename_column('col4', 'vy')
    tab.rename_column('col5', 'x0')
    tab.rename_column('col6', 'y0')
    logLike = tab['logLike']
    weights = tab['weights']
    tab.remove_columns(('weights', 'logLike'))
    a = np.array(tab['vx'])
    b = np.array(tab['vy'])
    c = np.array(tab['x0'])
    d = np.array(tab['y0'])
    e = np.array([a,b,c,d]).T
    fig = corner.corner(e, labels = ['vx', 'vy', 'x0', 'y0'], label_kwargs={'fontsize' : 15},quantiles = [0.16, 0.84], 
                                                    show_titles = True, title_kwargs={'fontsize':15})
    fig.savefig('bic2/' + star + '_'+ prefix+'_corner.png')
    #pair_posterior(tab, weights, outfile='./plot_{0}_{1}_linear_corner.png'.format(star, prefix))

    # print optimal solution and error
    cdf = np.cumsum(weights)
    max_idx = weights.argmax()
    mu_idx = np.where( abs(cdf - 0.5) == min(abs(cdf - 0.5)))[0][0]
    sig_low_idx = np.where( abs(cdf - (0.5 - 0.34135)) == min(abs(cdf - (0.5 - 0.34135))))[0][0]
    sig_high_idx = np.where( abs(cdf - (0.5 + 0.34135)) == min(abs(cdf - (0.5 + 0.34135))))[0][0]
    maxi = tab[max_idx]
    mu = tab[mu_idx]
    sig_low = tab[sig_low_idx]
    sig_high = tab[sig_high_idx]
    for i,s in enumerate(parameters):
        print('{0}: {1:.3f}+/-{2:.3f}'.format(s, maxi[s],(sig_high[s]-sig_low[s])/2 ))
    
    
    # print statistical numbers
    idx = logLike.argmin()
    ll = logLike[idx] / -2.
    BIC = np.log(len(x_obs)*2) * n_params - 2*ll
    AIC = 2*n_params - 2*ll
    idx_n = np.where(table_now_read['name']==star)[0]
    table_now_read['BIC_L'][idx_n] = BIC
    table_now_read['AIC_L'][idx_n] = AIC
    table_now_read['loglike_L'][idx_n] = ll
    #table_now_read['pos_weight_L'][idx_n] = weights[idx]
    #t['BIC_L'][idx]
    #idx = np.where(tlin['name']==star)[0]
    table_now_read['pos_weight_L'][idx_n] = weights[idx]
    table_now_read.write('linear_acc_ll_corrected.txt.txt',format='ascii', fast_writer=False) 
    print("Linear - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]))

    p = open('summary_{0}.txt'.format(star), 'a')
    p.write("Linear - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}\n".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]))
    p.close()
 
    # model smooth data
    time_array = np.linspace(time.min(), time.max(), 1000)
    vx, vy, x0, y0 = best_fit['parameters']
    vx = vx * 0.001        # arcsec/yr 
    vy = vy * 0.001
    tf = 1990
    x_model = vx * (time_array-tf) + x0 
    y_model = vy * (time_array-tf) + y0
    
    # plot the fit
    plt.figure(figsize=(10,10))
    plt.clf()
    plt.errorbar(x_obs, y_obs, xerr=xe, yerr=ye,  fmt='o', color='b', label='data')
    plt.plot(x_model, y_model, 'r--', label='fit_model')
    plt.xlabel('X offset from SgrA* (arcsec)')
    plt.ylabel('Y offset from SgrA* (arcsec)')
    plt.legend(loc='upper right')
    plt.annotate("Linear - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, \nloglikelihood={5:.3f}, posterior(weight) = {6}".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]), (0.02, 0.9), xycoords='axes fraction', color='r')
    plt.annotate('fit params: vx={0[0]:.2f}, vy={0[1]:.2f}, x0={0[2]:.1f}, y0={0[3]:.1f}'.format(best_fit['parameters']), 
            (0.02, 0.8), xycoords='axes fraction', color='r')
    plt.tight_layout()
    plt.savefig('bic2/plot_{0}_{1}_linear_niter{2}_nlive{3}_et{4}.png'.format(star, prefix, n_iter, n_live, et))
    plt.close()
    return 

#################################
### Accel: ax, ay, vx, vy, x0, y0
#################################

def accel_multinest(star, prefix = 'try2', n_iter = 100, n_live = 1000, et = 0.3, run=True):
    root = "/g/lu/scratch/siyao/work/2_align/18_09_26/"
    t = Table.read(root+'points_3_c/'+star+'.points', format='ascii')
    time = t['col1']
    x_obs = t['col2']*-1.
    y_obs = t['col3']
    xe = t['col4']
    ye = t['col5']

## prediction from align
    tlin = Table.read(root + 'polyfit_4_trim/linear.txt', format='ascii')
    idx = np.where(tlin['name']==star)[0]
    tlin = tlin[idx]
    tacc = Table.read(root + 'polyfit_4_trim/accel.txt', format='ascii')
    idx = np.where(tacc['name']==star)[0]
    tacc= tacc[idx]

# calculate the acceleration upper limit at each projected distance
    cc = objects.Constants()

# keep record of the statistics number
    p = open('summary_{0}.txt'.format(star), 'a')
    p.close()
    table_now_read = Table.read('model23/model23_all.txt', format='ascii')
    def ln_prior(cube, ndim, nparams):
        cube[0] = cube[0]*tacc['ax']*4*1000. - tacc['ax']*1000.      # vx:    in mas/yr 
        cube[1] = cube[1]*tacc['ay']*4*1000. - tacc['ay']*1000.      # vy:    in mas/yr 
        cube[2] = cube[2]*tacc['vx']*6*1000. - 2*tacc['vx']*1000.      # vx:    in mas/yr 
        cube[3] = cube[3]*tacc['vy']*6*1000. - 2*tacc['vy']*1000.      # vy:    in mas/yr 
        cube[4] = cube[4]*tacc['x0']*4 - tacc['x0']                  # x0:    in arcsec 
        cube[5] = cube[5]*tacc['y0']*4 - tacc['y0']                  # y0:    in arcsec 
        return
    
    def ln_likelyhood(cube, ndim, nparams):
        # define model params
        ax = cube[0] * 0.001
        ay = cube[1] * 0.001
        vx = cube[2] * 0.001
        vy = cube[3] * 0.001
        x0 = cube[4]
        y0 = cube[5]
    
        # predicted x and y
        tf = 1990
        x = 0.5 * ax * (time-tf)**2 + vx * (time-tf) + x0 
        y = 0.5 * ay * (time-tf)**2 + vy * (time-tf) + y0
        ll = -np.sum(((x_obs-x)/xe)**2) - np.sum(((y_obs-y)/ye)**2) 
        return ll
    
    parameters = ["ax", "ay", "vx", "vy", "x0", "y0"]
    n_params = len(parameters)

    # run the fitter 
    if run:
        pymultinest.run(ln_likelyhood, ln_prior, n_params, outputfiles_basename=star+'_'+prefix+'_', resume = False, verbose = True, n_iter_before_update=n_iter,
                n_live_points=n_live, evidence_tolerance=et)
        json.dump(parameters, open(star+'_'+prefix + '_params.json', 'w')) # save parameter names
    a = pymultinest.Analyzer(outputfiles_basename=star + '_' + prefix + '_', n_params = n_params)
    print(a.get_best_fit())
    best_fit =  a.get_best_fit()
    
    # make the plot
    ## plot the distribution of a posteriori possible models
    tab = Table.read( './'+ star + '_' + prefix + '_.txt', format='ascii')
    tab.rename_column('col1', 'weights')
    tab.rename_column('col2', 'logLike')
    tab.rename_column('col3', 'ax')
    tab.rename_column('col4', 'ay')
    tab.rename_column('col5', 'vx')
    tab.rename_column('col6', 'vy')
    tab.rename_column('col7', 'x0')
    tab.rename_column('col8', 'y0')
    logLike = tab['logLike']
    weights = tab['weights']
    tab.remove_columns(('weights', 'logLike'))
    a = np.array(tab['ax'])
    b = np.array(tab['ay'])
    c = np.array(tab['vx'])
    d = np.array(tab['vy'])
    e = np.array(tab['x0'])
    f = np.array(tab['y0'])
    g = np.array([a,b,c,d,e,f]).T
    pair_posterior(tab, weights, outfile='model23/plot_{0}_{1}_accel_corner.png'.format(star, prefix))
    fig = corner.corner(g, labels = ['ax', 'ay','vx', 'vy', 'x0', 'y0'], label_kwargs={'fontsize' : 15},quantiles = [0.16,0.84], 
                                                    show_titles = True, title_kwargs={'fontsize':18})

#fig.subplots_adjust(right=1,top=1)
    fig.savefig('model23/' + star + '_'+ prefix+'_corner.png')
    #pair_posterior(tab, weights, outfile='./plot_{0}_{1}_accel_corner.png'.format(star, prefix))

    # print optimal solution and error
    cdf = np.cumsum(weights)
    max_idx = weights.argmax()
    mu_idx = np.where( abs(cdf - 0.5) == min(abs(cdf - 0.5)))[0][0]
    sig_low_idx = np.where( abs(cdf - (0.5 - 0.34135)) == min(abs(cdf - (0.5 - 0.34135))))[0][0]
    sig_high_idx = np.where( abs(cdf - (0.5 + 0.34135)) == min(abs(cdf - (0.5 + 0.34135))))[0][0]
    maxi = tab[max_idx]
    mu = tab[mu_idx]
    sig_low = tab[sig_low_idx]
    sig_high = tab[sig_high_idx]
    for i,s in enumerate(parameters):
        print('{0}: {1:.3f}+/-{2:.3f}'.format(s, maxi[s],(sig_high[s]-sig_low[s])/2 ))
 
    # print statistical numbers
    idx = logLike.argmin()
    ll = logLike[idx] / -2.
    BIC = np.log(len(x_obs)*2) * n_params - 2*ll
    AIC = 2*n_params - 2*ll
    idx_n = np.where(table_now_read['name'] == star)[0]
    table_now_read['BIC_2'][idx_n] = BIC
    table_now_read['AIC_2'][idx_n] = AIC
    table_now_read['loglike_2'][idx_n] = ll
    table_now_read['pos_weight_2'][idx_n] = weights[idx]
    table_now_read.write('model23/model23_all.txt', format = 'ascii', fast_writer=False)
    print("Accel - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]))

    p = open('summary_{0}.txt'.format(star), 'a')
    p.write("Accel - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}\n".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]))
    p.close()
 
    # model smooth data
    time_array = np.linspace(time.min(), time.max(), 1000)
    ax, ay, vx, vy, x0, y0 = best_fit['parameters']
    ax = ax * 0.001
    ay = ay * 0.001
    vx = vx * 0.001        # arcsec/yr 
    vy = vy * 0.001
    tf = 1990
    x_model = 0.5 * ax * (time_array-tf)**2 + vx * (time_array-tf) + x0 
    y_model = 0.5 * ay * (time_array-tf)**2 + vy * (time_array-tf) + y0
    
    # plot the fit
    plt.figure(figsize=(10,10))
    plt.clf()
    plt.errorbar(x_obs, y_obs, xerr=xe, yerr=ye,  fmt='o', color='b', label='data')
    plt.plot(x_model, y_model, 'r--', label='fit_model')
    plt.xlabel('X offset from SgrA* (arcsec)')
    plt.ylabel('Y offset from SgrA* (arcsec)')
    plt.legend(loc='upper right')
    plt.annotate("Accel - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, \n loglikelihood={5:.3f}, posterior(weight) = {6}\n".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]), (0.02, 0.9), xycoords='axes fraction', color='r')
    plt.annotate('fit params: ax={0[0]:.2f}, ay={0[1]:.2f}, vx={0[2]:.2f}, vy={0[3]:.2f}, x0={0[4]:.1f}, y0={0[5]:.1f}'.format(best_fit['parameters']), 
            (0.02, 0.8), xycoords='axes fraction', color='r')
    plt.tight_layout()
    plt.savefig('model23/plot_{0}_{1}_accel_niter{2}_nlive{3}_et{4}.png'.format(star, prefix, n_iter, n_live, et))
    plt.close()

    return 

#################################
### Accel_Ar: ar, vx, vy, x0, y0
#################################

def accel_ar_multinest(star, prefix = 'try3r', n_iter = 100, n_live = 1000, et = 0.3, run=True):
    root = "/g/lu/scratch/siyao/work/2_align/18_09_26/"
    t = Table.read(root+'points_3_c/'+star+'.points', format='ascii')
    time = t['col1']
    x_obs = t['col2']*-1.
    y_obs = t['col3']
    xe = t['col4']
    ye = t['col5']

## prediction from align
    tlin = Table.read(root + 'polyfit_4_trim/linear.txt', format='ascii')
    idx = np.where(tlin['name']==star)[0]
    tlin = tlin[idx]
    tacc = Table.read(root + 'polyfit_4_trim/accel.txt', format='ascii')
    idx = np.where(tacc['name']==star)[0]
    tacc= tacc[idx]

# calculate the acceleration upper limit at each projected distance
    cc = objects.Constants()

# keep record of the statistics number
    p = open('summary_{0}.txt'.format(star), 'a')
    p.close()
    table_now_read = Table.read('model23/model23_all.txt', format='ascii')
    def ln_prior(cube, ndim, nparams):
        r2d_cm = tacc['r'] * cc.dist * cc.cm_in_au # cm
        a2d = -cc.G * cc.mass * cc.msun / r2d_cm**2 # cm/s^2
        a2d_km = a2d*1e-5*cc.sec_in_yr #km/s/yr
        a2d = a2d_km/37726.*1000. #mas/yr^2
        cube[0] = cube[0] * a2d                                        # ar: in mas/yr^2
        cube[1] = cube[1]*tacc['vx']*6*1000. - 2*tacc['vx']*1000.      # vx:    in mas/yr 
        cube[2] = cube[2]*tacc['vy']*6*1000. - 2*tacc['vy']*1000.      # vy:    in mas/yr 
        cube[3] = cube[3]*tacc['x0']*4 - tacc['x0']                    # x0:    in arcsec 
        cube[4] = cube[4]*tacc['y0']*4 - tacc['y0']                    # y0:    in arcsec 
        return
    
    def ln_likelyhood(cube, ndim, nparams):
        # define model params
        ax = cube[0] * 0.001 * tacc['x0'] / np.hypot(tacc['x0'], tacc['y0']) 
        ay = cube[0] * 0.001 * tacc['y0'] / np.hypot(tacc['x0'], tacc['y0']) 
        vx = cube[1] * 0.001
        vy = cube[2] * 0.001
        x0 = cube[3]
        y0 = cube[4]
    
        # predicted x and y
        tf = 1990
        x = 0.5 * ax * (time-tf)**2 + vx * (time-tf) + x0 
        y = 0.5 * ay * (time-tf)**2 + vy * (time-tf) + y0
        ll = -np.sum(((x_obs-x)/xe)**2) - np.sum(((y_obs-y)/ye)**2) 
        return ll
    
    parameters = ["ar", "vx", "vy", "x0", "y0"]
    n_params = len(parameters)

    # run the fitter 
    if run:
        pymultinest.run(ln_likelyhood, ln_prior, n_params, outputfiles_basename=star+'_'+prefix+'_', resume = False, verbose = True, n_iter_before_update=n_iter,
                n_live_points=n_live, evidence_tolerance=et)
        json.dump(parameters, open(star+'_'+prefix + '_params.json', 'w')) # save parameter names
    a = pymultinest.Analyzer(outputfiles_basename=star + '_' + prefix + '_', n_params = n_params)
    print(a.get_best_fit())
    best_fit =  a.get_best_fit()
    
    # make the plot
    ## plot the distribution of a posteriori possible models
    tab = Table.read( './'+ star + '_' + prefix + '_.txt', format='ascii')
    tab.rename_column('col1', 'weights')
    tab.rename_column('col2', 'logLike')
    tab.rename_column('col3', 'ar')
    tab.rename_column('col4', 'vx')
    tab.rename_column('col5', 'vy')
    tab.rename_column('col6', 'x0')
    tab.rename_column('col7', 'y0')
    logLike = tab['logLike']
    weights = tab['weights']
    tab.remove_columns(('weights', 'logLike'))
    a = np.array(tab['ar'])
    b = np.array(tab['vx'])
    c = np.array(tab['vy'])
    d = np.array(tab['x0'])
    e = np.array(tab['y0'])
    #f = np.array(tab['y0'])
    f = np.array([a,b,c,d,e]).T
    fig = corner.corner(f, labels = ['ar','vx', 'vy', 'x0', 'y0'], label_kwargs={'fontsize' : 10},quantiles = [0.16, 0.84], 
                                                    show_titles = True, title_kwargs={'fontsize':18})
    pair_posterior(tab, weights, outfile='model23/plot_{0}_{1}_accel_ar_corner.png'.format(star, prefix))
    fig.savefig('model23/' + star + '_'+ prefix+'_corner.png')
    
    
    # print optimal solution and error
    cdf = np.cumsum(weights)
    max_idx = weights.argmax()
    mu_idx = np.where( abs(cdf - 0.5) == min(abs(cdf - 0.5)))[0][0]
    sig_low_idx = np.where( abs(cdf - (0.5 - 0.34135)) == min(abs(cdf - (0.5 - 0.34135))))[0][0]
    sig_high_idx = np.where( abs(cdf - (0.5 + 0.34135)) == min(abs(cdf - (0.5 + 0.34135))))[0][0]
    maxi = tab[max_idx]
    mu = tab[mu_idx]
    sig_low = tab[sig_low_idx]
    sig_high = tab[sig_high_idx]
    for i,s in enumerate(parameters):
        print('{0}: {1:.3f}+/-{2:.3f}'.format(s, maxi[s],(sig_high[s]-sig_low[s])/2 ))

    # print statistical numbers
    idx = logLike.argmin()
    ll = logLike[idx] / -2.
    BIC = np.log(len(x_obs)*2) * n_params - 2*ll
    AIC = 2*n_params - 2*ll
    idx_n = np.where(table_now_read['name'] == star)[0]
    table_now_read['BIC_3r'][idx_n] = BIC
    table_now_read['AIC_3r'][idx_n] = AIC
    table_now_read['loglike_3r'][idx_n] = ll
    table_now_read['pos_weight_3r'][idx_n] = weights[idx]
    table_now_read.write('model23/model23_all.txt', format = 'ascii', fast_writer=False)
    print("Accel_ar - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]))

    p = open('summary_{0}.txt'.format(star), 'a')
    p.write("Accel_ar - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}\n".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]))
    p.close()
 
    # model smooth data
    time_array = np.linspace(time.min(), time.max(), 1000)
    ar, vx, vy, x0, y0 = best_fit['parameters']
    ax = ar * 0.001 * tacc['x0'] / np.hypot(tacc['x0'], tacc['y0']) 
    ay = ar * 0.001 * tacc['y0'] / np.hypot(tacc['x0'], tacc['y0']) 
    vx = vx * 0.001        # arcsec/yr 
    vy = vy * 0.001
    tf = 1990
    x_model = 0.5 * ax * (time_array-tf)**2 + vx * (time_array-tf) + x0 
    y_model = 0.5 * ay * (time_array-tf)**2 + vy * (time_array-tf) + y0
    
    # plot the fit
    plt.figure(figsize=(10,10))
    plt.clf()
    plt.errorbar(x_obs, y_obs, xerr=xe, yerr=ye,  fmt='o', color='b', label='data')
    plt.plot(x_model, y_model, 'r--', label='fit_model')
    plt.xlabel('X offset from SgrA* (arcsec)')
    plt.ylabel('Y offset from SgrA* (arcsec)')
    plt.legend(loc='upper right')
    plt.annotate("Accel_ar - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, \nloglikelihood={5:.3f}, posterior(weight) = {6}".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]), (0.02, 0.9), xycoords='axes fraction', color='r')
    plt.annotate('fit params: ar={0[0]:.2f}, vx={0[1]:.2f}, vy={0[2]:.2f}, x0={0[3]:.1f}, y0={0[4]:.1f}'.format(best_fit['parameters']), 
            (0.02, 0.8), xycoords='axes fraction', color='r')
    plt.tight_layout()
    plt.savefig('model23/plot_{0}_{1}_accel_ar_niter{2}_nlive{3}_et{4}.png'.format(star, prefix, n_iter, n_live, et))
    plt.close()


    return 

#################################
### Linear + Binary: w, Omega, i, e, p, tp, aleph, vx, vy, x0, y0
#################################

def linear_binary_multinest(star, prefix = 'try3', n_iter = 100, n_live = 1000, et = 0.3, run=True):
    root = "/g/lu/scratch/siyao/work/2_align/18_09_26/"
    t = Table.read(root+'points_3_c/'+star+'.points', format='ascii')
    time = t['col1']
    x_obs = t['col2']*-1.
    y_obs = t['col3']
    xe = t['col4']
    ye = t['col5']

## prediction from align
    tlin = Table.read(root + 'polyfit_4_trim/linear.txt', format='ascii')
    idx = np.where(tlin['name']==star)[0]
    tlin = tlin[idx]
    tacc = Table.read(root + 'polyfit_4_trim/accel.txt', format='ascii')
    idx = np.where(tacc['name']==star)[0]
    tacc= tacc[idx]

# calculate the acceleration upper limit at each projected distance
    cc = objects.Constants()

# keep record of the statistics number
    p = open('summary_{0}.txt'.format(star), 'a')
    p.close()
    table_now_read = Table.read('linear_acc_comb_final.txt', format='ascii')
    def ln_prior(cube, ndim, nparams):
        cube[0] = cube[0] * 360                   # omega: in degree 
        cube[1] = cube[1] * 360                   # bigOm: in degree
        #cube[2] = cube[2] * 180                   # incl:  in degree
        cube[2] = np.degrees(np.arccos(1- cube[2] * 2))          # incl:  in degree, flat in sin(i)
        cube[3] = np.sqrt(cube[3])              # e: f(e) = e
        cube[4] = cube[4] * 50       # p:     in year 
        cube[5] = cube[5] * 20 + 2000             # tp:    in year 
        cube[6] = cube[6] * 5                     # aleph: in mas 

        ## prediction from align
        cube[7] = cube[7]*tlin['vx']*4*1000. - tlin['vx']*1000.      # vx:    in mas/yr 
        cube[8] = cube[8]*tlin['vy']*4*1000. - tlin['vy']*1000.      # vy:    in mas/yr 
        cube[9] = cube[9]*tlin['x0']*4 - tlin['x0']                  # x0:    in arcsec 
        cube[10] = cube[10]*tlin['y0']*4 - tlin['y0']                # y0:    in arcsec 

        return np.log(cube[3]) + np.log(np.sin(np.radians(cube[2])))
    
    def ln_likelyhood(cube, ndim, nparams):
        # chagne the loop
        cube[0]  = cube[0] % 360
        cube[1]  = cube[1] % 360
        cube[2]  = cube[2] % 180 
        cube[5] = (cube[5] - 2000) % 20 + 2000
    
        # define model params
        orb = orbits.Orbit()
        omega = cube[0]
        bigOm = cube[1]
        incl = cube[2]
        e_mag = cube[3]
        p = cube[4]
        tp = cube[5]
        aleph = cube[6]
        vx = cube[7]
        vy = cube[8]
        x0 = cube[9]
        y0 = cube[10]
        if e_mag<0 or e_mag>1:
            return -1e10000
        orb.w = omega              # degree
        orb.o = bigOm              # degree
        orb.i = incl               # degree
        orb.e = e_mag             
        orb.p = p                  # year
        orb.tp = tp                # the time of the periastron passage
        orb.aleph = aleph * 0.001  # semi-major axis of photocenter in arcsec
        orb.vx = vx * 0.001        # arcsec/yr 
        orb.vy = vy * 0.001
        orb.x0 = x0                # arcsec
        orb.y0 = y0
    
        # predicted x and y
        x, y = orb.oal2xy(time)
        ll = -0.5 * np.sum(((x_obs-x)/xe)**2 - np.log(1/xe**2)) - 0.5 * np.sum(((y_obs-y)/ye - np.log(1/ye**2))**2) 
        return ll
        
    
    parameters = ["w", "bigOm", "i", "e", "p", "tp", "aleph", "vx", "vy", "x0", "y0"]
    n_params = len(parameters)

    # run the fitter 
    if run:
        pymultinest.run(ln_likelyhood, ln_prior, n_params, outputfiles_basename=star+'_'+prefix+'_', resume = False, verbose = True, n_iter_before_update=n_iter,
                n_live_points=n_live, evidence_tolerance=et)
        json.dump(parameters, open(star+'_'+prefix + '_params.json', 'w')) # save parameter names
    a = pymultinest.Analyzer(outputfiles_basename=star+'_'+prefix+'_', n_params = n_params)
    print(a.get_best_fit())
    best_fit =  a.get_best_fit()
    
    ## make the plot
    # model smooth data
    time_array = np.linspace(time.min(), time.max(), 1000)
    orb = orbits.Orbit()
    omega, bigOm, incl, e_mag, p, tp, aleph, vx, vy, x0, y0 = best_fit['parameters']
    orb.w = omega              # degree
    orb.o = bigOm              # degree
    orb.i = incl               # degree
    orb.e = e_mag             
    orb.p = p                  # year
    orb.tp = tp                # the time of the periastron passage
    orb.aleph = aleph * 0.001  # semi-major axis of photocenter in arcsec
    orb.vx = vx * 0.001        # arcsec/yr 
    orb.vy = vy * 0.001
    orb.x0 = x0                # arcsec
    orb.y0 = y0
    x_model, y_model = orb.oal2xy(time_array)

    ## plot the distribution of a posteriori possible models
    tab = Table.read( './'+ star + '_' + prefix + '_.txt', format='ascii')
    tab.rename_column('col1', 'weights')
    tab.rename_column('col2', 'logLike')
    tab.rename_column('col3', 'w')
    tab.rename_column('col4', 'bigOm')
    tab.rename_column('col5', 'i')
    tab.rename_column('col6', 'e')
    tab.rename_column('col7', 'p')
    tab.rename_column('col8', 'tp')
    tab.rename_column('col9', 'aleph')
    tab.rename_column('col10', 'vx')
    tab.rename_column('col11', 'vy')
    tab.rename_column('col12', 'x0')
    tab.rename_column('col13', 'y0')
    weights = tab['weights']
    logLike = tab['logLike']
    tab.remove_columns(('weights', 'logLike'))
    a = np.array(tab['w'])
    b = np.array(tab['bigOm'])
    c = np.array(tab['i'])
    d = np.array(tab['e'])
    e = np.array(tab['p'])
    f = np.array(tab['tp'])
    g = np.array(tab['aleph'])
    h = np.array(tab['vx'])
    i = np.array(tab['vy'])
    j = np.array(tab['x0'])
    k = np.array(tab['y0'])
    l = np.array([a,b,c,d,e,f,g,h,i,j,k]).T
    fig = corner.corner(l, labels = ['w', 'bigOm','i','e','p','tp','aleph','vx', 'vy', 'x0', 'y0'], label_kwargs={'fontsize' : 22},quantiles = [0.16, 0.84], show_titles = True, title_kwargs={'fontsize':14})

#fig.subplots_adjust(right=1.5,top=1.5)
    fig.savefig('bic2/' + star + '_'+ prefix+'_corner_jupyter.png')
    pair_posterior(tab, weights, outfile='bic2/plot_{0}_{1}_linear_binary_corner.png'.format(star, prefix))
    
    # print optimal solution and error
    cdf = np.cumsum(weights)
    max_idx = weights.argmax()
    mu_idx = np.where( abs(cdf - 0.5) == min(abs(cdf - 0.5)))[0][0]
    sig_low_idx = np.where( abs(cdf - (0.5 - 0.34135)) == min(abs(cdf - (0.5 - 0.34135))))[0][0]
    sig_high_idx = np.where( abs(cdf - (0.5 + 0.34135)) == min(abs(cdf - (0.5 + 0.34135))))[0][0]
    maxi = tab[max_idx]
    mu = tab[mu_idx]
    sig_low = tab[sig_low_idx]
    sig_high = tab[sig_high_idx]
    for i,s in enumerate(parameters):
        print('{0}: {1:.3f} +/- {2:.3f}'.format(s, maxi[s], abs(sig_high[s]-sig_low[s])/2))


    # print statistical numbers
    idx = logLike.argmin()
    ll = logLike[idx] / -2.
    BIC = np.log(len(x_obs)*2) * n_params - 2*ll
    AIC = 2*n_params - 2*ll
    idx_n = np.where(table_now_read['name']==star)[0]
    table_now_read['BIC_LB'][idx_n] = BIC
    table_now_read['AIC_LB'][idx_n] = AIC
    table_now_read['loglike_LB'][idx_n] = ll
    #table_now_read['pos_weight_L'][idx_n] = weights[idx]
    #t['BIC_L'][idx]
    #idx = np.where(tlin['name']==star)[0]
    table_now_read['pos_weight_LB'][idx_n] = weights[idx]
    table_now_read.write('linear_acc_ll_corrected.txt',format='ascii', fast_writer=False) 
    print("Linear + Binary - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]))

    p = open('summary_{0}.txt'.format(star), 'a')
    p.write("Linear + Binary - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}\n".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]))
    p.close()
    
    ## plot the fit
    plt.figure(figsize=(10,10))
    plt.clf()
    plt.errorbar(x_obs, y_obs, xerr=xe, yerr=ye,  fmt='o', color='b', label='data')
    plt.plot(x_model, y_model, 'r--', label='fit_model')
    plt.xlabel('X offset from SgrA* (arcsec)')
    plt.ylabel('Y offset from SgrA* (arcsec)')
    plt.legend(loc='upper right')
    plt.annotate("Linear + Binary - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, \nloglikelihood={5:.3f}, posterior(weight) = {6}".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]), (0.02, 0.9), xycoords='axes fraction', color='r')
    plt.annotate('fit params: w={0[0]:.0f},bigOm={0[1]:.0f},i={0[2]:.0f}, e={0[3]:.2f}, p={0[4]:.2f},\n \
            tp={0[5]:.0f},semi_major axis={0[6]:.1f}, vx={0[7]:.2f}, vy={0[8]:.2f}, x0={0[9]:.1f}, y0={0[10]:.1f}'.format(best_fit['parameters']), 
            (0.02, 0.8), xycoords='axes fraction', color='r')
    plt.tight_layout()
    plt.savefig('bic2/plot_{0}_{1}_linear_binary_niter{2}_nlive{3}_et{4}.png'.format(star, prefix, n_iter, n_live, et))
    plt.close()

   
    return 

#################################
### Accel + Binary: w, Omega, i, e, p, tp, aleph, ax, ay, vx, vy, x0, y0
#################################

def accel_binary_multinest(star, prefix = 'try5', n_iter = 100, n_live = 1000, et = 0.3, run=True):
    root = "/g/lu/scratch/siyao/work/2_align/18_09_26/"
    t = Table.read(root+'points_3_c/'+star+'.points', format='ascii')
    time = t['col1']
    x_obs = t['col2']*-1.
    y_obs = t['col3']
    xe = t['col4']
    ye = t['col5']

## prediction from align
    tlin = Table.read(root + 'polyfit_4_trim/linear.txt', format='ascii')
    idx = np.where(tlin['name']==star)[0]
    tlin = tlin[idx]
    tacc = Table.read(root + 'polyfit_4_trim/accel.txt', format='ascii')
    idx = np.where(tacc['name']==star)[0]
    tacc= tacc[idx]

# calculate the acceleration upper limit at each projected distance
    cc = objects.Constants()
    period = 10
# keep record of the statistics number
    p = open('summary_{0}.txt'.format(star), 'a')
    p.close()
    table_now_read = Table.read('model56/model56_all.txt', format='ascii')
    def ln_prior(cube, ndim, nparams):
        cube[0] = cube[0] * 360                   # omega: in degree 
        cube[1] = cube[1] * 360                   # bigOm: in degree
        #cube[2] = cube[2] * 180                   # incl:  in degree
        cube[2] = np.degrees(np.arccos(1- cube[2] * 2))          # incl:  in degree, flat in sin(i)
        cube[3] = np.sqrt(cube[3])              # e: f(e) = e
        cube[4] = cube[4] * 4 + period - 2        # p:     in year 
        cube[5] = cube[5] * 20 + 2000             # tp:    in year 
        cube[6] = cube[6] * 5                     # aleph: in mas 

        ## prediction from align
        cube[7] = cube[7]*tacc['ax']*4*1000. - tacc['ax']*1000.         # ax:    in mas/yr^2 
        cube[8] = cube[8]*tacc['ay']*4*1000. - tacc['ay']*1000.         # ay:    in mas/yr^2 
        cube[9] = cube[9]*tacc['vx']*6*1000. - 2*tacc['vx']*1000.       # vx:    in mas/yr 
        cube[10] = cube[10]*tacc['vy']*6*1000. - 2*tacc['vy']*1000.     # vy:    in mas/yr 
        cube[11] = cube[11]*tacc['x0']*4 - tacc['x0']                   # x0:    in arcsec 
        cube[12] = cube[12]*tacc['y0']*4 - tacc['y0']                   # y0:    in arcsec 
        return np.log(cube[3]) + np.log(np.sin(np.radians(cube[2])))
    
    def ln_likelyhood(cube, ndim, nparams):
        # chagne the loop
        cube[0]  = cube[0] % 360
        cube[1]  = cube[1] % 360
        cube[2]  = cube[2] % 180
        cube[5] = (cube[5] - 2000) % 20  + 2000
    
        # define model params
        orb = orbits.Orbit()
        omega = cube[0]
        bigOm = cube[1]
        incl = cube[2]
        e_mag = cube[3]
        p = cube[4]
        tp = cube[5]
        aleph = cube[6]
        ax = cube[7]
        ay = cube[8]
        vx = cube[9]
        vy = cube[10]
        x0 = cube[11]
        y0 = cube[12]
        if e_mag<0 or e_mag>1:
            return -1e10000
        orb.w = omega              # degree
        orb.o = bigOm              # degree
        orb.i = incl               # degree
        orb.e = e_mag             
        orb.p = p                  # year
        orb.tp = tp                # the time of the periastron passage
        orb.aleph = aleph * 0.001  # semi-major axis of photocenter in arcsec
        orb.ax = ax * 0.001        # arcsec/yr^2
        orb.ay = ay * 0.001
        orb.vx = vx * 0.001        # arcsec/yr 
        orb.vy = vy * 0.001
        orb.x0 = x0                # arcsec
        orb.y0 = y0
    
        # predicted x and y
        x, y = orb.oal2xy(time, accel=True)
        ll = -np.sum(((x_obs-x)/xe)**2) - np.sum(((y_obs-y)/ye)**2) 
        return ll
        
    
    parameters = ["w", "bigOm", "i", "e", "p", "tp", "aleph", "ax", "ay", "vx", "vy", "x0", "y0"]
    n_params = len(parameters)

    # run the fitter 
    if run:
        pymultinest.run(ln_likelyhood, ln_prior, n_params, outputfiles_basename=star+'_'+prefix+'_', resume = False, verbose = True, n_iter_before_update=n_iter,
                n_live_points=n_live, evidence_tolerance=et)
        json.dump(parameters, open(star+'_'+prefix + '_params.json', 'w')) # save parameter names
    a = pymultinest.Analyzer(outputfiles_basename=star+'_'+prefix+'_', n_params = n_params)
    print(a.get_best_fit())
    best_fit =  a.get_best_fit()
    
    # make the plot
    ## plot the distribution of a posteriori possible models
    tab = Table.read( './'+ star + '_' + prefix + '_.txt', format='ascii')
    tab.rename_column('col1', 'weights')
    tab.rename_column('col2', 'logLike')
    tab.rename_column('col3', 'w')
    tab.rename_column('col4', 'bigOm')
    tab.rename_column('col5', 'i')
    tab.rename_column('col6', 'e')
    tab.rename_column('col7', 'p')
    tab.rename_column('col8', 'tp')
    tab.rename_column('col9', 'aleph')
    tab.rename_column('col10', 'ax')
    tab.rename_column('col11', 'ay')
    tab.rename_column('col12', 'vx')
    tab.rename_column('col13', 'vy')
    tab.rename_column('col14', 'x0')
    tab.rename_column('col15', 'y0')
    weights = tab['weights']
    logLike = tab['logLike']
    tab.remove_columns(('weights', 'logLike'))
    a = np.array(tab['w'])
    b = np.array(tab['bigOm'])
    c = np.array(tab['i'])
    d = np.array(tab['e'])
    e = np.array(tab['p'])
    f = np.array(tab['tp'])
    g = np.array(tab['aleph'])
    h = np.array(tab['ax'])
    i = np.array(tab['ay'])
    j = np.array(tab['vx'])
    k = np.array(tab['vy'])
    l = np.array(tab['x0'])
    m = np.array(tab['y0'])
    n = np.array([a,b,c,d,e,f,g,h,i,j,k,l,m]).T
    fig = corner.corner(n, labels = ['w', 'bigOm','i','e','p','tp','aleph','ax','ay','vx', 'vy', 'x0', 'y0'], label_kwargs={'fontsize' : 18},quantiles = [0.16, 0.84], show_titles = True, title_kwargs={'fontsize':14})
    fig.savefig('model56/' + star + '_'+ prefix+'_corner.png')
    pair_posterior(tab, weights, outfile='model56/plot_{0}_{1}_accel_binary_corner.png'.format(star, prefix))
    
    # print statistical numbers
    idx = logLike.argmin()
    ll = logLike[idx] / -2.
    BIC = np.log(len(x_obs)*2) * n_params - 2*ll
    AIC = 2*n_params - 2*ll
    idx_n = np.where(table_now_read['name']==star)[0]
    table_now_read['BIC_5'][idx_n] = BIC
    table_now_read['AIC_5'][idx_n] = AIC
    table_now_read['loglike_5'][idx_n] = ll
    #table_now_read['pos_weight_6'][idx_n] = weights[idx]
    #t['BIC_L'][idx]
    #idx = np.where(tlin['name']==star)[0]
    table_now_read['pos_weight_5'][idx_n] = weights[idx]
    table_now_read.write('model56/model56_all.txt',format='ascii', fast_writer=False) 
    print("Accel + Binary - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]))

    p = open('summary_{0}.txt'.format(star), 'a')
    p.write("Accel + Binary - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}\n".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]))
    p.close()
 
    # model smooth data
    time_array = np.linspace(time.min(), time.max(), 1000)
    orb = orbits.Orbit()
    omega, bigOm, incl, e_mag, p, tp, aleph, ax, ay, vx, vy, x0, y0 = best_fit['parameters']
    orb.w = omega              # degree
    orb.o = bigOm              # degree
    orb.i = incl               # degree
    orb.e = e_mag             
    orb.p = p                  # year
    orb.tp = tp                # the time of the periastron passage
    orb.aleph = aleph * 0.001  # semi-major axis of photocenter in arcsec
    orb.ax = ax * 0.001        # arcsec/yr^2
    orb.ay = ay * 0.001
    orb.vx = vx * 0.001        # arcsec/yr 
    orb.vy = vy * 0.001
    orb.x0 = x0                # arcsec
    orb.y0 = y0
    x_model, y_model = orb.oal2xy(time_array, accel=True)
    
    ## plot the fit
    plt.figure(figsize=(10,10))
    plt.clf()
    plt.errorbar(x_obs, y_obs, xerr=xe, yerr=ye,  fmt='o', color='b', label='data')
    plt.plot(x_model, y_model, 'r--', label='fit_model')
    plt.xlabel('X offset from SgrA* (arcsec)')
    plt.ylabel('Y offset from SgrA* (arcsec)')
    plt.legend(loc='upper right')
    plt.annotate("Accel + Binary - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, \nloglikelihood={5:.3f}, posterior(weight) = {6}".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]), (0.02, 0.9), xycoords='axes fraction', color='r')
    plt.annotate('fit params: w={0[0]:.0f},bigOm={0[1]:.0f},i={0[2]:.0f}, e={0[3]:.2f}, p={0[4]:.2f}\n, tp={0[5]:.0f},semi_major axis={0[6]:.1f}, \n \
            ax={0[7]:.2f}, ay={0[8]:.2f}, vx={0[9]:.2f}, vy={0[10]:.2f}, x0={0[11]:.1f}, y0={0[12]:.1f}'.format(best_fit['parameters']), 
            (0.02, 0.8), xycoords='axes fraction', color='r')
    plt.tight_layout()
    plt.savefig('model56/plot_{0}_{1}_accel_binary_niter{2}_nlive{3}_et{4}.png'.format(star, prefix, n_iter, n_live, et))
    plt.close()

    return 

#################################
### Accel_Ar + Binary: w, Omega, i, e, p, tp, aleph, ar, vx, vy, x0, y0
#################################

def accel_ar_binary_multinest(star, prefix = 'try6r', n_iter = 100, n_live = 1000, et = 0.3, run=True):
    root = "/g/lu/scratch/siyao/work/2_align/18_09_26/"
    t = Table.read(root+'points_3_c/'+star+'.points', format='ascii')
    time = t['col1']
    x_obs = t['col2']*-1.
    y_obs = t['col3']
    xe = t['col4']
    ye = t['col5']

## prediction from align
    tlin = Table.read(root + 'polyfit_4_trim/linear.txt', format='ascii')
    idx = np.where(tlin['name']==star)[0]
    tlin = tlin[idx]
    tacc = Table.read(root + 'polyfit_4_trim/accel.txt', format='ascii')
    idx = np.where(tacc['name']==star)[0]
    tacc= tacc[idx]

# calculate the acceleration upper limit at each projected distance
    cc = objects.Constants()
    period = 10
# keep record of the statistics number
    p = open('summary_{0}.txt'.format(star), 'a')
    p.close()
    table_now_read = Table.read('model56/model56_all.txt', format='ascii')
    def ln_prior(cube, ndim, nparams):
        cube[0] = cube[0] * 360                   # omega: in degree 
        cube[1] = cube[1] * 360                   # bigOm: in degree
        #cube[2] = cube[2] * 180                   # incl:  in degree
        cube[2] = np.degrees(np.arccos(1- cube[2] * 2))          # incl:  in degree, flat in sin(i)
        cube[3] = np.sqrt(cube[3])              # e: f(e) = e
        cube[4] = cube[4] * 4 + period - 2        # p:     in year 
        cube[5] = cube[5] * 20 + 2000             # tp:    in year 
        cube[6] = cube[6] * 5                     # aleph: in mas 

        ## prediction from align
        r2d_cm = tacc['r'] * cc.dist * cc.cm_in_au # cm
        a2d = -cc.G * cc.mass * cc.msun / r2d_cm**2 # cm/s^2
        a2d_km = a2d*1e-5*cc.sec_in_yr #km/s/yr
        a2d = a2d_km/37726.*1000. #mas/yr^2
        cube[7] = cube[7]*a2d                                          # ar:    in mas/yr^2 
        cube[8] = cube[8]*tacc['vx']*6*1000. - 2*tacc['vx']*1000.      # vx:    in mas/yr 
        cube[9] = cube[9]*tacc['vy']*6*1000. - 2*tacc['vy']*1000.      # vy:    in mas/yr 
        cube[10] = cube[10]*tacc['x0']*4 - tacc['x0']                  # x0:    in arcsec 
        cube[11] = cube[11]*tacc['y0']*4 - tacc['y0']                  # y0:    in arcsec 

        return np.log(cube[3]) + np.log(np.sin(np.radians(cube[2])))
    
    def ln_likelyhood(cube, ndim, nparams):
        # chagne the loop
        cube[0]  = cube[0] % 360
        cube[1]  = cube[1] % 360
        cube[2]  = cube[2] % 180
        cube[5] = (cube[5] - 2000) % 20  + 2000
    
        # define model params
        orb = orbits.Orbit()
        omega = cube[0]
        bigOm = cube[1]
        incl = cube[2]
        e_mag = cube[3]
        p = cube[4]
        tp = cube[5]
        aleph = cube[6]
        ar = cube[7]
        vx = cube[8]
        vy = cube[9]
        x0 = cube[10]
        y0 = cube[11]

        if e_mag<0 or e_mag>1:
            return -1e10000
        orb.w = omega              # degree
        orb.o = bigOm              # degree
        orb.i = incl               # degree
        orb.e = e_mag             
        orb.p = p                  # year
        orb.tp = tp                # the time of the periastron passage
        orb.aleph = aleph * 0.001  # semi-major axis of photocenter in arcsec
        orb.ax = ar * 0.001 * tacc['x0'] / np.hypot(tacc['x0'], tacc['y0'])  #arcsec/yr^2,,,, ar projects on x
        orb.ay = ar * 0.001 * tacc['y0'] / np.hypot(tacc['x0'], tacc['y0'])  # ar projects on y
        orb.vx = vx * 0.001        # arcsec/yr 
        orb.vy = vy * 0.001
        orb.x0 = x0                # arcsec
        orb.y0 = y0
    
        # predicted x and y
        x, y = orb.oal2xy(time, accel=True)
        ll = -np.sum(((x_obs-x)/xe)**2) - np.sum(((y_obs-y)/ye)**2) 
        return ll
        
    
    parameters = ["w", "bigOm", "i", "e", "p", "tp", "aleph", "ar", "vx", "vy", "x0", "y0"]
    n_params = len(parameters)

    # run the fitter 
    if run:
        pymultinest.run(ln_likelyhood, ln_prior, n_params, outputfiles_basename=star+'_'+prefix+'_', resume = False, verbose = True, n_iter_before_update=n_iter,
                n_live_points=n_live, evidence_tolerance=et)
        json.dump(parameters, open(star+'_'+prefix + '_params.json', 'w')) # save parameter names
    a = pymultinest.Analyzer(outputfiles_basename=star+'_'+prefix+'_', n_params = n_params)
    print(a.get_best_fit())
    best_fit =  a.get_best_fit()
    
    # make the plot
    # model smooth data
    time_array = np.linspace(time.min(), time.max(), 1000)
    orb = orbits.Orbit()
    omega, bigOm, incl, e_mag, p, tp, aleph, ar, vx, vy, x0, y0 = best_fit['parameters']
    orb.w = omega              # degree
    orb.o = bigOm              # degree
    orb.i = incl               # degree
    orb.e = e_mag             
    orb.p = p                  # year
    orb.tp = tp                # the time of the periastron passage
    orb.aleph = aleph * 0.001  # semi-major axis of photocenter in arcsec
    orb.ax = ar * 0.001 * tacc['x0'] / np.hypot(tacc['x0'], tacc['y0']) 
    orb.ay = ar * 0.001 * tacc['y0'] / np.hypot(tacc['x0'], tacc['y0']) 
    orb.vx = vx * 0.001        # arcsec/yr 
    orb.vy = vy * 0.001
    orb.x0 = x0                # arcsec
    orb.y0 = y0
    x_model, y_model = orb.oal2xy(time_array, accel=True)
    
    ## make the plot
    ## plot the distribution of a posteriori possible models
    tab = Table.read( './'+ star + '_' + prefix + '_.txt', format='ascii')
    tab.rename_column('col1', 'weights')
    tab.rename_column('col2', 'logLike')
    tab.rename_column('col3', 'w')
    tab.rename_column('col4', 'bigOm')
    tab.rename_column('col5', 'i')
    tab.rename_column('col6', 'e')
    tab.rename_column('col7', 'p')
    tab.rename_column('col8', 'tp')
    tab.rename_column('col9', 'aleph')
    tab.rename_column('col10', 'ar')
    tab.rename_column('col11', 'vx')
    tab.rename_column('col12', 'vy')
    tab.rename_column('col13', 'x0')
    tab.rename_column('col14', 'y0')
    weights = tab['weights']
    logLike = tab['logLike']
    tab.remove_columns(('weights', 'logLike'))
    a = np.array(tab['w'])
    b = np.array(tab['bigOm'])
    c = np.array(tab['i'])
    d = np.array(tab['e'])
    e = np.array(tab['p'])
    f = np.array(tab['tp'])
    g = np.array(tab['aleph'])
    h = np.array(tab['ar'])
    i = np.array(tab['vx'])
    j = np.array(tab['vy'])
    k = np.array(tab['x0'])
    l = np.array(tab['y0'])
    #m = np.array(tab['y0'])
    m = np.array([a,b,c,d,e,f,g,h,i,j,k,l]).T
    fig = corner.corner(m, labels = ['w', 'bigOm','i','e','p','tp','aleph','ar','vx', 'vy', 'x0', 'y0'], label_kwargs={'fontsize' : 17},quantiles = [0.16, 0.84], show_titles = True, title_kwargs={'fontsize':14})
    fig.savefig('model56/' + star + '_'+ prefix+'_corner.png')
    pair_posterior(tab, weights, outfile='model56/plot_{0}_{1}_accel_ar_binary_corner.png'.format(star, prefix))
 
    # print statistical numbers
    idx = logLike.argmin()
    ll = logLike[idx] / -2.
    BIC = np.log(len(x_obs)*2) * n_params - 2*ll
    AIC = 2*n_params - 2*ll
    idx_n = np.where(table_now_read['name']==star)[0]
    table_now_read['BIC_6'][idx_n] = BIC
    table_now_read['AIC_6'][idx_n] = AIC
    table_now_read['loglike_6'][idx_n] = ll
    #table_now_read['pos_weight_6'][idx_n] = weights[idx]
    #t['BIC_L'][idx]
    #idx = np.where(tlin['name']==star)[0]
    table_now_read['pos_weight_6'][idx_n] = weights[idx]
    table_now_read.write('model56/model56_all.txt',format='ascii', fast_writer=False) 
    print("Accel_Ar + Binary - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]))

    p = open('summary_{0}.txt'.format(star), 'a')
    p.write("Accel_Ar + Binary - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}\n".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]))
    p.close()
 
    # plot the fit
    plt.figure(figsize=(10,10))
    plt.clf()
    plt.errorbar(x_obs, y_obs, xerr=xe, yerr=ye,  fmt='o', color='b', label='data')
    plt.plot(x_model, y_model, 'r--', label='fit_model')
    plt.xlabel('X offset from SgrA* (arcsec)')
    plt.ylabel('Y offset from SgrA* (arcsec)')
    plt.legend(loc='upper right')
    plt.annotate("Accel_Ar + Binary - {0}: BIC={1:.2f}, AIC={2:.2f}, ndata={3}, nparams={4}, loglikelihood={5:.3f}, posterior(weight) = {6}".format(
        prefix, BIC, AIC, len(x_obs)*2, n_params, ll, weights[idx]), (0.02, 0.9), xycoords='axes fraction', color='r')
    plt.annotate('fit params: w={0[0]:.0f},bigOm={0[1]:.0f},i={0[2]:.0f}, e={0[3]:.2f}, p={0[4]:.2f}\n, \
            tp={0[5]:.0f},semi_major axis={0[6]:.1f}, ar={0[7]:.2f}\n, \
            vx={0[8]:.2f}, vy={0[9]:.2f}, x0={0[10]:.1f}, y0={0[11]:.1f}'.format(best_fit['parameters']), (0.02, 0.8), xycoords='axes fraction', color='r')
    plt.tight_layout()
    plt.savefig('model56/plot_{0}_{1}_accel_ar_binary_niter{2}_nlive{3}_et{4}.png'.format(star, prefix, n_iter, n_live, et))
    plt.close()
    return