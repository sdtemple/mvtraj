import pandas as pd
import numpy as np
import math
from random import uniform
from copy import deepcopy
import matplotlib.pyplot as plt

class Trajectory:
    ''' Multivariate longitudinal object for k-means clustering
    
    Parameters
    ----------
    identity: int
        A unique identifier
    cluster: int
        Unique cluster assignment
    dist: float
        Frechet distance linked to cluster assignment
    parameterization: list
        List of 2-tuples describing a curve parameterization
    scalar: float
        Multiplier to relate time with multivariate data
    times: array_like
        Univariate vector of time points
    longitudinal: array_like
        Multivariate longitudinal data
    '''
    
    def __init__(self, identity, longitudinal, times, scalar):
        self.identity = identity
        self.cluster = None
        self.dist = np.inf
        self.parameterization = None
        self.scalar = scalar
        self.times = times * scalar
        self.longitudinal = longitudinal

def lp_norm(x, y, xt, yt, p = 2):
    norm = np.power(x - y, p)
    norm = np.sum(norm)
    norm = norm + np.power(xt - yt, p)
    norm = np.power(norm, 1 / p)
    return norm

def zero_inflated_lp_norm(x, y, xt, yt, p = 2):
    ''' Lp norm ignoring zeros
    
    Parameters
    ----------
    x : array_like
        Multidimensional vector
    y: array_like
        Multidimensional vector
    xt: float
        A scaled time value
    yt: float
        A scaled time value
    p: float (int)
    
    Returns
    -------
    float
        A distance metric
    '''
    x0 = (x > 0).astype(int)
    y0 = (y > 0).astype(int)
    norm = np.power(x - y, p)
    norm = norm * x0 * y0 # ignore zero coordinates
    norm = np.sum(norm)
    norm = norm + np.power(xt - yt, p)
    norm = np.power(norm, 1 / p)
    return norm

def free_space(x, y, xt, yt, normfcn):
    ''' Calculate the free space diagram
    
    Parameters
    ----------
    x : array_like
        An array of longitudinal observations
    y : array_like
        An array of longitudinal observations
    xt: array_like
        Vector of scaled times
    yt: array_like
        Vector of scaled times
    normfcn: function
        A norm, e.g. Euclidean norm
    
    Returns
    -------
    array_like
        The free space diagram
    '''
    
    # initialize free space
    nrow = x.shape[0] + 1
    ncol = y.shape[0] +1
    freespace = np.zeros((nrow, ncol))
    
    # inf buffers
    for j in range(ncol):
        freespace[0][j] = np.inf
    for i in range(1, nrow):
        freespace[i][0] = np.inf
      
    # first row
    freespace[1][1] = normfcn(x[0], y[0], xt[0], yt[0])
    for j in range(2, ncol):
        a = freespace[1][j-1]
        b = normfcn(x[0], y[j-1], xt[0], yt[j-1])
        freespace[1][j] = max(a, b)
    
    # first column
    for i in range(2, nrow):
        a = freespace[i-1][1]
        c = normfcn(x[i-1], y[0], xt[i-1], yt[0])
        freespace[i][1] = max(a, c)

    # complete free space diagram
    for i in range(2, nrow):
        for j in range(2, ncol):
            a = freespace[i-1][j-1]
            b = freespace[i-1][j]
            c = freespace[i][j-1]
            d = normfcn(x[i-1], y[j-1], xt[i-1], yt[j-1])
            freespace[i][j] = max(d, min(a, b, c))
            
    return freespace

def frechet_dist(freespace):
    ''' Get Frechet distance from the free space diagram
    
    Parameters
    ----------
    freespace: array_like
        The free space diagram when calculating the Frechet distance
    
    Returns
    -------
    float
        The final cell in the free space diagram
    '''
    nrow = freespace.shape[0]
    ncol = freespace.shape[1]
    return freespace[nrow-1][ncol-1]

def backtrack(freespace):
    ''' Determine curve parameterizations linked to the Frechet distance
    
    Parameters
    ----------
    freespace: array_like
        The free space diagram when calculating the Frechet distance
    
    Returns
    -------
    list (2-tuples)
        Describing a curve parameterization 
    '''
    i = freespace.shape[0] - 1
    j = freespace.shape[1] - 1
    traj = [(i-1,j-1)]
    while (i > 1) or (j > 1):
        c = freespace[i][j-1]
        b = freespace[i-1][j]
        a = freespace[i-1][j-1]
        idx = np.argmin([a,b,c])
        if idx == 0:
            i = i - 1
            j = j - 1
        elif idx == 1:
            i = i - 1
        else:
            j = j - 1
        traj.append((i-1,j-1))
    traj.reverse()
    return traj

def mean_trajectory(cluster):
    ''' Calculate Frechet mean for a longitudinal cluster
    
    Parameters
    ----------
    cluster: list
        A list of Trajectory class objects
        
    Returns
    -------
    tuple (array_like)
        The Frechet mean
    '''
    
    # time points in cluster
    times = list(set([y for x, y in cluster[0].parameterization]))

    # time 0
    
    # first trajectory in cluster
    steps = [x for x, y in cluster[0].parameterization if y == times[0]]
    vec = np.array([cluster[0].longitudinal[steps[0]]]) # initialize
    for s in steps[1:]:
        vec = np.append(vec, [cluster[0].longitudinal[s]], axis = 0)
    vec = np.mean(vec, axis = 0, where = vec > 0) # mean over individual
    cvec = np.array([vec])
    
    # other trajectories in cluster
    for i in range(1, len(cluster)):
        steps = [x for x, y in cluster[i].parameterization if y == times[0]]
        vec = np.array([cluster[i].longitudinal[steps[0]]]) # initialize
        for s in steps[1:]:
            vec = np.append(vec, [cluster[i].longitudinal[s]], axis = 0)
        vec = np.mean(vec, axis = 0, where = vec > 0)
        cvec = np.append(cvec, [vec], axis = 0) # mean over individual
    
    cvec = np.nanmean(cvec, axis = 0) # mean over individuals, ignorning nans
    cvec = np.true_divide(cvec, np.sum(cvec)) # scalar multiply to sum to 1
    traj = np.array([cvec])
    
    # other times
    for t in times[1:]:
        
        # first trajectory
        steps = [x for x, y in cluster[0].parameterization if y == t]
        vec = np.array([cluster[0].longitudinal[steps[0]]])
        for s in steps[1:]:
            vec = np.append(vec, [cluster[0].longitudinal[s]], axis = 0)
        vec = np.mean(vec, axis = 0, where = vec > 0) # mean over individual
        cvec = np.array([vec])
    
        # other trajectories
        for i in range(1, len(cluster)):
            steps = [x for x, y in cluster[i].parameterization if y == t]
            vec = np.array([cluster[i].longitudinal[steps[0]]])
            for s in steps[1:]:
                vec = np.append(vec, [cluster[i].longitudinal[s]], axis = 0)
            vec = np.mean(vec, axis = 0, where = vec > 0) # mean over individual
            cvec = np.append(cvec, [vec], axis = 0) 
    
        cvec = np.nanmean(cvec, axis = 0) # mean over individuals, ignoring nans
        cvec = np.true_divide(cvec, np.sum(cvec)) # scalar multiply to sum to 1
        traj = np.append(traj, [cvec], axis = 0)
    
    # averaging times
    nclust = len(cluster)
    timer = np.zeros(len(times))
    for t in times:
        for i in range(nclust):
            steps = [x for x, y in cluster[i].parameterization if y == t]
            timer[t] = timer[t] + np.mean(cluster[i].times[(steps[0]):(steps[-1] + 1)]) # mean time per individual
    timer = np.true_divide(timer, nclust)
        
    return((traj, timer))
    
def a_dunn_like_index(curves, groups, normfcn):
    ''' A Dunn-like index based on Frechet distance
    
    Parameters
    ----------
    curves: list
        A collection of Trajectory class objects
    groups: list
        A collection of Trajectory class objects
    normfcn: function
        A norm, e.g. Euclidean norm
    
    Returns
    -------
        A Dunn-like index
    '''
    def mean(x):
        return sum(x) / len(x)
    Delta = [mean([x.dist for x in curves if x.cluster == y.cluster]) for y in groups]
    
    prev = np.inf
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            curr = frechet_dist(free_space(groups[i].longitudinal,
                                           groups[j].longitudinal,
                                           groups[i].times,
                                           groups[j].times,
                                           normfcn))
                                          
            if curr < prev:
                prev = curr

    return prev / max(Delta)

def rand_index(x, y):
    ''' Rand index to compare clusters
    
    Parameters
    ----------
    x: list
        Cluster assignments
    y: list
        Cluster assignments
        
    Returns
    -------
    float
        The Rand index
    '''
    n = len(x)
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j]:
                if y[i] == y[j]:
                    a += 1
                else:
                    c += 1
            else:
                if y[i] == y[j]:
                    d += 1
                else:
                    b += 1
                    
    return (a + b) / (a + b + c + d)

def multiline(traj, scalar, steps = False, xl = (0,100), yl = (0,1), leg = False):
    ''' Plot colored lines in same figure
    
    Parameters
    ----------
    traj: Tracjectory class object
    scalar: float
    steps: bool
        True for discrete steps and False for continuous time
    xl: 2-tuple
        xlim parameter in matplotlib
    yl: 2-tuple
        ylim parameter in matplotlib
    leg: bool
        True for legend
    '''
    x = traj.longitudinal
    time = range(x.shape[0]) if steps else traj.times * (1 / scalar)  
    for i in range(x.shape[1]):
        plt.plot(time, x[:,i], label = i)
    plt.ylim(yl)
    plt.xlim(xl)
    if leg:
        plt.legend(loc='lower left', bbox_to_anchor=(1, 0.6))

def multiline_save(name, ttl, traj, scalar, steps = False, xl = (0,100), yl = (0,1), 
                   leg = False, xa = False, ya = False, xlbl = 'Time', ylbl = 'Relative Abundance'):
    ''' Plot colored lines in same figure and save
    
    Parameters
    ----------
    name: string
        A file name
    ttl: string
        Plot title
    traj: Tracjectory class object
    scalar: float
    steps: bool
        True for discrete steps and False for continuous time
    xl: 2-tuple
        xlim parameter in matplotlib
    yl: 2-tuple
        ylim parameter in matplotlib
    leg: bool
        True for legend
    xa: bool
        True for x-axis label
    ya: bool
        True for y-axis label
    xlbl: string
        x-axis label
    ylbl: string
        y-axis label
    '''
    x = traj.longitudinal
    time = range(x.shape[0]) if steps else traj.times * (1 / scalar)  
    for i in range(x.shape[1]):
        plt.plot(time, x[:,i], label = i)
    plt.ylim(yl)
    plt.xlim(xl)
    if xa:
        plt.xlabel(xlbl)
    if ya:
        plt.ylabel(ylbl)
    plt.suptitle(ttl)
    if leg:
        plt.legend(loc='lower left', bbox_to_anchor=(1, 0.6))
    plt.savefig(name)
    plt.close()
    
def introduce_zeros(p, df, idx):
    ''' Switch some data to zeros to study zero inflation
    
    Parameters
    ----------
    p: float
        Switch to zero probability
    df: pandas DataFrame
    idx: list
        Column indices
        
    Returns
    -------
    pandas DataFrame
    '''
    ndf = deepcopy(df)
    nrow = ndf.shape[0]
    az = []
    for i in range(nrow):
        for j in idx:
            if (uniform(0, 1) < p): # switch to zero
                ndf.iloc[i,j] = 0
                
        total = ndf.iloc[i,idx].sum()
        if total > 0:
            ndf.iloc[i,idx] = ndf.iloc[i,idx] / total # sum to 1
        else:
            az.append(i) # record when entirely zeros
            
    ndf = ndf.drop(az)
    return ndf