import numpy as np 
import scipy
import math
from scipy.spatial import distance
import scipy.stats
from numpy.linalg import inv


def MWDistance(point1,point2,weights,means,covariances):
    # Function for computing model weighted distance between two points
    # The model weighted distance takes the landscape modelled by a GMM into acount to compute distances 
    
    # Lowering precision to avoid precision errors
    point1 = np.round(point1,12)
    point2 = np.round(point2,12)

    assert len(point1) == len(point2), "Length of the vectors for distance don't match."
    
    if list(point1) == list(point2):
        return 0
    
    point1 = np.array(point1)
    point2 = np.array(point2)
  
    K = len(weights) # number of GMM componenets
   
    #Computing G metric
    G_top = []
    G_bottom = []
    for k in range(K): # looping over all components to generate sums in G
        weight_k = weights[k]
        mean_k = means[k]
        C_k = scipy.linalg.inv(covariances[k])
        
        v = point1 - point2
        u = mean_k - point2
        b2 =  1/(v.T @ C_k @ v)
        a = b2 * (v.T @ C_k @ u)
        Z = u.T @ C_k @ u - b2 * (v.T @ C_k @ u)**2
        
        erf_term1 = (1-a) / (np.sqrt(2*b2))
        erf_term2 = (-a) / (np.sqrt(2*b2))
        
        integral_term1 = np.sqrt((np.pi*b2)/2)*np.exp(-Z/2)
        integral_term2 = math.erf(erf_term1)-math.erf(erf_term2)
        integral = integral_term1 * integral_term2
        
        G_top.append(C_k*weight_k*integral)
        G_bottom.append(weight_k*integral)
        
    # Computing sums in G metric
    G_top = sum(G_top)
    G_bottom = sum(G_bottom)
    G = G_top / G_bottom    
    
    # Computing model weighted distance
    return distance.mahalanobis(point1 , point2, G)




def soft_fairness(gamma,A):
    '''
    Function for determining fairness of a soft clustering output.
    Fairness is defined in terms of entropy of the cluster configuration.
    Finds the entropy for one protected group over all clusters. 

    Input Parameters
    ----------
    gamma : np.array (responsibility matrix of dimension N x k)
    A     : np.array (protected attribute vector of length N)
    
        Input is responsibility vector gamma and protected attribute vector A.

    Returns
    ---------
    H       : float (entropy of cluster configuration)
    H_ratio : float (entropy ratio of cluster configuration)
    balance : float (cluster balance from cluster with minimum balance)
    
        Outputs soft entropy, entropy ratio and balance of cluster configuration. 
        Entropy ratio is defined by ratio of cluster entropy to the optimal entropy
        for the number of clusters (i.e. same proportion of fraction of protected attribute in all clusters).
        
    '''
    # Setup
    k  = gamma.shape[1]                           # Number of clusters 
    r = sum(A)/len(A)                             # Fraction of protected attribute in full data
    
    # Cluster Masses
    total_mass     = sum(gamma)                   # Weighted total size by the sum of point masses (responsibilities)
    protected_mass = sum(gamma*A[:,None])         # Weighted protected size by the sum of protected mass

    # Configuration Entropy
    p = protected_mass/(total_mass)               # Weighted probability of protected attribute 
    p = p[~np.isnan(p)]                           # If a cluster has no members do not consider it.
    
    H = scipy.stats.entropy(p)

    # Baseline Entropy
    k = len(p)                                    # Number of components containing data points
    p_optimal = np.ones(k)/k                      # Baseline is optimal entropy 
    H_optimal = -sum(p_optimal*np.log(p_optimal)) # Entropy of configuration
    H_ratio   =  H / H_optimal                    # Balance between [0,1] defined by entropy versus perfect entropy

    # Cluster Balance
    if 0 in p:                                    # If monochromatic cluster avoid dividing by zero and let balance be zero
        balance = 0
    else:                                         # If balance is not zero compute the balance
        R = r / p                                 # R_(c,p)  - ratio of protected attribute in data and cluster
        balance = np.min(np.array([R,1/R]))       # Cluster balance

    return H,H_ratio,balance,p


def MHD_GMM_Cost(gamma,points,weights,means,covariances):
    # Function for computing cost of GMM fit. 
    # Cost is the sum of weighted Mahalanobis distances from each point to their associated mixture components. 
    # Weights are given by the responsibilities gamma
    point_costs = []
    K = len(weights) # number of GMM componenets

    for i in range(points.shape[0]):
        for j in range(K):
            point1 = points[i]
            point2 = means[j]
            point_costs.append(gamma[i,j]*scipy.spatial.distance.mahalanobis(point1,point2,np.linalg.inv(covariances[j])))
    return sum(point_costs)


def center_cost(points,weights,means,covariances):
    # Helper function for computing the fairlet decomposition cost from all fairlet members to their centers,
    # where the centers are chosen as the mean value for each fairlet.
    point_costs=[]
    point2 = np.mean(points,axis=0)
    for i in range(points.shape[0]):
        point1 = points[i]
        point_costs.append(MWDistance(point1,point2,weights,means,covariances))
    return sum(point_costs)



def logL(k,S,D,G_metric):
    '''
    Function for computing the loglikelihood of the decomposition. 
    
    Likelihood of an observation from a presumed normal distribution with cov matrix S and Mahalanobis distance D.
    k is the multivariate dimension    
    S is the covariance matrix. Can also be G metric (weighted avergae of multiple S along the path).
    D is the distance (Mahalanobis or MWD)
    G_metric [False,True] flag stating if we are using G_metric (multiple components) or covariance matrix (single component).
    '''
    
    # If using the G metric from MWD remember that G is the inverted D matrix. 
    if G_metric == True:
        S = np.linalg.inv(S)
        
    c = (1/2) * ( np.log(np.linalg.det(S)) + k * np.log(2*np.pi) )
    logLikelihood = -(1/2) * D**2 - c
    return logLikelihood




def MWDG(point1,point2,weights,means,covariances,prob_at_x):
    # Function for returning the G metric along the path from point 1 to point 2
    # Used for the decomposition likelihood computation.
    
    # Lowering precision to avoid precision errors
    point1 = np.round(point1,12)
    point2 = np.round(point2,12)

    assert len(point1) == len(point2), "Length of the vectors for distance don't match."
    
    # If points are the same return distance of zero and G metric as probabilistically weighted average of inverse covs
    if list(point1) == list(point2):        
        sum_terms = []
        for i in range(len(covariances)):
            sum_terms.append(prob_at_x[i]*np.linalg.inv(covariances[i]))   
        G = np.sum(np.array(sum_terms),axis=0)
        return 0,G 
    
    point1 = np.array(point1)
    point2 = np.array(point2)
  
    K = len(weights) # number of GMM componenets
    
    #Computing G metric
    G_top = []
    G_bottom = []
    for k in range(K): # looping over all components to generate sums in G
        weight_k = weights[k]
        mean_k = means[k]
        C_k = scipy.linalg.inv(covariances[k])
        
        v = point1 - point2
        u = mean_k - point2
        b2 =  1/(v.T @ C_k @ v)
        a = b2 * (v.T @ C_k @ u)
        Z = u.T @ C_k @ u - b2 * (v.T @ C_k @ u)**2
        
        erf_term1 = (1-a) / (np.sqrt(2*b2))
        erf_term2 = (-a) / (np.sqrt(2*b2))
        
        integral_term1 = np.sqrt((np.pi*b2)/2)*np.exp(-Z/2)
        integral_term2 = math.erf(erf_term1)-math.erf(erf_term2)
        integral = integral_term1 * integral_term2
        
        G_top.append(C_k*weight_k*integral)
        G_bottom.append(weight_k*integral)
        
    # Computing sums in G metric
    G_top = sum(G_top)
    G_bottom = sum(G_bottom)
    G = G_top / G_bottom
    
    dist = distance.mahalanobis(point1 , point2, G)
      
    
    # Computing model weighted distance
    return dist, G