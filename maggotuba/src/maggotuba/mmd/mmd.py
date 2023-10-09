import numpy as np
import scipy.stats as stats

def rbf_dot(patterns1,patterns2,sigma):
    '''
    Arguments:
        patterns1 : N * D matrix of samples from the first distribution
        patterns2 : M * D matrix of samples from the second distribution
        sigma       : width of the gaussian kernel
    Returns:
        H         : N * M matrix of kernel evaluations
    ''' 

    # Note : patterns are transposed for compatibility with C code.

    size1=patterns1.shape
    size2=patterns2.shape

    # new vectorised version
    G = np.sum(patterns1**2, axis=1, keepdims=True)
    H = np.sum(patterns2**2, axis=1, keepdims=True)

    Q = np.tile(G,[1,size2[0]])
    R = np.tile(H.transpose(),[size1[0],1])
    H = Q + R - 2*np.dot(patterns1, patterns2.transpose()) # N * M
                                                           # ||patterns1[i]-patterns2[j]||_2^2


    H = np.exp(-H/2/sigma**2) # N * M
                            # k(patterns1[i], patterns2[j]) = RBF(patterns1[i]-patterns2[j])

    return H

def compute_linear_estimator_of_mmd(X, Y, sigma):
    '''Compute linear time estimator of the mmd and associated variance
    Parameters
    ----------
        X : array-like of shape (n_samples1, n_features)
            First sample
        Y : array-like of shape (n_samples2, n_features)
            Second sample
        sigma : float
            kernel bandwidth
    Outputs
    -------
        mmd2 : float
            estimate of the mmd squared
        var : float
            empirical estimate of sigma^2_l/2.
            See "A kernel two sample test", Gretton et al., corollary 16
        m : int
            useful to compute p-value

    '''
    # equalize size
    if X.shape[0] != Y.shape[0]:
        m = min(X.shape[0], Y.shape[0])
    else:
        m = X.shape[0]
    m = m-m%2 # ensure even length
    X = X[:m]
    Y = Y[:m]

    def rbf(r):
        return np.exp(-0.5*np.sum(r**2,axis=-1)/sigma**2)
    def h(x1, y1, x2, y2):
        return rbf(x1-x2) + rbf(y1-y2) - rbf(x1-y2) - rbf(x2-y1)

    h = h(X[::2], Y[::2], X[1::2], Y[1::2])
    mmd2 = np.mean(h)
    var = np.var(h, ddof=1)
    return mmd2, var, m

def pval(mmd2, sigma2, m):
    return 1.-stats.norm.cdf(np.sqrt(m/(2*sigma2))*mmd2) 

def compute_quadratic_estimator_of_mmd(X, Y, sigma):
    '''Compute quadratic time estimator of the mmd and associated variance
    Parameters
    ----------
        X : array-like of shape (n_samples1, n_features)
            First sample
        Y : array-like of shape (n_samples2, n_features)
            Second sample
        sigma : float
            kernel bandwidth
    Outputs
    -------
        mmd2 : float
            estimate of the mmd squared
        var : float
            empirical estimate of sigma^2_l.
            See "A kernel two sample test", Gretton et al., corollary 16
        m : int
            useful to compute p-value

    '''
    if X.shape == Y.shape:
        return computeTestStatSameSampleSize(X, Y, sigma)
    else:
        return computeTestStatDiffSampleSize(X, Y, sigma)


def computeTestStatSameSampleSize(X, Y, sigma):
  K = rbf_dot(X,X,sigma)
  L = rbf_dot(Y,Y,sigma)
  KL = rbf_dot(X,Y,sigma)
  m = X.shape[0] # number of rows in X

  # MMD statistic. Here we use biased 
  # v-statistic. NOTE: this is m * MMD_b
  H = K + L - KL - KL.transpose()
  sqMMD = 2/m/(m-1)*np.sum(np.triu(H, k=1))

  return sqMMD

def computeTestStatDiffSampleSize(X, Y, sigma):
    K  = rbf_dot(X,X,sigma)
    L  = rbf_dot(Y,Y,sigma)
    KL = rbf_dot(X,Y,sigma)

    m = X.shape[0]
    n = Y.shape[0]

    A = 2*np.sum(np.triu(K, k=1))
    B = 2*np.sum(np.triu(L, k=1))
    C = np.sum(KL)

    sqMMD = A/m/(m-1) + B/n/(n-1) - 2/m/n*C
    return sqMMD

# Witness function plotting helpers

def computeWitnessFunctionOnMesh(xx, yy, data1, data2, sigma):
    def wit(x, y):
        point = np.array([[x,y]])
        return np.mean(np.exp(-0.5*np.sum((point-data1)**2, axis=1)/sigma)) - np.mean(np.exp(-0.5*np.sum((point-data2)**2, axis=1)/sigma))
    
    witness = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            witness[i,j] = wit(xx[i,j], yy[i,j])
    return witness

def get_mesh_from_2d_embeddings(*data, N=50):
    data = np.vstack(data)
    xmin, xmax = np.min(data[:,0]), np.max(data[:,0])
    xmin, xmax = xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin)
    ymin, ymax = np.min(data[:,1]), np.max(data[:,1])
    ymin, ymax = ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin)
    span_x = np.linspace(xmin,xmax,N)
    span_y = np.linspace(ymin,ymax,N)

    return np.meshgrid(span_x, span_y)

def get_inliers(points):
    def objective(r):
        return np.mean(np.logical_and(np.abs(points[:,0])<r, np.abs(points[:,1])<r))-0.99
    from scipy.optimize import root_scalar
    res = root_scalar(objective, bracket=(0,100), x0=1., method='bisect')
    r = res.root
    inliers_ind = np.logical_and(np.abs(points[:,0])<r, np.abs(points[:,1])<r)
    return points[inliers_ind]