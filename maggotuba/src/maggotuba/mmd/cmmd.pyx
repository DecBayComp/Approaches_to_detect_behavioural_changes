# could be even faster with some caching for the squares.

from libc.math cimport exp, pow
import numpy as np
import scipy.stats as stats
cimport cython

ctypedef fused numeric:
    float
    double
    long double

# MMD2 FUNCTIONS
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) 
def cmmd2(numeric[:, :] array_1, numeric[:, :] array_2, numeric sigma):
    cdef Py_ssize_t n1 = array_1.shape[0]
    cdef Py_ssize_t n2 = array_2.shape[0]
    cdef Py_ssize_t d = array_1.shape[1]
    cdef numeric square1 = 0.0
    cdef numeric square2 = 0.0
    cdef numeric product = 0.0
    cdef numeric result  = 0.0
    cdef numeric norm = 0.0

    assert(array_1.shape[1] == array_2.shape[1])

    cdef Py_ssize_t i, j, k

    # scalar square 1
    for i in range(n1):
        for j in range(i+1, n1):
            norm = 0.0
            for k in range(d):
                norm += pow(array_1[i,k]-array_1[j,k], 2)
            square1 += exp(-norm/(2.*sigma*sigma))
    square1 *= 2./n1/(n1-1)

    # scalar square 2
    for i in range(n2):
        for j in range(i+1, n2):
            norm = 0.0
            for k in range(d):
                norm += pow(array_2[i,k]-array_2[j,k], 2)
            square2 += exp(-norm/(2.*sigma*sigma))
    square2 *= 2./n2/(n2-1)

    # scalar product
    for i in range(n1):
        for j in range(n2):
            norm = 0.0
            for k in range(d):
                norm += pow(array_1[i,k]-array_2[j,k], 2)
            product += exp(-norm/(2.*sigma*sigma))
    product *= 1./(n1*n2)



    return square1 + square2 - 2 * product

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)           
def cscalar_square(float[:, :] array, float sigma):
    cdef Py_ssize_t n = array.shape[0]
    cdef Py_ssize_t d = array.shape[1]
    cdef float square = 0.0

    cdef Py_ssize_t i, j, k

    for i in range(n):
        for j in range(i+1, n):
            norm = 0.0
            for k in range(d):
                norm += pow(array[i,k]-array[j,k], 2)
            square += exp(-norm/(2.*sigma*sigma))
    square *= 2./n/(n-1)
    return square

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)  
def cscalar_product(float[:, :] array_1, float[:, :] array_2, float sigma):
    cdef Py_ssize_t n1 = array_1.shape[0]
    cdef Py_ssize_t n2 = array_2.shape[0]
    cdef Py_ssize_t d = array_1.shape[1]
    cdef float square1 = 0.0
    cdef float square2 = 0.0
    cdef float product = 0.0
    cdef float result  = 0.0
    cdef float norm = 0.0

    assert(array_1.shape[1] == array_2.shape[1])


    cdef Py_ssize_t i, j, k

    for i in range(n1):
        for j in range(n2):
            norm = 0.0
            for k in range(d):
                norm += pow(array_1[i,k]-array_2[j,k], 2)
            product += exp(-norm/(2.*sigma*sigma))
    product *= 1./(n1*n2)
    
    return product

# BOOTSTRAP
cdef Py_ssize_t index_distmat(Py_ssize_t i, Py_ssize_t j, Py_ssize_t N):
    assert(i<j)
    return (i+1)*i/2 + i*(N-i-1) + (j-i) - 1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) 
def cbootstrap(numeric[:,:] array_1, numeric[:,:] array_2, numeric sigma, int nboot):
    cdef Py_ssize_t n1 = array_1.shape[0]
    cdef Py_ssize_t n2 = array_2.shape[0]
    cdef Py_ssize_t d = array_1.shape[1]
    cdef Py_ssize_t N = n1+n2

    cdef numeric square1 = 0.0
    cdef numeric square2 = 0.0
    cdef numeric product = 0.0
    cdef numeric norm = 0.0
    cdef numeric testStat = 0.0

    if numeric is float:
        distmat     = np.zeros(N*(N-1)//2, dtype=np.single)
        bootstrap   = np.zeros(nboot,      dtype=np.single)
        stacked_arr = np.zeros((n1+n2, d), dtype=np.single)
    elif numeric is double:
        distmat     = np.zeros(N*(N-1)//2, dtype=np.double)
        bootstrap   = np.zeros(nboot,      dtype=np.double)
        stacked_arr = np.zeros((n1+n2, d), dtype=np.double)
    elif numeric is cython.longdouble:
        distmat     = np.zeros(N*(N-1)//2, dtype=np.longdouble)
        bootstrap   = np.zeros(nboot,      dtype=np.longdouble)
        stacked_arr = np.zeros((n1+n2, d), dtype=np.longdouble)

    cdef numeric[:,::1] stacked_arr_view = stacked_arr
    cdef numeric[:] distmat_view = distmat
    cdef numeric[:] bootstrap_view = bootstrap

    assert(array_1.shape[1] == array_2.shape[1])
    assert(array_1.shape[0] > 1)
    assert(array_2.shape[0] > 1)

    cdef Py_ssize_t i, j, k, boot

    # initialize the stacked array
    for i in range(n1+n2):
        for k in range(d):
            if i < n1:
                stacked_arr_view[i,k] = array_1[i,k]
            else:
                stacked_arr_view[i,k] = array_2[i-n1,k]

    # compute the gram matrix
    for i in range(n1+n2):
        for j in range(i+1, n1+n2):
            norm = 0.0
            for k in range(d):
                norm += pow(stacked_arr_view[i,k]-stacked_arr_view[j,k], 2)
            distmat_view[index_distmat(i,j,N)] += exp(-norm/(2.*sigma*sigma))
    print("Done with the gram matrix")

    split_indicator = np.zeros(N, dtype=np.intc)
    cdef int[:] split_indicator_view = split_indicator

    split = np.zeros(n1, dtype=np.intc)
    cdef int[:] split_view = split


    rng = np.random.default_rng()

    # do the bootstrap
    for boot in range(nboot):
        square1 = 0.
        square2 = 0.
        product = 0.

        # zero the indicator
        for i in range(n1):
            split_indicator_view[split_view[i]] = 0
        # refill the indicator with new permutation
        split_view = rng.choice(N, size=n1, replace=False, shuffle=False).astype(np.intc)
        for i in range(n1):
            split_indicator_view[split_view[i]] = 1

        # traverse the distance matrix and compute the sums
        for i in range(n1+n2):
            for j in range(i+1, n1+n2):
                if split_indicator_view[i]:
                    if split_indicator_view[j]:
                        # compute square1
                        square1 += distmat_view[index_distmat(i, j, N)]
                    else:
                        # compute product
                        product += distmat_view[index_distmat(i, j, N)]
                else:
                    if split_indicator_view[j]:
                        # compute product
                        product += distmat_view[index_distmat(i, j, N)]
                    else:
                        # compute square2
                        square2 += distmat_view[index_distmat(i, j, N)]
        
        # normalize
        square1 *= (2./n1)/(n1-1)
        square2 *= (2./n2)/(n2-1)
        product *= 1./(n1*n2)

        # store record
        bootstrap_view[boot] = square1 + square2 - 2 * product

        if not(boot%100):
            print(f"{boot}/{nboot}")

    # compute the actual test statistic
    square1 = 0.
    square2 = 0.
    product = 0.
    # traverse the distance matrix and compute the sums
    for i in range(n1+n2):
        for j in range(i+1, n1+n2):
            if i < n1:
                if j < n1:
                    # compute square1
                    square1 += distmat_view[index_distmat(i, j, N)]
                else:
                    # compute product
                    product += distmat_view[index_distmat(i, j, N)]
            else:
                if j < n1:
                    # compute product
                    product += distmat_view[index_distmat(i, j, N)]
                else:
                    # compute square2
                    square2 += distmat_view[index_distmat(i, j, N)]
    
    # normalize
    square1 *= (2./n1)/(n1-1)
    square2 *= (2./n2)/(n2-1)
    product *= 1./(n1*n2)
    testStat = square1 + square2 - 2 * product

    return testStat, bootstrap


#######################################################################
#
#                     WITNESS FUNCTION EVALUATION
#
########################################################################

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) 
def cwitness(numeric[:,:] array_1, numeric[:,:] array_2, numeric sigma):
    cdef Py_ssize_t n1 = array_1.shape[0]
    cdef Py_ssize_t n2 = array_2.shape[0]
    cdef Py_ssize_t d = array_1.shape[1]
    cdef Py_ssize_t N = n1+n2

    cdef numeric norm = 0.0

    if numeric is float:
        distmat     = np.zeros(N*(N-1)//2, dtype=np.single)
        witness     = np.zeros(n1+n2,      dtype=np.single)
        stacked_arr = np.zeros((n1+n2, d), dtype=np.single)
    elif numeric is double:
        distmat     = np.zeros(N*(N-1)//2, dtype=np.double)
        witness     = np.zeros(n1+n2,      dtype=np.double)
        stacked_arr = np.zeros((n1+n2, d), dtype=np.double)
    elif numeric is cython.longdouble:
        distmat     = np.zeros(N*(N-1)//2, dtype=np.longdouble)
        witness     = np.zeros(n1+n2,      dtype=np.longdouble)
        stacked_arr = np.zeros((n1+n2, d), dtype=np.longdouble)

    cdef numeric[:,::1] stacked_arr_view = stacked_arr
    cdef numeric[:] distmat_view = distmat
    cdef numeric[:] witness_view = witness

    assert(array_1.shape[1] == array_2.shape[1])
    assert(array_1.shape[0] > 1)
    assert(array_2.shape[0] > 1)

    cdef Py_ssize_t i, j, k, boot

    # initialize the stacked array
    for i in range(n1+n2):
        for k in range(d):
            if i < n1:
                stacked_arr_view[i,k] = array_1[i,k]
            else:
                stacked_arr_view[i,k] = array_2[i-n1,k]

    # compute the gram matrix
    for i in range(n1+n2):
        for j in range(i+1, n1+n2):
            norm = 0.0
            for k in range(d):
                norm += pow(stacked_arr_view[i,k]-stacked_arr_view[j,k], 2)
            distmat_view[index_distmat(i,j,N)] += exp(-norm/(2.*sigma*sigma))

    cdef numeric lhs, rhs
    for i in range(n1+n2):
        lhs = 0.
        rhs = 0.
        for j1 in range(n1):
            if j1 > i:
                lhs += distmat_view[index_distmat(i, j1, N)]
            else:
                lhs += distmat_view[index_distmat(j1, i, N)]

        for j2 in range(n1, n1+n2):
            if j2 > i:
                rhs += distmat_view[index_distmat(i, j2, N)]
            else:
                rhs += distmat_view[index_distmat(j2, i, N)]
        witness_view[i] = lhs/n1 - rhs/n2

    return witness[:n1].copy(), witness[n1:].copy()
