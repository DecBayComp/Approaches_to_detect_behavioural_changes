from .cmmd import cmmd2, cbootstrap
import numpy as np
from timeit import timeit
from memory_profiler import memory_usage
import matplotlib.pyplot as plt

def rbf_dot(patterns1,patterns2,sigma):
    size1=patterns1.shape
    size2=patterns2.shape
    G = np.sum(patterns1**2, axis=1, keepdims=True)
    H = np.sum(patterns2**2, axis=1, keepdims=True)
    Q = np.tile(G,[1,size2[0]])
    R = np.tile(H.transpose(),[size1[0],1])
    H = Q + R - 2*np.dot(patterns1, patterns2.transpose()) # N * M
                                                           # ||patterns1[i]-patterns2[j]||_2^2
    H = np.exp(-H/2/sigma**2) # N * M
                            # k(patterns1[i], patterns2[j]) = RBF(patterns1[i]-patterns2[j])

    return H

def computeTestStat(X, Y, sigma):
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

def main1():
    rng = np.random.default_rng()

    mem_np, t_np, mem_cy, t_cy = [], [], [], []
    sizes = [100, 200, 300, 400, 500, 600, 1000, 2500, 5000, 10000]
    for s in sizes:
        print(s)
        x = rng.normal(size=(s,10))
        y = rng.normal(size=(s,10))+0.1

        t_np.append(timeit(lambda: computeTestStat(x, y, 1.0), number=10))
        mem_np.append(max(memory_usage(lambda: computeTestStat(x, y, 1.0))))

        t_cy.append(timeit(lambda: cmmd2(x, y, 1.0), number=10))
        mem_cy.append(max(memory_usage(lambda: cmmd2(x, y, 1.0))))

    print("sizes        | ", ' '.join([f"{s}".ljust(8) for s in sizes]))
    print("cython times | ", ' '.join([f"{t:1.3f}".ljust(8) for t in t_cy]))
    print("numpy  times | ", ' '.join([f"{t:1.3f}".ljust(8) for t in t_np]))
    print("cython mem   | ", ' '.join([f"{m:.0f}".ljust(8) for m in mem_cy]))
    print("numpy  mem   | ", ' '.join([f"{m:.0f}".ljust(8) for m in mem_np]))

    # Plot
    _, axs = plt.subplots(1, 2)
    axs[0].set_title('Execution time')
    axs[0].set_xlabel('Number of samples')
    axs[0].plot(sizes, t_np)
    axs[0].plot(sizes, t_cy)
    axs[0].legend(labels=['numpy', 'cython'])
    axs[1].set_title('Memory usage')
    axs[1].set_xlabel('Number of samples')
    axs[1].plot(sizes, mem_np)
    axs[1].plot(sizes, mem_cy)
    axs[1].legend(labels=['numpy', 'cython'])
    plt.show()

    # check correctness
    for _ in range(10):
        x = rng.normal(size=(1000,10))
        y = rng.normal(size=(1000,10))+0.1
        try:
            assert(np.isclose(computeTestStat(x, y, 1.0), cmmd2(x, y, 1.0)))
            print("Successful test.")
        except:
            print("Failed test.")

def main2():
    rng = np.random.default_rng()
    x = rng.normal(size=(20,10))
    y = rng.normal(size=(100,10))+0.1

    nboot = 10000
    
    print("Building bootstraps for visual comparison...")
    testStat, bootstrap = cbootstrap(x,y,1.,nboot)

    def manual_bootstrap(x, y, sigma, nboot):
        bootstrap2 = np.empty(nboot)
        stacked_xy = np.vstack([x,y])
        for boot in range(nboot):
            split = rng.choice(len(x)+len(y), len(x), replace=False, shuffle=False)
            u, v = stacked_xy[split], np.delete(stacked_xy, split, axis=0)
            bootstrap2[boot] = cmmd2(u, v, sigma)
        testStat2 = cmmd2(x, y, sigma)
        return testStat2, bootstrap2

    testStat2, bootstrap2 = manual_bootstrap(x, y, 1.0, nboot)

    assert np.isclose(testStat, testStat2)
    print("Test statistics : ", testStat, ' | ', testStat2)
    print(len(bootstrap), len(bootstrap2))

    plt.figure()
    plt.gca().hist(bootstrap,  alpha=0.6, bins=100, density=True)
    plt.gca().hist(bootstrap2, alpha=0.6, bins=100, density=True)
    plt.show()

    # time profiling
    manual_times = []
    cython_times = []
    sizes = [100, 200, 300, 400, 500, 600, 1000]#, 2500, 5000, 10000]
    print("Profiling...")
    for s in sizes:
        x = rng.normal(size=(s,10))
        y = rng.normal(size=(s,10))+0.1
        print(f"Size {s}...")
        if s <= 1000:
            manual_times.append(timeit(lambda: manual_bootstrap(x, y, 1.0, 1000), number=1))
        else:
            manual_times.append(np.nan)
        cython_times.append(timeit(lambda: cbootstrap(x, y, 1.0, 1000), number=1))

    plt.figure()
    plt.plot(sizes, manual_times)
    plt.plot(sizes, cython_times)
    plt.legend(labels=['Memory efficient cython bootstrap, 1000 draws', 'Time efficient cython bootstrap, 1000 draws'])
    plt.xlabel('Sample size')
    plt.ylabel('Time (s)')
    plt.show()

def test_indexer():
    print("Testing the indexer...")
    def indexer(i,j,N):
        return ((i+1)*i)//2 + i*(N-i-1) + (j-i) - 1
    for i in range(10):
        for j in range(10):
            if j < i+1:
                print(" *", end=' ')
            else:
                print(str(indexer(i,j,10)).ljust(2), end=' ')
        print()
    print("Done.")
    print(80*'-')

if __name__ == '__main__':
    test_indexer()
    main2()