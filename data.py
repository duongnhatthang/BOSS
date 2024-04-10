import numpy as np

def generate_contexts(N, d, rho, seed=0, delta=10e-5):
    ## Generate contexts
    # input: N, d
    # output: [X(1),...,X(N)] list with N contexts in R^d
    n = int(N/2)
    if N % 2 == 1:
        X_mean = np.arange(-2*n, 2*n+1, 2)
    else:
        X_mean = np.hstack((np.arange(-2*n,0,2), np.arange(2,2*n+1,2)))
    X_cov = (1-rho)*np.eye(N) + rho*np.ones((N,N))
    np.random.seed(seed) # For reproducibility
    X_set = np.array([np.random.multivariate_normal(mean=X_mean, cov=X_cov) for i in range(d-1)])
    selected_cordinate = np.random.choice(np.arange(0,d-1,1), size=N)
    append_set = np.array([X_set[selected_cordinate[i],i] for i in range(N)])
    X_set = np.vstack((X_set, append_set))
    u = np.random.uniform(0,1,N)
    return([u[i]*X_set[:,i]/np.linalg.norm(X_set[:,i]) for i in range(N)])
