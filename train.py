import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve, minimize
from sklearn.linear_model import LogisticRegression

## For quick update of Vinv
def sherman_morrison(X, V, w=1):
    result = V-(w*np.einsum('ij,j,k,kl -> il', V, X, X, V))/(1.+w*np.einsum('i,ij,j ->', X, V, X))
    return result


'''
TS
'''
class TS:
    def __init__(self, d, v):
        ## Initialization
        self.beta_hat=np.zeros(d)
        self.f=np.zeros(d)
        self.Binv=np.eye(d)
        self.t = 0

        ## Hyperparameters
        self.v=v
        self.settings = {'v': self.v}

    def select_ac(self,contexts):
        ## Sample beta_tilde.
        N=len(contexts)
        V=(self.v**2)*self.Binv
        beta_tilde=np.random.multivariate_normal(self.beta_hat, V, size=N)
        est=np.array([np.dot(contexts[i], beta_tilde[i,]) for i in range(N)])
        ## Selecting action with tie-breaking.
        a_t=np.argmax(est)
        self.X_a=contexts[a_t]
        return(a_t)

    def update(self,reward):
        self.f=self.f+reward*self.X_a
        self.Binv = sherman_morrison(X=self.X_a, V=self.Binv)
        self.beta_hat=np.dot(self.Binv, self.f)

'''
UCB
'''
class UCB:
    def __init__(self, d, alpha, lam=1):
        self.alpha=alpha
        self.d=d
        self.yx=np.zeros(d)
        self.Binv=lam*np.eye(d)
        self.beta_hat = np.zeros(d)
        self.settings = {'alpha': self.alpha}

    def select_ac(self, contexts):
        means = np.array([np.dot(X, self.beta_hat) for X in contexts])
        stds = np.array([np.sqrt(X.T @ self.Binv @ X) for X in contexts])
        ucbs = means + self.alpha*stds
        a_t = np.argmax(ucbs)
        self.X_a = contexts[a_t]
        return(a_t)

    def update(self,reward):
        self.Binv = sherman_morrison(self.X_a, self.Binv)
        self.yx = self.yx+reward*self.X_a
        self.beta_hat = self.Binv @ self.yx

'''
PHE
'''
class PHE:
    def __init__(self, d, alpha, lam=1):
        self.alpha=alpha
        self.d=d
        self.yx=np.zeros(d)
        self.Binv=lam*np.eye(d)
        self.beta_hat = np.zeros(d)
        self.settings = {'alpha': self.alpha}
        self.context_list = []
        self.reward_list = []

    def select_ac(self, contexts):
        scores = np.array([np.dot(X, self.beta_hat) for X in contexts])
        a_t = np.argmax(scores)
        self.X_a = contexts[a_t]
        self.context_list.append(self.X_a)
        return(a_t)

    def update(self,reward):
        self.reward_list.append(reward[0])
        self.noise = np.random.normal(0, self.alpha, size=(len(self.reward_list)))
        pseudo_reward = np.array(self.reward_list) + self.noise
        pseudo_reward = np.repeat(pseudo_reward, self.d).reshape(-1, self.d)

        self.Binv = sherman_morrison(self.X_a, self.Binv)
        self.yx = np.sum(np.multiply(np.array(self.context_list), pseudo_reward), axis=0)
        self.beta_hat = self.Binv @ self.yx