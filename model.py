import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve, minimize
from sklearn.linear_model import LogisticRegression
from scipy.stats import ortho_group

## For quick update of Vinv
def sherman_morrison(X, V, w=1):
    result = V-(w*np.einsum('ij,j,k,kl -> il', V, X, X, V))/(1.+w*np.einsum('i,ij,j ->', X, V, X))
    return result


'''
TS
'''
class TS:
    def __init__(self, d, v):
        ## Hyperparameters
        self.v=v
        self.settings = {'v': self.v}
        self.d = d

        ## Initialization
        self.reset()

    def select_ac(self,contexts):
        ## Sample theta_tilde.
        N=len(contexts)
        V=(self.v**2)*self.Binv
        theta_tilde=np.random.multivariate_normal(self.theta_hat, V, size=N)
        est=np.array([np.dot(contexts[i], theta_tilde[i,]) for i in range(N)])
        ## Selecting action with tie-breaking.
        a_t=np.argmax(est)
        self.X_a=contexts[a_t]
        return(a_t)
    
    def select_ctx(self, contexts):
        """
        Instead of returning an index of the context chosen from contexts,
        returning a customized context
        """
        a_t = self.select_ac(contexts)
        return contexts[a_t]

    def update(self,reward):
        self.f=self.f+reward*self.X_a
        self.Binv = sherman_morrison(X=self.X_a, V=self.Binv)
        self.theta_hat=np.dot(self.Binv, self.f)

    def reset(self):
        self.theta_hat=np.zeros(self.d)
        self.f=np.zeros(self.d)
        self.Binv=np.eye(self.d)
        self.t = 0

'''
UCB
'''
class UCB:
    def __init__(self, d, alpha, lam=1):
        self.alpha=alpha
        self.d=d
        self.lam=lam
        self.settings = {'alpha': self.alpha}
        self.reset()

    def select_ac(self, contexts):
        means = np.array([np.dot(X, self.theta_hat) for X in contexts])
        stds = np.array([np.sqrt(X.T @ self.Binv @ X) for X in contexts])
        ucbs = means + self.alpha*stds
        a_t = np.argmax(ucbs)
        self.X_a = contexts[a_t]
        return(a_t)
    
    def select_ctx(self, contexts):
        """
        Instead of returning an index of the context chosen from contexts,
        returning a customized context
        """
        a_t = self.select_ac(contexts)
        return contexts[a_t]

    def update(self,reward):
        self.Binv = sherman_morrison(self.X_a, self.Binv)
        self.yx = self.yx+reward*self.X_a
        self.theta_hat = self.Binv @ self.yx

    def reset(self):
        self.yx=np.zeros(self.d)
        self.Binv=self.lam*np.eye(self.d)
        self.theta_hat = np.zeros(self.d)

'''
PHE (Perturbed-History Exploration ?)
'''
class PHE:
    def __init__(self, d, alpha, lam=1):
        self.alpha=alpha
        self.d=d
        self.lam=lam
        self.settings = {'alpha': self.alpha}
        self.reset()

    def select_ac(self, contexts):
        scores = np.array([np.dot(X, self.theta_hat) for X in contexts])
        a_t = np.argmax(scores)
        self.X_a = contexts[a_t]
        self.context_list.append(self.X_a)
        return(a_t)
    
    def select_ctx(self, contexts):
        """
        Instead of returning an index of the context chosen from contexts,
        returning a customized context
        """
        a_t = self.select_ac(contexts)
        return contexts[a_t]

    def update(self,reward):
        self.reward_list.append(reward[0])
        self.noise = np.random.normal(0, self.alpha, size=(len(self.reward_list)))
        pseudo_reward = np.array(self.reward_list) + self.noise
        pseudo_reward = np.repeat(pseudo_reward, self.d).reshape(-1, self.d)

        self.Binv = sherman_morrison(self.X_a, self.Binv)
        self.yx = np.sum(np.multiply(np.array(self.context_list), pseudo_reward), axis=0)
        self.theta_hat = self.Binv @ self.yx

    def reset(self):
        self.yx=np.zeros(self.d)
        self.Binv=self.lam*np.eye(self.d)
        self.theta_hat = np.zeros(self.d)
        self.context_list = []
        self.reward_list = []

'''
PEGE
'''
class PEGE:
    def __init__(self, d, tau_1, EXR_contexts=None, lam=1):
        self.tau_1=tau_1
        self.d=d
        self.lam=lam
        self.settings = {'tau_1': self.tau_1}
        self.EXR_contexts=EXR_contexts
        self.reset()

    def select_ctx(self, contexts):
        if self.step <= self.tau_1:
            if self.EXR_contexts is None:
                self.X_a = np.random.uniform(low=-1, high=1, size=self.d)
                u = np.random.uniform(0,1) #Scaling factor
                self.X_a = u*self.X_a/np.linalg.norm(self.X_a) #ensure unit ball length
            else:
                idx = self.step % len(self.EXR_contexts)
                self.X_a = self.EXR_contexts[idx]
        else:
            scores = np.array([np.dot(X, self.theta_hat) for X in contexts])
            a_t = np.argmax(scores)
            self.X_a = contexts[a_t]
        self.context_list.append(self.X_a)
        self.step += 1
        return self.X_a 

    def update(self,reward):
        if self.step <= self.tau_1:
            self.Binv = sherman_morrison(self.X_a, self.Binv)
            self.yx = self.yx+reward*self.X_a
            self.theta_hat = self.Binv @ self.yx

    def reset(self):
        self.yx=np.zeros(self.d)
        self.Binv=self.lam*np.eye(self.d)
        self.theta_hat = np.zeros(self.d)
        self.step = 0

'''
PMA
'''
class PMA:
    def __init__(self, input_dict, true_B, alpha, lam=1):
        self.alpha=alpha
        self.lam=lam
        self.input_dict=input_dict
        T = self.input_dict["T"]
        d = self.input_dict["d"]
        m = self.input_dict["m"]
        n_task = self.input_dict["n_task"]

        self.C_miss = T
        tau_1 = d**(4/3)*T**(1/3)
        tau_2 = m*np.sqrt(T)
        alpha = d/np.sqrt(tau_1)
        self.C_hit = tau_2+T*(m**2/tau_2 + alpha**2)

        self.expert_list = []
        for _ in range(input_dict["PMA_n_expert"]-1):
            B = ortho_group.rvs(dim=d)
            B = np.array(B)[:,:m]
            self.expert_list.append(B)
        self.expert_list.append(true_B) #The expert list always contain the true B

        self.p = self.input_dict["PMA_const"]*np.sqrt(T*d*m/(n_task*(d**(4/3)*T**(1/3)+d**(2/3)*T**(2/3))))
        self.p = min(self.p ,1)

        self.expert_losses = [0]*input_dict["PMA_n_expert"]
        self.lr = input_dict["PMA_lr_const"]*1
        self.reset()

    def select_ac(self, contexts):
        if self.is_EXR:
            return self.base_model(contexts)
        else:
            return self.base_model(self.B_hat.T @ contexts)

    def update(self, reward):
        self.base_model.update(reward)

    def check_alpha_cover(self, B):
        #TODO: implement here
        return False
        
    def reset(self):
        T = self.input_dict["T"]
        d = self.input_dict["d"]
        m = self.input_dict["m"]
        PMA_n_expert = self.input_dict["PMA_n_expert"]

        is_first_round = sum(self.expert_losses)==0
        if is_first_round:
            self.q = [1/PMA_n_expert]*PMA_n_expert # distribution to sample the expert
        else:
            if self.is_EXR: # Update self.q here
                for i in range(PMA_n_expert):
                    if self.check_alpha_cover(self.expert_list[i]):
                        C_i = self.C_hit
                    else:
                        C_i = self.C_miss
                    l_i = (C_i - self.C_hit)/self.p
                    self.expert_losses[i] += l_i
                tmp = np.copy(self.expert_losses)
                tmp = np.exp(tmp)
                self.q = tmp/sum(tmp)

        expert_idx = np.random.choice(PMA_n_expert, p=self.q)
        self.B_hat = self.expert_list(expert_idx)

        self.is_EXR = np.random.binomial(n=1, p=self.p)
        #TODO: change to PEGE (PHE?) with tau_1, tau_2 above
        if self.is_EXR:
            self.base_model = UCB(d=d, alpha=self.alpha, lam=self.lam)
        else:
            self.base_model = UCB(d=m, alpha=self.alpha, lam=self.lam)