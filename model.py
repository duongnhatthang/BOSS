import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve, minimize
from sklearn.linear_model import LogisticRegression
from scipy.stats import ortho_group
from evaluation import logger
from scipy.stats import entropy
from data import generate_contexts

## For quick update of Vinv
def sherman_morrison(X, V, w=1):
    result = V-(w*np.einsum('ij,j,k,kl -> il', V, X, X, V))/(1.+w*np.einsum('i,ij,j ->', X, V, X))
    return result

# For nice printing
def round(X, precision = .03):
    return np.round(np.array(X)/precision)*precision

'''
TS
'''
class TS:
    def __init__(self, d, v):
        ## Hyperparameters
        self.v=v
        self.others = {'v': self.v}
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

    def select_ctx_without_ctx(self):
        """
        Assuming unit ball action
        """
        V=(self.v**2)*self.Binv
        theta_tilde=np.random.multivariate_normal(self.theta_hat, V, size=1)
        self.X_a = theta_tilde/np.linalg.norm(theta_tilde)
        return self.X_a

    def update(self,reward):
        self.f=self.f+reward*self.X_a
        self.Binv = sherman_morrison(X=self.X_a, V=self.Binv)
        self.theta_hat=np.dot(self.Binv, self.f)

    def reset(self):
        self.theta_hat=np.zeros(self.d) + 1e-6
        self.f=np.zeros(self.d)
        self.Binv=np.eye(self.d)
        self.t = 0

'''
UCB
'''
class UCB:
    def __init__(self, d, alpha, unit_ball_action=True, lam=1):
        self.alpha=alpha
        self.d=d
        self.lam=lam
        self.others = {'alpha': self.alpha}
        if unit_ball_action:
            self.contexts = generate_contexts(n_gen_context=20, d=d) #TODO: random seed here if needed later
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

    def select_ctx_without_ctx(self):
        """
        Assuming unit ball action set
        """
        a_t = self.select_ac(self.contexts)
        return self.contexts[a_t]

    def update(self,reward):
        self.Binv = sherman_morrison(self.X_a, self.Binv)
        self.yx = self.yx+reward*self.X_a
        self.theta_hat = self.Binv @ self.yx
        # logger.info(f"theta_hat={round(self.theta_hat)}")

    def reset(self):
        self.yx=np.zeros(self.d)
        self.Binv=self.lam*np.eye(self.d)
        self.theta_hat = np.zeros(self.d) + 1e-6

'''
PHE (Perturbed-History Exploration ?)
'''
class PHE:
    def __init__(self, d, alpha, lam=1):
        self.alpha=alpha
        self.d=d
        self.lam=lam
        self.others = {'alpha': self.alpha}
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

    def select_ctx_without_ctx(self):
        """
        Assuming unit ball action set
        """
        self.X_a = self.theta_hat/np.linalg.norm(self.theta_hat)
        return self.X_a

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
        self.others = {'tau_1': self.tau_1}
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
        self.step += 1
        return self.X_a

    def select_ctx_without_ctx(self):
        """
        Assuming unit ball action set
        """
        if self.step <= self.tau_1:
            if self.EXR_contexts is None:
                self.X_a = np.random.uniform(low=-1, high=1, size=self.d)
                u = np.random.uniform(0,1) #Scaling factor
                self.X_a = u*self.X_a/np.linalg.norm(self.X_a) #ensure unit ball length
            else:
                idx = self.step % len(self.EXR_contexts)
                self.X_a = self.EXR_contexts[idx]
        else:
            self.X_a = self.theta_hat/np.linalg.norm(self.theta_hat)
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
PEGE_oracle
'''
class PEGE_oracle:
    def __init__(self, true_B, tau_1, lam=1):
        self.tau_1=tau_1
        self.true_B=true_B
        self.d, self.m = true_B.shape
        self.lam=lam
        self.others = {'tau_1': self.tau_1}
        self.EXR_contexts=[true_B[:,i] for i in range(self.m)]
        self.reset()

    def select_ctx(self, contexts):
        if self.step <= self.tau_1:
            idx = self.step % len(self.EXR_contexts)
            # self.X_a = self.EXR_contexts[idx]
            ctx_out = self.EXR_contexts[idx]
        else:
            scores = np.array([np.dot(X, self.theta_hat) for X in contexts])
            a_t = np.argmax(scores)
            # self.X_a = contexts[a_t]
            ctx_out = contexts[a_t]
        self.X_a = self.true_B.T @ ctx_out
        self.step += 1
        return ctx_out

    def select_ctx_without_ctx(self):
        """
        Assuming unit ball action set
        """
        if self.step <= self.tau_1:
            idx = self.step % len(self.EXR_contexts)
            ctx_out = self.EXR_contexts[idx]
        else:
            ctx_out = self.theta_hat/np.linalg.norm(self.theta_hat)
        self.X_a = self.true_B.T @ ctx_out
        self.step += 1
        return ctx_out

    def update(self,reward):
        if self.step <= self.tau_1:
            self.Binv = sherman_morrison(self.X_a, self.Binv)
            self.yx = self.yx+reward*self.X_a
            self.w_hat = self.Binv @ self.yx
            self.theta_hat = self.true_B @ self.w_hat

    def reset(self):
        self.yx=np.zeros(self.m)
        self.Binv=self.lam*np.eye(self.m)
        self.theta_hat = np.zeros(self.d)
        self.w_hat = np.zeros(self.m)
        self.step = 0

'''
UCB_oracle
'''
class UCB_oracle:
    def __init__(self, d, m, alpha, true_B, lam=1):
        self.alpha=alpha
        self.d=d
        self.m=m
        self.lam=lam
        self.true_B=true_B
        self.others = {'alpha': self.alpha}
        self.reset()

    def select_ac(self, contexts):
        means = np.array([np.dot(X, self.w_hat) for X in contexts])
        stds = np.array([np.sqrt(X.T @ self.Binv @ X) for X in contexts])
        ucbs = means + self.alpha*stds
        a_t = np.argmax(ucbs)
        self.X_a = contexts[a_t]
        return(a_t)

    def select_ctx(self, contexts):
        ctx_list = [self.true_B.T @ c for c in contexts]
        a_t = self.select_ac(ctx_list)
        return contexts[a_t]

    def update(self,reward):
        self.Binv = sherman_morrison(self.X_a, self.Binv)
        self.yx = self.yx+reward*self.X_a
        self.w_hat = self.Binv @ self.yx
        self.theta_hat = self.true_B @ self.w_hat

    def reset(self):
        self.yx=np.zeros(self.m)
        self.Binv=self.lam*np.eye(self.m)
        self.theta_hat = np.zeros(self.d)
        self.w_hat = np.zeros(self.m)
        self.step = 0

'''
BOSS_protocol
'''
class BOSS_protocol:
    def __init__(self, input_dict, tau_2_const, stop_exr):
        """
        Implement here
        """
        self.input_dict=input_dict
        self.stop_exr=stop_exr
        T = self.input_dict["T"]
        m = self.input_dict["m"]
        self.tau_2 = tau_2_const*m*np.sqrt(T)
        self.tau_2 = min(self.tau_2, T)
        assert self.tau_2 >= m, f"tau_2 ({round(self.tau_2)}) < m ({m})"
        self.task_idx = -1
        self.is_first_round = True

    def select_ctx(self, contexts):
        if self.is_EXR:
            return self.base_model.select_ctx(contexts)
        else:
            if self.base_model.step < self.tau_2:
                EXR_contexts = [self.B_hat[:,i] for i in range(self.input_dict["m"])]
                idx = self.base_model.step % len(EXR_contexts)
                self.X_a = EXR_contexts[idx]
                self.base_model.X_a = self.B_hat.T @ self.X_a
                self.base_model.step += 1
            else: #<w,u> = w.T B.T@B u = <Bw, Bu> = <theta, Bu>
                ctx_list = [self.B_hat.T @ c for c in contexts]
                out_ctx = self.base_model.select_ctx(ctx_list)
                self.X_a = self.B_hat @ out_ctx
            return self.X_a

    def select_ctx_without_ctx(self):
        if self.is_EXR:
            return self.base_model.select_ctx_without_ctx()
        else:
            if self.base_model.step < self.tau_2:
                EXR_contexts = [self.B_hat[:,i] for i in range(self.input_dict["m"])]
                idx = self.base_model.step % len(EXR_contexts)
                self.X_a = EXR_contexts[idx]
                self.base_model.X_a = self.B_hat.T @ self.X_a
                self.base_model.step += 1
            else: #<w,u> = w.T B.T@B u = <Bw, Bu> = <theta, Bu>
                out_ctx = self.base_model.select_ctx_without_ctx()
                self.X_a = self.B_hat @ out_ctx
            return self.X_a

    def update(self, reward):
        self.base_model.update(reward)
        if self.is_EXR:
            self.theta_hat = self.base_model.theta_hat
        else:
            w_hat = self.base_model.theta_hat
            self.theta_hat = self.B_hat @ w_hat

    def update_B_hat(self):
        """
        Implement here
        """
        pass

    def reset(self):
        d = self.input_dict["d"]
        m = self.input_dict["m"]

        # Decay Exr prob if needed
        self.task_idx += 1
        self.p = self.p/(1+self.input_dict["p_decay_rate"]*self.task_idx) # Decay EXR's prob overtime
        self.p = min(self.p ,1)
        if self.input_dict["p_decay_rate"] > 0:
            logger.info(f"Exp prob = {round(self.p)}")

        self.update_B_hat()

        # Choose EXR/EXT and base_model
        if self.is_first_round:
            self.is_EXR = True
        elif self.task_idx > self.stop_exr:
            self.is_EXR = False
        else:
            self.is_EXR = np.random.binomial(n=1, p=self.p)
        if self.is_first_round or self.is_EXR:
            self.base_model = PEGE(d=d, tau_1=self.tau_1, lam=self.lam)
        else:
            self.base_model = PEGE(d=m, tau_1=self.tau_2, lam=self.lam)
        self.is_first_round = False

        if self.input_dict["fixed_params"] is not None: #For comparison between BOSS and SeqRepL
            self.p, self.tau_1, self.tau_2 = self.input_dict["fixed_params"]
'''
PMA
'''
class PMA(BOSS_protocol):
    def __init__(self, input_dict, true_B, lam=1):
        super().__init__(input_dict, input_dict["PMA_tau2_const"], input_dict["PMA_stop_exr"])
        self.true_B=true_B #For debug
        self.lam=lam
        self.input_dict=input_dict
        T = self.input_dict["T"]
        d = self.input_dict["d"]
        m = self.input_dict["m"]
        n_task = self.input_dict["n_task"]
        self.others = []
        self.theta_hat_list = []

        self.C_miss = T
        self.tau_1 = self.input_dict["PMA_tau1_const"]*d**(4/3)*T**(1/3)
        self.tau_1 = min(self.tau_1, T)
        assert self.tau_1 >= d, f"tau_1 ({round(self.tau_1)}) < d ({d})"
        self.alpha = self.input_dict["PMA_alpha_const"]*d/np.sqrt(self.tau_1)
        self.C_hit = self.tau_2+T*(m**2/self.tau_2 + self.alpha**2)
        C_info = self.tau_1+T*(d**2/self.tau_1 + self.alpha**2)

        if self.C_hit > self.C_miss:
            c = np.ceil(self.C_hit/self.C_miss)
            self.C_miss*=c
        if C_info > self.C_miss:
            c = np.ceil(C_info/self.C_miss)
            self.C_miss*=c
        assert self.C_hit <= self.C_miss, f"self.C_hit ({round(self.C_hit)}) > self.C_miss ({self.C_miss})"
        assert self.C_hit <= C_info, f"self.C_hit ({round(self.C_hit)}) > C_info ({C_info})"
        assert C_info <= self.C_miss, f"C_info ({round(C_info)}) > self.C_miss ({self.C_miss})"

        self.expert_list = []
        for _ in range(input_dict["PMA_n_expert"]-1):
            B = ortho_group.rvs(dim=d)
            B = np.array(B)[:,:m]
            self.expert_list.append(B)
        if self.input_dict["PMA_no_oracle"]:
            B = ortho_group.rvs(dim=d)
            B = np.array(B)[:,:m]
            self.expert_list.append(B)
        else:
            self.expert_list.append(true_B) #The expert list always contain the true B

        self.expert_losses = [0]*input_dict["PMA_n_expert"]
        self.lr = input_dict["PMA_lr_const"]*1
        self.p = self.input_dict["PMA_exr_const"]*np.sqrt(T*d*m/(n_task*(d**(4/3)*T**(1/3)+d**(2/3)*T**(2/3))))
        self.reset()
        logger.info(f"PMA's exp prob = {round(self.p)}, tau_1 = {round(self.tau_1)}, tau_2 = {round(self.tau_2)}")

    def check_alpha_cover(self, B):
        w_opt = B.T @ self.theta_hat # OLS solution
        err = np.linalg.norm(self.theta_hat - B @ w_opt)
        # print(f"err={err}, alpha={self.alpha}")
        return err <= self.alpha

    def update_B_hat(self):
        PMA_n_expert = self.input_dict["PMA_n_expert"]
        # Update q (dist of expert)
        if self.is_first_round:
            self.q = [1/PMA_n_expert]*PMA_n_expert # distribution to sample the expert
        else:
            if self.is_EXR: # Update self.q here
                tmp = 0
                all_loss_equal = True
                for i in range(PMA_n_expert):
                    if self.check_alpha_cover(self.expert_list[i]):
                        C_i = self.C_hit
                    else:
                        C_i = self.C_miss
                    l_i = (C_i - self.C_hit)/self.p
                    self.expert_losses[i] += l_i
                    if tmp == 0:
                        tmp = l_i
                    elif all_loss_equal == True and tmp!=l_i:
                        all_loss_equal = False
                if all_loss_equal:
                    if l_i == 0:
                        logger.info(f"Warning: check_alpha_cover fails (covers all experts) l={l_i}. Decrease PMA_alpha_const.")
                    else:
                        logger.info(f"Warning: check_alpha_cover fails (rejects all experts) l={l_i}. Increase PMA_alpha_const.")
                else:
                    tmp0 = np.copy(self.expert_losses)
                    tmp1 = tmp0-min(tmp0) #prevent overflow
                    tmp2 = np.exp(-self.lr*tmp1)
                    self.q = tmp2/sum(tmp2)

        expert_idx = np.random.choice(PMA_n_expert, p=self.q)
        self.B_hat = self.expert_list[expert_idx]
        entropy_q = entropy(self.q)
        if entropy_q == entropy([1/PMA_n_expert]*PMA_n_expert):
            logger.info("Warning: PMA's expert dist is still uniform. Reduce PMA_alpha_const or increase PMA_tau1_const")
        if self.input_dict["PMA_no_oracle"]==False and self.q[-1]<0.9 and self.input_dict["PMA_n_expert"]<21:
            logger.info(f"entropy_q = {round(entropy_q)}, q_true={round(self.q[-1])}")

        if self.is_first_round is False:
            self.others.append(self.B_hat)
            self.theta_hat_list.append(self.theta_hat)

'''
SeqRepL
'''
class SeqRepL(BOSS_protocol):
    def __init__(self, input_dict, lam=1):
        super().__init__(input_dict, input_dict["SeqRepL_tau2_const"], input_dict["SeqRepL_stop_exr"])
        self.lam=lam
        self.others = []
        self.theta_hat_list = []
        self.input_dict=input_dict
        T = self.input_dict["T"]
        d = self.input_dict["d"]
        m = self.input_dict["m"]
        n_task = self.input_dict["n_task"]

        self.tau_1 = self.input_dict["SeqRepL_tau1_const"]*d*m*np.sqrt(T)
        self.tau_1 = min(self.tau_1, T)
        self.p = self.input_dict["SeqRepL_exr_const"]*np.sqrt(n_task)

        self.svd_matrix = np.zeros((d,d))
        self.reset()
        logger.info(f"SeqRepL's exp prob = {round(self.p)}, tau_1 = {round(self.tau_1)}, tau_2 = {round(self.tau_2)}")

    def update_B_hat(self):
        if self.is_first_round==False and self.is_EXR:
            tmp = np.array([self.theta_hat]) #shape = (1,d)
            self.svd_matrix += (tmp.T @ tmp)
        m = self.input_dict["m"]
        U, S, Vh = np.linalg.svd(self.svd_matrix, full_matrices=True)
        self.B_hat = U[:,:m]

        if self.is_first_round==False:
            self.others.append(self.B_hat)
            self.theta_hat_list.append(self.theta_hat)

    def reset(self):
        super().reset()
        if self.input_dict["SeqRepL_exr_list"] is not None:
            if self.task_idx in self.input_dict["SeqRepL_exr_list"]:
                self.is_EXR = True
                self.base_model = PEGE(d=self.input_dict["d"], tau_1=self.tau_1, lam=self.lam)
            else:
                self.is_EXR = False
                self.base_model = PEGE(d=self.input_dict["m"], tau_1=self.tau_2, lam=self.lam)