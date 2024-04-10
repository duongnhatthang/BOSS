import numpy as np
from tqdm import trange
from train import UCB, TS, PHE
from data import generate_contexts
import time, json, itertools

def eval_UCB(N, d, alpha_set=[0.001, 0.01, 0.1, 1], T=30000, M=10, rho=0.5, R=1, seed=0, output=False):
    #evaluate UCB
    #inputs: M, N, d, T, rho, seed, B(bound for the beta)
    results = []
    for alpha in alpha_set:
        cumul_regret = np.zeros((M,T))
        beta_err = np.zeros((M,T))
        elapsed_time = np.zeros((M,T))
        for m in range(M):
            print('UCB Simulation %d, N=%d, d=%d, alpha=%.3f' % (m+1, N, d, alpha))
            # call model
            M_UCB = UCB(d=d, alpha=alpha)
            # true beta
            np.random.seed(seed+m)
            #beta = np.random.uniform(-1,1,d)
            beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d)
            opt_reward = []
            UCB_reward = []

            for t in trange(T):
                # generate contexts
                contexts = generate_contexts(N, d, rho, seed=seed+t+m)
                # optimal reward
                opt_reward.append(np.amax(np.array(contexts) @ beta))
                # time
                start = time.time()
                a_t = M_UCB.select_ac(contexts)
                reward = np.dot(contexts[a_t],beta) + np.random.normal(0, R, size=1)
                UCB_reward.append(np.dot(contexts[a_t],beta))
                M_UCB.update(reward)
                elapsed_time[m,t] = time.time() - start
                beta_err[m,t] = np.linalg.norm(M_UCB.beta_hat-beta)

            cumul_regret[m,:] = np.cumsum(opt_reward)-np.cumsum(UCB_reward)
        ##Save at dict
        results.append({'model':'UCB',
                        'settings':M_UCB.settings,
                        'regrets':cumul_regret.tolist(),
                        'beta_err':beta_err.tolist(),
                        'time':elapsed_time.tolist()})
    if output:
        # Plotting
        last_regret = []
        for result in results:
            alpha = result['settings']['alpha']
            last_regret.append(np.mean(result['regrets'], axis=0)[-1])
        best_UCB = results[np.argmin(last_regret)]
        return best_UCB
    else:
        # Save to txt file
        with open('./results/UCB_d%d_N%d.txt' % (d, N), 'w+') as outfile:
            json.dump(results, outfile)


def eval_TS(N, d, v_set=[0.001, 0.01, 0.1, 1], T=30000, M=10, rho=0.5, R=1, seed=0, output=False):
    #evaluate TS
    #inputs: M, N, d, T, rho, seed, B(bound for the beta)
    results = []
    for v in v_set:
        cumul_regret = np.zeros((M,T))
        beta_err = np.zeros((M,T))
        elapsed_time = np.zeros((M,T))
        for m in range(M):
            print('TS Simulation %d, N=%d, d=%d, v=%.3f' % (m+1, N, d, v))
            # call model
            M_TS = TS(d=d, v=v)
            # true beta
            np.random.seed(seed+m)
            #beta = np.random.uniform(-1,1,d)
            beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d)
            opt_reward = []
            TS_reward = []

            for t in trange(T):
                # generate contexts
                contexts = generate_contexts(N, d, rho, seed=seed+t+m)
                # optimal reward
                opt_reward.append(np.amax(np.array(contexts) @ beta))
                # time
                start = time.time()
                a_t = M_TS.select_ac(contexts)
                reward = np.dot(contexts[a_t],beta) + np.random.normal(0, R, size=1)
                TS_reward.append(np.dot(contexts[a_t],beta))
                M_TS.update(reward)
                elapsed_time[m,t] = time.time() - start
                beta_err[m,t] = np.linalg.norm(M_TS.beta_hat-beta)

            cumul_regret[m,:] = np.cumsum(opt_reward)-np.cumsum(TS_reward)
        ##Save at dict
        results.append({'model':'TS',
                        'settings':M_TS.settings,
                        'regrets':cumul_regret.tolist(),
                        'beta_err':beta_err.tolist(),
                        'time':elapsed_time.tolist()})
        
    if output:
        # Plotting
        last_regret = []
        for result in results:
            v = result['settings']['v']
            last_regret.append(np.mean(result['regrets'], axis=0)[-1])
        best_TS = results[np.argmin(last_regret)]
        return best_TS
    else:
        ##ave to txt file
        with open('./results/TS_d%d_N%d.txt' % (d, N), 'w+') as outfile:
            json.dump(results, outfile)


def eval_PHE(N, d, alpha_set=[0.001, 0.01, 0.1, 1], T=30000, M=10, rho=0.5, R=1, seed=0, output=False):
    #evaluate PHE
    #inputs: M, N, d, T, rho, seed, B(bound for the beta)
    results = []
    for alpha in alpha_set:
        cumul_regret = np.zeros((M,T))
        beta_err = np.zeros((M,T))
        elapsed_time = np.zeros((M,T))
        for m in range(M):
            print('PHE Simulation %d, N=%d, d=%d, alpha=%.3f' % (m+1, N, d, alpha))
            # call model
            M_PHE = PHE(d=d, alpha=alpha)
            # true beta
            np.random.seed(seed+m)
            #beta = np.random.uniform(-1,1,d)
            beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d)
            opt_reward = []
            PHE_reward = []

            for t in trange(T):
                # generate contexts
                contexts = generate_contexts(N, d, rho, seed=seed+t+m)
                # optimal reward
                opt_reward.append(np.amax(np.array(contexts) @ beta))
                # time
                start = time.time()
                a_t = M_PHE.select_ac(contexts)
                reward = np.dot(contexts[a_t],beta) + np.random.normal(0, R, size=1)
                PHE_reward.append(np.dot(contexts[a_t],beta))
                M_PHE.update(reward)
                elapsed_time[m,t] = time.time() - start
                beta_err[m,t] = np.linalg.norm(M_PHE.beta_hat-beta)

            cumul_regret[m,:] = np.cumsum(opt_reward)-np.cumsum(PHE_reward)
        ##Save at dict
        results.append({'model':'PHE',
                        'settings':M_PHE.settings,
                        'regrets':cumul_regret.tolist(),
                        'beta_err':beta_err.tolist(),
                        'time':elapsed_time.tolist()})
    if output:
        # Plotting
        last_regret = []
        for result in results:
            alpha = result['settings']['alpha']
            last_regret.append(np.mean(result['regrets'], axis=0)[-1])
        best_PHE = results[np.argmin(last_regret)]
        return best_PHE
    else:
        # Save to txt file
        with open('./results/PHE_d%d_N%d.txt' % (d, N), 'w+') as outfile:
            json.dump(results, outfile)