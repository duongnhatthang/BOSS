import numpy as np
from tqdm import trange
from train import UCB, TS, PHE
from data import generate_contexts
import time, json, itertools

def _eval_one_sim(model, beta, n_gen_context, d, T, sim_idx, elapsed_time, beta_err, rho=0.5, noise_std=1, seed=0):
    opt_reward = []
    model_reward = []

    for t in trange(T):
        # generate contexts
        contexts = generate_contexts(n_gen_context, d, rho, seed=seed+t+sim_idx)
        # optimal reward
        opt_reward.append(np.amax(np.array(contexts) @ beta))
        # time
        start = time.time()
        a_t = model.select_ac(contexts)
        reward = np.dot(contexts[a_t],beta) + np.random.normal(0, noise_std, size=1)
        model_reward.append(np.dot(contexts[a_t],beta))
        model.update(reward)
        elapsed_time[sim_idx,t] = time.time() - start
        beta_err[sim_idx,t] = np.linalg.norm(model.beta_hat-beta)
    return opt_reward, model_reward

def eval_UCB(n_gen_context, d, alpha_set=[0.001, 0.01, 0.1, 1], T=30000, n_sim=10, rho=0.5, noise_std=1, seed=0, output=False):
    #evaluate UCB
    #inputs: n_sim, n_gen_context, d, T, rho, seed, B(bound for the beta)
    results = []
    for alpha in alpha_set:
        cumul_regret = np.zeros((n_sim,T))
        beta_err = np.zeros((n_sim,T))
        elapsed_time = np.zeros((n_sim,T))
        for sim_idx in range(n_sim):
            print('UCB Simulation %d, N=%d, d=%d, alpha=%.3f' % (sim_idx+1, n_gen_context, d, alpha))
            # call model
            M_UCB = UCB(d=d, alpha=alpha)
            # true beta
            np.random.seed(seed+sim_idx)
            #beta = np.random.uniform(-1,1,d)
            beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d)
            opt_reward, UCB_reward = _eval_one_sim(M_UCB, beta, n_gen_context, d, T, sim_idx, elapsed_time, beta_err, rho, noise_std, seed)

            cumul_regret[sim_idx,:] = np.cumsum(opt_reward)-np.cumsum(UCB_reward)
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
        with open('./results/UCB_d%d_N%d.txt' % (d, n_gen_context), 'w+') as outfile:
            json.dump(results, outfile)


def eval_TS(n_gen_context, d, v_set=[0.001, 0.01, 0.1, 1], T=30000, n_sim=10, rho=0.5, noise_std=1, seed=0, output=False):
    #evaluate TS
    #inputs: n_sim, n_gen_context, d, T, rho, seed, B(bound for the beta)
    results = []
    for v in v_set:
        cumul_regret = np.zeros((n_sim,T))
        beta_err = np.zeros((n_sim,T))
        elapsed_time = np.zeros((n_sim,T))
        for sim_idx in range(n_sim):
            print('TS Simulation %d, N=%d, d=%d, v=%.3f' % (sim_idx+1, n_gen_context, d, v))
            # call model
            M_TS = TS(d=d, v=v)
            # true beta
            np.random.seed(seed+sim_idx)
            #beta = np.random.uniform(-1,1,d)
            beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d)
            opt_reward, TS_reward = _eval_one_sim(M_TS, beta, n_gen_context, d, T, sim_idx, elapsed_time, beta_err, rho, noise_std, seed)

            cumul_regret[sim_idx,:] = np.cumsum(opt_reward)-np.cumsum(TS_reward)
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
        with open('./results/TS_d%d_N%d.txt' % (d, n_gen_context), 'w+') as outfile:
            json.dump(results, outfile)


def eval_PHE(n_gen_context, d, alpha_set=[0.001, 0.01, 0.1, 1], T=30000, n_sim=10, rho=0.5, noise_std=1, seed=0, output=False):
    #evaluate PHE
    #inputs: n_sim, n_gen_context, d, T, rho, seed, B(bound for the beta)
    results = []
    for alpha in alpha_set:
        cumul_regret = np.zeros((n_sim,T))
        beta_err = np.zeros((n_sim,T))
        elapsed_time = np.zeros((n_sim,T))
        for sim_idx in range(n_sim):
            print('PHE Simulation %d, N=%d, d=%d, alpha=%.3f' % (sim_idx+1, n_gen_context, d, alpha))
            # call model
            M_PHE = PHE(d=d, alpha=alpha)
            # true beta
            np.random.seed(seed+sim_idx)
            #beta = np.random.uniform(-1,1,d)
            beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d)
            opt_reward, PHE_reward = _eval_one_sim(M_PHE, beta, n_gen_context, d, T, sim_idx, elapsed_time, beta_err, rho, noise_std, seed)

            cumul_regret[sim_idx,:] = np.cumsum(opt_reward)-np.cumsum(PHE_reward)
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
        with open('./results/PHE_d%d_N%d.txt' % (d, n_gen_context), 'w+') as outfile:
            json.dump(results, outfile)