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

def _post_process(output, results, n_gen_context, d, name):
    if output:
        # Plotting
        last_regret = []
        for result in results:
            if name == 'TS':
                v = result['settings']['v']
            else:
                alpha = result['settings']['alpha']
            last_regret.append(np.mean(result['regrets'], axis=0)[-1])
        best = results[np.argmin(last_regret)]
        return best
    else:
        # Save to txt file
        with open('./results/{%s}_d%d_N%d.txt' % (name, d, n_gen_context), 'w+') as outfile:
            json.dump(results, outfile)

def eval(name, n_gen_context, d, params_set=[0.001, 0.01, 0.1, 1], T=30000, n_sim=10, rho=0.5, noise_std=1, seed=0, output=False, beta = None):
    #inputs: n_sim, n_gen_context, d, T, rho, seed, B(bound for the beta)
    results = []
    for param in params_set:
        cumul_regret = np.zeros((n_sim,T))
        beta_err = np.zeros((n_sim,T))
        elapsed_time = np.zeros((n_sim,T))
        for sim_idx in range(n_sim):
            print('%s Simulation %d, N=%d, d=%d, alpha=%.3f' % (name, sim_idx+1, n_gen_context, d, param))
            # call model
            if name=="UCB":
                model = UCB(d=d, alpha=param)
            elif name=="TS":
                model = TS(d=d, v=param)
            elif name=="PHE":
                model = PHE(d=d, alpha=param)
            # true beta
            np.random.seed(seed+sim_idx)
            #beta = np.random.uniform(-1,1,d)
            if beta is None:
                beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d)
            opt_reward, UCB_reward = _eval_one_sim(model, beta, n_gen_context, d, T, sim_idx, elapsed_time, beta_err, rho, noise_std, seed)

            cumul_regret[sim_idx,:] = np.cumsum(opt_reward)-np.cumsum(UCB_reward)
        ##Save at dict
        results.append({'model':name,
                        'settings':model.settings,
                        'regrets':cumul_regret.tolist(),
                        'beta_err':beta_err.tolist(),
                        'time':elapsed_time.tolist()})
    return _post_process(output, results, n_gen_context, d, name)