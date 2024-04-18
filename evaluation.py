import numpy as np
from tqdm import trange
from model import UCB, TS, PHE
from data import generate_contexts
import time, json, itertools

# def _eval_one_sim(model, beta, n_gen_context, d, T, sim_idx, elapsed_time, beta_err, rho=0.5, noise_std=1, seed=0):
def _eval_one_sim(input_dict, model,beta, sim_idx, elapsed_time, beta_err):
    n_gen_context = input_dict["n_gen_context"]
    T = input_dict["T"]
    d = input_dict["d"]
    rho = input_dict["rho"]
    noise_std = input_dict["noise_std"]
    seed = input_dict["seed"]
    opt_reward = []
    model_reward = []

    for t in range(T):
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

# def _post_process(output, results, n_gen_context, d, name):
def _post_process(input_dict, results):
    output = input_dict["output"]
    d = input_dict["d"]
    name = input_dict["name"]
    n_gen_context = input_dict["n_gen_context"]
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

def eval(input_dict):
    name = input_dict["name"]
    n_gen_context = input_dict["n_gen_context"]
    params_set = input_dict["params_set"]
    T = input_dict["T"]
    d = input_dict["d"]
    n_sim = input_dict["n_sim"]
    seed = input_dict["seed"]
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
            np.random.seed(seed+sim_idx)
            # true beta
            beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d) #ensure unit ball length
            opt_reward, model_reward = _eval_one_sim(input_dict, model, beta, sim_idx, elapsed_time, beta_err)

            cumul_regret[sim_idx,:] = np.cumsum(opt_reward)-np.cumsum(model_reward)
        ##Save at dict
        results.append({'model':name,
                        'settings':model.settings,
                        'regrets':cumul_regret.tolist(),
                        'beta_err':beta_err.tolist(),
                        'time':elapsed_time.tolist()})
    return _post_process(input_dict, results)


from scipy.stats import ortho_group
def eval_multi(input_dict):
    #inputs: n_sim, n_gen_context, d, T, rho, seed, B(bound for the beta)
    name = input_dict["name"]
    n_gen_context = input_dict["n_gen_context"]
    params_set = input_dict["params_set"]
    T = input_dict["T"]
    d = input_dict["d"]
    n_sim = input_dict["n_sim"]
    seed = input_dict["seed"]
    m = input_dict["m"]
    n_task = input_dict["n_task"]

    B = ortho_group.rvs(dim=d)
    B = np.array(B)[:,:m]
    results = []
    for param in params_set:
        cumul_regret = np.zeros((n_sim,T,n_task))
        beta_err = np.zeros((n_sim,T,n_task))
        elapsed_time = np.zeros((n_sim,T,n_task))
        for sim_idx in range(n_sim):
            print('%s Simulation %d, N_gen_ctx=%d, d=%d, alpha=%.3f' % (name, sim_idx+1, n_gen_context, d, param))
            # call model
            if name=="UCB":
                model = UCB(d=d, alpha=param)
            elif name=="TS":
                model = TS(d=d, v=param)
            elif name=="PHE":
                model = PHE(d=d, alpha=param)
            np.random.seed(seed+sim_idx)
            for task_idx in trange(n_task):
                beta_err_i = beta_err[:,:,task_idx]
                elapsed_time_i = elapsed_time[:,:,task_idx]
                # true beta
                w_i = np.random.uniform(-1,1,m)
                u = np.random.uniform(0,1) #Scaling factor
                beta = B @ w_i
                beta = u*beta/np.linalg.norm(beta) #ensure unit ball length
                opt_reward, model_reward = _eval_one_sim(input_dict, model, beta, sim_idx, elapsed_time_i, beta_err_i)
                cumul_regret[sim_idx,:,task_idx] = np.cumsum(opt_reward)-np.cumsum(model_reward)
                model.reset()
        ##Save at dict
        results.append({'model':name,
                        'settings':model.settings,
                        'regrets':cumul_regret,
                        'beta_err':beta_err,
                        'time':elapsed_time})
    return _post_process(input_dict, results)