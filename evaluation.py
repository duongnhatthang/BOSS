import numpy as np
from tqdm import trange
from data import generate_contexts
import time, json, itertools
from scipy.stats import ortho_group
import logging
import sys

date_strftime_format = "%Y-%m-%y %H:%M:%S"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt=date_strftime_format,
)

# Creating an object
logger = logging.getLogger()
logger.disabled = True
from model import UCB, TS, PHE, PEGE, PMA, UCB_oracle, PEGE_oracle, SeqRepL, BARON

MODE_RANDOM = 0
MODE_ADVERSARY_RANDOMIZE = 1
MODE_ADV_TASK_DIVERSITY = 2
MODE_ADVERSARY = 3


def _eval_one_sim(input_dict, model, theta, sim_idx, elapsed_time, theta_err):
    T = input_dict["T"]
    d = input_dict["d"]
    rho = input_dict["rho"]
    noise_std = input_dict["noise_std"]
    seed = input_dict["seed"]
    unit_ball_action = input_dict["unit_ball_action"]
    opt_reward = []
    model_reward = []

    for t in range(T):
        # generate contexts
        if seed is not None:
            seed = seed + t + sim_idx
        if unit_ball_action:
            # optimal reward
            opt_a = theta / np.linalg.norm(theta)
            opt_reward.append(opt_a.T @ theta)

            # time
            start = time.time()
            X_t = model.select_ctx_unit_ball_action_set()
        else:
            n_gen_context = input_dict["n_gen_context"]
            contexts = generate_contexts(n_gen_context, d, rho, seed=seed)
            # optimal reward
            opt_reward.append(np.amax(np.array(contexts) @ theta))

            # time
            start = time.time()
            X_t = model.select_ctx(contexts)
        reward = np.dot(X_t, theta) + np.random.normal(0, noise_std, size=1)
        model_reward.append(np.dot(X_t, theta))
        model.update(reward)
        elapsed_time[sim_idx, t] = time.time() - start
        theta_err[sim_idx, t] = np.linalg.norm(model.theta_hat - theta)
    return opt_reward, model_reward, model.theta_hat


def _post_process(input_dict, results):
    output = input_dict["output"]
    d = input_dict["d"]
    name = input_dict["name"]
    if output:
        # Plotting
        last_regret = []
        for result in results:  # Iters over different params
            if name == "TS":
                v = result["others"]["v"]
            elif name == "PEGE":
                v = result["others"]["tau_1"]
            elif name == "UCB" or name == "PHE" or name == "UCB_oracle":
                alpha = result["others"]["alpha"]
            else:
                pass
            last_regret.append(np.mean(result["regrets"], axis=0)[-1])
        idx = np.argmin(last_regret)
        logger.info(f"idx of params chosen: {idx}")
        best = results[idx]
        return best
    else:
        # n_gen_context = input_dict["n_gen_context"]
        # Save to txt file
        # with open('./results/{%s}_d%d_N%d.txt' % (name, d, n_gen_context), 'w+') as outfile:
        #     json.dump(results, outfile)
        with open("./results/{%s}_d%d.txt" % (name, d), "w+") as outfile:
            json.dump(results, outfile)


def eval(input_dict):
    name = input_dict["name"]
    # n_gen_context = input_dict["n_gen_context"]
    params_set = input_dict["params_set"]
    T = input_dict["T"]
    d = input_dict["d"]
    n_sim = input_dict["n_sim"]
    seed = input_dict["seed"]
    results = []
    for param in params_set:
        cumul_regret = np.zeros((n_sim, T))
        theta_err = np.zeros((n_sim, T))
        elapsed_time = np.zeros((n_sim, T))
        for sim_idx in range(n_sim):
            print("%s Simulation %d, d=%d, param=%.3f" % (name, sim_idx + 1, d, param))
            # call model
            if name == "UCB":
                model = UCB(d=d, alpha=param)
            elif name == "TS":
                model = TS(d=d, v=param)
            elif name == "PHE":
                model = PHE(d=d, alpha=param)
            elif name == "PEGE":
                model = PEGE(d=d, tau_1=param)

            if seed is not None:
                np.random.seed(seed + sim_idx)
            # true theta
            theta = np.random.uniform(
                -1 / np.sqrt(d), 1 / np.sqrt(d), d
            )  # ensure unit ball length
            opt_reward, model_reward, theta_hat = _eval_one_sim(
                input_dict, model, theta, sim_idx, elapsed_time, theta_err
            )

            cumul_regret[sim_idx, :] = np.cumsum(opt_reward) - np.cumsum(model_reward)
        results.append(
            {
                "model": name,
                "settings": model.settings,
                "regrets": cumul_regret.tolist(),
                "theta_err": theta_err.tolist(),
                "time": elapsed_time.tolist(),
            }
        )
    return _post_process(input_dict, results)


def _cal_angle_err(theta, theta_hat):
    v1_u = theta / np.linalg.norm(theta)
    v2_u = theta_hat / np.linalg.norm(theta_hat)
    return 1 - np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


def eval_multi(input_dict):
    # inputs: n_sim, n_gen_context, d, T, rho, seed, B(bound for the theta)
    name = input_dict["name"]
    # n_gen_context = input_dict["n_gen_context"]
    params_set = input_dict["params_set"]
    T = input_dict["T"]
    d = input_dict["d"]
    n_sim = input_dict["n_sim"]
    seed = input_dict["seed"]
    m = input_dict["m"]
    n_task = input_dict["n_task"]

    results = []
    for param in params_set:
        cumul_regret = np.zeros((n_sim, n_task))
        B_hat_err = np.zeros((n_sim, n_task))
        theta_hat_err = np.zeros((n_sim, n_task))
        angle_err = np.zeros((n_sim, n_task))
        theta_err_all = np.zeros((n_sim, T, n_task))
        elapsed_time_all = np.zeros((n_sim, T, n_task))
        cumul_regret_all = np.zeros((n_sim, T, n_task))

        for sim_idx in range(n_sim):
            print("%s Simulation %d, d=%d" % (name, sim_idx + 1, d))
            if seed is not None:
                np.random.seed(seed + sim_idx)
            n_revealed = 0
            B = ortho_group.rvs(dim=d)
            B = np.array(B)[:, :m]

            # call model
            if name == "UCB":
                model = UCB(d=d, alpha=param)
            elif name == "TS":
                model = TS(d=d, v=param)
            elif name == "PHE":
                model = PHE(d=d, alpha=param)
            elif name == "PEGE":
                model = PEGE(d=d, tau_1=param)
            elif name == "PMA":
                model = PMA(input_dict=input_dict, true_B=B)
            elif name == "UCB_oracle":
                model = UCB_oracle(d=d, m=m, alpha=param, true_B=B)
            elif name == "PEGE_oracle":
                model = PEGE_oracle(true_B=B, tau_1=param)
            elif name == "SeqRepL":
                model = SeqRepL(input_dict=input_dict)
            elif name == "BARON":
                model = BARON(input_dict=input_dict)

            task_regret = np.zeros((n_task,))
            B_list = []
            theta_list = []
            for task_idx in trange(n_task):
                theta_err_i = theta_err_all[:, :, task_idx]
                elapsed_time_i = elapsed_time_all[:, :, task_idx]
                theta, n_revealed, B_n = gen_params(B, input_dict, task_idx, n_revealed)
                opt_reward, model_reward, theta_hat = _eval_one_sim(
                    input_dict, model, theta, sim_idx, elapsed_time_i, theta_err_i
                )
                task_regret[task_idx] = sum(opt_reward) - sum(model_reward)
                model.reset()
                B_list.append(B_n)
                theta_list.append(theta)
                cumul_regret_all[sim_idx, :, task_idx] = np.cumsum(
                    np.array(opt_reward) - np.array(model_reward)
                )
                angle_err[sim_idx, task_idx] = _cal_angle_err(theta, theta_hat)
            cumul_regret[sim_idx, :] = np.cumsum(task_regret)
            if name == "SeqRepL" or name == "PMA" or name == "BARON":
                for i, B_hat in enumerate(model.others):
                    B_perp_B_perp_transpose = np.eye(d) - B_hat @ B_hat.T
                    U, S, Vh = np.linalg.svd(
                        B_perp_B_perp_transpose, full_matrices=True
                    )
                    B_perp = U[:, : d - m]
                    B_hat_err[sim_idx, i] = np.linalg.norm(B_perp.T @ B_list[i])
                for i, theta_hat_hat in enumerate(model.theta_hat_list):
                    theta_hat_err[sim_idx, i] = np.linalg.norm(
                        theta_hat_hat - theta_list[i]
                    )
        results.append(
            {
                "model": name,
                "others": model.others,
                "B_hat_err": B_hat_err,
                "theta_hat_err": theta_hat_err,
                "regrets": cumul_regret,
                "regrets_all": cumul_regret_all,
                "theta_err": theta_err_all,
                "angle_err": angle_err,
                "time": elapsed_time_all,
            }
        )
    return _post_process(input_dict, results)


def gen_params(B, input_dict, task_idx, n_revealed):
    def _gen_params_from_B(B, m):
        w_i = np.random.uniform(-1, 1, m)
        u = np.random.uniform(0.8, 1)  # Scaling factor
        theta = B @ w_i
        theta = u * theta / np.linalg.norm(theta)  # ensure unit ball length
        return theta, u * w_i

    T = input_dict["T"]
    d = input_dict["d"]
    m = input_dict["m"]
    n_task = input_dict["n_task"]
    adv_exr_const = input_dict["adv_exr_const"]
    mode = input_dict["mode"]
    if mode == MODE_RANDOM:
        theta = _gen_params_from_B(B, m)
        low_rank_B = B
    else:
        if mode == MODE_ADVERSARY_RANDOMIZE:
            q = (
                adv_exr_const
                * (d - m)
                / ((np.sqrt(T) - m) * (1 + np.sqrt(2 * (n_task - task_idx - 1))))
            )  # probability of revealing a new dimension
            q = min(q, 1)
        elif mode == MODE_ADV_TASK_DIVERSITY or mode == MODE_ADVERSARY:
            q = 1
        if input_dict["adv_exr_task"] is None:
            if n_revealed < m:
                reveal_new = np.random.binomial(n=1, p=q)
                if reveal_new == 1 or n_revealed == 0:
                    n_revealed = min(n_revealed + 1, m)
                    logger.info(
                        f"==== [Adv reveals] {n_revealed}/{m} dims with prob q = {q}"
                    )
        else:  # Reveal at tasks in input_dict["adv_exr_task"]
            if task_idx in input_dict["adv_exr_task"]:
                n_revealed = min(n_revealed + 1, m)
                logger.info(
                    f"==== [Adv reveals] {n_revealed}/{m} dims with prob q = {q}"
                )
        low_rank_B = np.copy(B)
        low_rank_B[:, n_revealed:] = 0
        theta, w_i = _gen_params_from_B(low_rank_B, m)
        # input_dict["theta"]=theta #For debug only
        # input_dict["w_i"]=w_i #For debug only
        if (
            mode == MODE_ADVERSARY
        ):  # if MODE_ADVERSARY, only reveal the first theta when SeqRepL explore
            assert (
                input_dict["SeqRepL_exr_list"] is not None
            ), "Not init input_dict['SeqRepL_exr_list']"
            low_rank_B_tmp = np.copy(low_rank_B)
            if task_idx in input_dict["SeqRepL_exr_list"]:
                low_rank_B_tmp[:, 1:] = 0
            else:
                if n_revealed > 1:
                    low_rank_B_tmp[:, 0] = 0
            theta, w_i = _gen_params_from_B(low_rank_B_tmp, m)

    return theta, n_revealed, low_rank_B
