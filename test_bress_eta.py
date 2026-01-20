"""
Quick test to compare BRESS with different eta (learning rate) values.

This script runs a small-scale experiment to see how different eta values
affect BRESS performance compared to other algorithms.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings

from evaluation import (
    eval_multi,
    logger,
    MODE_ADVERSARY,
)

# Disable logger for cleaner output
logger.disabled = True

# Suppress warnings
warnings.filterwarnings('ignore')


def get_SeqRepL_exr_list(n_task):
    """Generate exploration task list (square schedule)."""
    SeqRepL_exr_list = []
    i = 0
    while i**2 <= n_task:
        SeqRepL_exr_list.append(i**2)
        i += 1
    return SeqRepL_exr_list


def run_eta_comparison():
    """
    Run comparison of BRESS with different eta values.
    Uses small parameters for quick testing.
    """
    print("=" * 60)
    print("BRESS Eta Comparison Test")
    print("=" * 60)
    
    # Parameters for meaningful comparison
    T = 200  # Timesteps per task
    n_task = 100  # Number of tasks
    d = 8  # Ambient dimension
    m = 3  # Subspace dimension
    n_sim = 3  # Number of simulations
    
    # Base input_dict
    input_dict = {
        "d": d,
        "unit_ball_action": True,
        "T": T,
        "n_sim": n_sim,
        "rho": 0.5,
        "noise_std": 1,
        "seed": 42,
        "output": True,
        "params_set": [None],
        "m": m,
        "n_task": n_task,
        "mode": MODE_ADVERSARY,
        "adv_exr_const": 0.1,
        "adv_exr_task": [0, 25, 50],  # Tasks where adversary reveals new dimension (must match m)
        "p_decay_rate": 0,
        
        # SeqRepL parameters
        "SeqRepL_exr_const": 1.5,
        "SeqRepL_tau1_const": 1,
        "SeqRepL_tau2_const": 1,
        "SeqRepL_stop_exr": n_task,
        
        # BRESS parameters
        "OnlinePCA_exr_const": 1.5,
        "OnlinePCA_tau1_const": 1,
        "OnlinePCA_tau2_const": 1,
        "OnlinePCA_stop_exr": n_task,
    }
    
    # Setup exploration list
    SeqRepL_exr_list = get_SeqRepL_exr_list(n_task)
    input_dict["SeqRepL_exr_list"] = SeqRepL_exr_list
    input_dict["fixed_params"] = [
        5 * len(SeqRepL_exr_list) / n_task,  # p
        min(150, T // 2),  # tau1
        min(50, T // 4),   # tau2
    ]
    
    print(f"\nParameters: d={d}, m={m}, T={T}, n_task={n_task}, n_sim={n_sim}")
    print(f"fixed_params={input_dict['fixed_params']}")
    print()
    
    results = {}
    
    # Test PEGE (no representation learning)
    print("-" * 40)
    print("Running PEGE (no representation learning)...")
    print("-" * 40)
    input_dict["name"] = "PEGE"
    input_dict["params_set"] = [T // 2]  # tau_1
    input_dict["OnlinePCA_eta"] = None
    try:
        results["PEGE"] = eval_multi(input_dict)
        final_regret = np.mean(results['PEGE']['regrets'], axis=0)[-1]
        print(f"  Final cumulative regret: {final_regret:.2f}")
    except Exception as e:
        print(f"  PEGE FAILED: {e}")
        results["PEGE"] = None
    
    # Test SeqRepL (baseline)
    print("-" * 40)
    print("Running SeqRepL (baseline)...")
    print("-" * 40)
    input_dict["name"] = "SeqRepL"
    input_dict["params_set"] = [None]
    input_dict["OnlinePCA_eta"] = None  # Reset eta
    try:
        results["SeqRepL"] = eval_multi(input_dict)
        final_regret = np.mean(results['SeqRepL']['regrets'], axis=0)[-1]
        print(f"  Final cumulative regret: {final_regret:.2f}")
    except Exception as e:
        print(f"  SeqRepL FAILED: {e}")
        results["SeqRepL"] = None
    
    # Different eta values to test
    # Default eta would be sqrt(2 * log(d) / expected_n_exr_tasks)
    # expected_n_exr_tasks ~ p * n_task
    p = input_dict["fixed_params"][0]
    expected_n_exr = max(1, int(p * n_task))
    default_eta = np.sqrt(2 * np.log(d) / expected_n_exr)
    
    print(f"\nDefault eta would be approximately: {default_eta:.4f}")
    
    # Test different eta scales (fewer values for faster testing)
    eta_scales = [0.5, 1.0, 2.0, 5.0]
    eta_values = [default_eta * scale for scale in eta_scales]
    
    for scale, eta in zip(eta_scales, eta_values):
        print("-" * 40)
        print(f"Running BRESS with eta={eta:.4f} (scale={scale}x)...")
        print("-" * 40)
        input_dict["name"] = "BRESS"
        input_dict["OnlinePCA_eta"] = eta
        try:
            key = f"BRESS_eta_{scale}x"
            results[key] = eval_multi(input_dict)
            final_regret = np.mean(results[key]['regrets'], axis=0)[-1]
            print(f"  Final cumulative regret: {final_regret:.2f}")
        except Exception as e:
            print(f"  BRESS (eta={eta:.4f}) FAILED: {e}")
            results[f"BRESS_eta_{scale}x"] = None
    
    # Also test BRESS with default eta (T-based)
    print("-" * 40)
    print("Running BRESS with default eta (T-based)...")
    print("-" * 40)
    input_dict["name"] = "BRESS"
    input_dict["OnlinePCA_eta"] = None  # Use T-based calculation
    try:
        results["BRESS_default"] = eval_multi(input_dict)
        final_regret = np.mean(results['BRESS_default']['regrets'], axis=0)[-1]
        print(f"  Final cumulative regret: {final_regret:.2f}")
    except Exception as e:
        print(f"  BRESS (default) FAILED: {e}")
        results["BRESS_default"] = None
    
    # Plot results
    plot_comparison(results, n_task, T)
    
    # Print summary
    print_summary(results)
    
    return results


def plot_comparison(results, n_task, T):
    """Plot comparison of cumulative regrets."""
    X = np.arange(1, n_task + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cumulative regret over tasks
    ax = axes[0]
    for name, result in results.items():
        if result is not None:
            mean = np.mean(result["regrets"], axis=0)
            std = np.std(result["regrets"], axis=0)
            ax.plot(X, mean, label=name)
            ax.fill_between(X, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("# of tasks")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Cumulative Regret over Tasks")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: B_hat error (subspace estimation)
    ax = axes[1]
    for name, result in results.items():
        if result is not None and "B_hat_err" in result:
            mean = np.mean(result["B_hat_err"], axis=0)
            std = np.std(result["B_hat_err"], axis=0)
            # Only plot non-zero values
            valid_idx = mean > 0
            if np.any(valid_idx):
                ax.plot(X[valid_idx], mean[valid_idx], label=name)
                ax.fill_between(X[valid_idx], 
                               (mean - std)[valid_idx], 
                               (mean + std)[valid_idx], alpha=0.2)
    ax.set_xlabel("# of tasks")
    ax.set_ylabel("B_hat Error")
    ax.set_title("Subspace Estimation Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/bress_eta_comparison.png", dpi=150)
    print(f"\nPlot saved to figures/bress_eta_comparison.png")
    plt.close()


def print_summary(results):
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Final Regret':<15} {'Final B_hat Err':<15}")
    print("-" * 55)
    
    for name, result in results.items():
        if result is not None:
            final_regret = np.mean(result["regrets"], axis=0)[-1]
            if "B_hat_err" in result:
                b_err = np.mean(result["B_hat_err"], axis=0)
                # Find last non-zero B_hat error
                valid_idx = b_err > 0
                if np.any(valid_idx):
                    final_b_err = b_err[valid_idx][-1]
                else:
                    final_b_err = 0.0
            else:
                final_b_err = float('nan')
            print(f"{name:<25} {final_regret:<15.2f} {final_b_err:<15.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    results = run_eta_comparison()

