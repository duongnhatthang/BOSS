"""
Test script for BRESS algorithm.
Based on "Without Task Diversity" section of exp.ipynb.

This script tests the new BRESS wrapper that uses Warmuth's 
Uncentered Online PCA algorithm for subspace estimation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - must be before pyplot import
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

from evaluation import (
    eval_multi,
    logger,
    MODE_ADVERSARY,
)

# Disable logger for cleaner output
logger.disabled = True

# Suppress trace normalization warnings from warmuth_algorithms
warnings.filterwarnings('ignore', message='.*trace.*normalizing.*')


def get_SeqRepL_exr_list(n_task):
    """Generate exploration task list (square schedule)."""
    SeqRepL_exr_list = []
    i = 0
    while i**2 <= n_task:
        SeqRepL_exr_list.append(i**2)
        i += 1
    return SeqRepL_exr_list


def check_params(input_dict):
    """Validate input parameters."""
    m = input_dict["m"]
    d = input_dict["d"]
    T = input_dict["T"]
    n_task = input_dict["n_task"]
    
    assert m <= d, f"m ({m}) >= d ({d})"
    if T < d**2:
        print(f"Warning: T ({T}) < d^2 ({d**2})")
    if n_task < np.sqrt(T):
        print(f"Warning: n_task ({n_task}) < sqrt(T) ({np.sqrt(T):.1f})")
    if input_dict["adv_exr_task"] is not None:
        assert max(input_dict["adv_exr_task"]) < n_task, \
            f"max exr task ({max(input_dict['adv_exr_task'])}) >= n_task ({n_task})"
        assert len(input_dict["adv_exr_task"]) == m, \
            f"exr list len ({len(input_dict['adv_exr_task'])}) != m ({m})"


def run_test_small():
    """
    Run a small test to verify BRESS works.
    Uses reduced parameters for quick testing.
    """
    print("=" * 60)
    print("Running SMALL TEST for BRESS")
    print("=" * 60)
    
    # Small parameters for quick testing
    T = 200  # Timesteps per task (reduced from 2000)
    n_task = 50  # Number of tasks (reduced from 6000)
    
    input_dict = {
        "d": 5,  # Ambient dimension (reduced from 10)
        "unit_ball_action": True,
        "T": T,
        "n_sim": 2,  # Number of simulations (reduced from 5)
        "rho": 0.5,
        "noise_std": 1,
        "seed": 42,  # Fixed seed for reproducibility
        "output": True,
        "params_set": [None],  # Placeholder
        "m": 2,  # Subspace dimension (reduced from 3)
        "n_task": n_task,
        "mode": MODE_ADVERSARY,
        "adv_exr_const": 0.1,
        "adv_exr_task": [0, 10, 25],  # Tasks where adversary reveals new dimension
        "p_decay_rate": 0,
        
        # PMA parameters (for comparison if needed)
        "PMA_exr_const": 1.5,
        "PMA_lr_const": 1,
        "PMA_n_expert": 100,  # Reduced for testing
        "PMA_tau1_const": 1,
        "PMA_tau2_const": 1,
        "PMA_alpha_const": 1,
        "PMA_stop_exr": n_task,
        "PMA_no_oracle": True,
        
        # SeqRepL parameters
        "SeqRepL_exr_const": 1.5,
        "SeqRepL_tau1_const": 1,
        "SeqRepL_tau2_const": 1,
        "SeqRepL_stop_exr": n_task,
        "fixed_params": None,
        "SeqRepL_exr_list": None,
        
        # BRESS parameters
        "OnlinePCA_exr_const": 1.5,
        "OnlinePCA_tau1_const": 1,
        "OnlinePCA_tau2_const": 1,
        "OnlinePCA_stop_exr": n_task,
    }
    
    # Setup exploration list
    SeqRepL_exr_list = get_SeqRepL_exr_list(n_task)
    input_dict["SeqRepL_exr_list"] = SeqRepL_exr_list
    input_dict["adv_exr_task"] = [0, n_task // 4, n_task // 2][:input_dict["m"]]
    
    # Set fixed params based on exploration list
    input_dict["fixed_params"] = [
        5 * len(SeqRepL_exr_list) / n_task,  # p
        min(100, T // 2),  # tau1
        min(30, T // 4),   # tau2
    ]
    
    check_params(input_dict)
    
    print(f"\nParameters:")
    print(f"  d={input_dict['d']}, m={input_dict['m']}, T={T}, n_task={n_task}")
    print(f"  n_sim={input_dict['n_sim']}")
    print(f"  adv_exr_task={input_dict['adv_exr_task']}")
    print(f"  fixed_params={input_dict['fixed_params']}")
    print()
    
    # Test BRESS
    print("-" * 40)
    print("Testing BRESS...")
    print("-" * 40)
    input_dict["name"] = "BRESS"
    input_dict["params_set"] = [None]
    
    try:
        result_online_pca = eval_multi(input_dict)
        print(f"  Final cumulative regret: {np.mean(result_online_pca['regrets'], axis=0)[-1]:.2f}")
        print("  BRESS test PASSED!")
        return result_online_pca, input_dict
    except Exception as e:
        print(f"  BRESS test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, input_dict


def run_comparison():
    """
    Run comparison between BRESS and SeqRepL.
    Uses moderate parameters for meaningful comparison.
    """
    print("=" * 60)
    print("Running COMPARISON TEST: BRESS vs SeqRepL")
    print("=" * 60)
    
    T = 500  # Timesteps per task
    n_task = 200  # Number of tasks
    
    input_dict = {
        "d": 8,
        "unit_ball_action": True,
        "T": T,
        "n_sim": 3,
        "rho": 0.5,
        "noise_std": 1,
        "seed": 42,
        "output": True,
        "params_set": [None],
        "m": 3,
        "n_task": n_task,
        "mode": MODE_ADVERSARY,
        "adv_exr_const": 0.1,
        "adv_exr_task": [0, 50, 100],
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
    
    SeqRepL_exr_list = get_SeqRepL_exr_list(n_task)
    input_dict["SeqRepL_exr_list"] = SeqRepL_exr_list
    input_dict["fixed_params"] = [
        5 * len(SeqRepL_exr_list) / n_task,
        min(200, T // 2),
        min(60, T // 4),
    ]
    
    check_params(input_dict)
    
    print(f"\nParameters:")
    print(f"  d={input_dict['d']}, m={input_dict['m']}, T={T}, n_task={n_task}")
    print()
    
    results = {}
    
    # Test SeqRepL
    print("-" * 40)
    print("Running SeqRepL...")
    print("-" * 40)
    input_dict["name"] = "SeqRepL"
    try:
        results["SeqRepL"] = eval_multi(input_dict)
        print(f"  Final cumulative regret: {np.mean(results['SeqRepL']['regrets'], axis=0)[-1]:.2f}")
    except Exception as e:
        print(f"  SeqRepL FAILED: {e}")
        results["SeqRepL"] = None
    
    # Test BRESS
    print("-" * 40)
    print("Running BRESS...")
    print("-" * 40)
    input_dict["name"] = "BRESS"
    try:
        results["BRESS"] = eval_multi(input_dict)
        print(f"  Final cumulative regret: {np.mean(results['BRESS']['regrets'], axis=0)[-1]:.2f}")
    except Exception as e:
        print(f"  BRESS FAILED: {e}")
        results["BRESS"] = None
    
    # Plot comparison if both succeeded
    if results["SeqRepL"] is not None and results["BRESS"] is not None:
        plot_comparison(results, n_task)
    
    return results


def plot_comparison(results, n_task):
    """Plot comparison between algorithms."""
    X = np.arange(1, n_task + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cumulative regret
    ax = axes[0, 0]
    for name, result in results.items():
        if result is not None:
            mean = np.mean(result["regrets"], axis=0)
            std = np.std(result["regrets"], axis=0)
            ax.plot(X, mean, label=name)
            ax.fill_between(X, mean - std, mean + std, alpha=0.3)
    ax.set_xlabel("# of tasks")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Cumulative Regret over Tasks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # B_hat error
    ax = axes[0, 1]
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
                               (mean + std)[valid_idx], alpha=0.3)
    ax.set_xlabel("# of tasks")
    ax.set_ylabel("B_hat Error")
    ax.set_title("Subspace Estimation Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Theta_hat error
    ax = axes[1, 0]
    for name, result in results.items():
        if result is not None and "theta_hat_err" in result:
            mean = np.mean(result["theta_hat_err"], axis=0)
            std = np.std(result["theta_hat_err"], axis=0)
            valid_idx = mean > 0
            if np.any(valid_idx):
                ax.plot(X[valid_idx], mean[valid_idx], label=name)
                ax.fill_between(X[valid_idx],
                               (mean - std)[valid_idx],
                               (mean + std)[valid_idx], alpha=0.3)
    ax.set_xlabel("# of tasks")
    ax.set_ylabel("Theta_hat Error")
    ax.set_title("Parameter Estimation Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Angle error
    ax = axes[1, 1]
    for name, result in results.items():
        if result is not None and "angle_err" in result:
            mean = np.mean(result["angle_err"], axis=0)
            std = np.std(result["angle_err"], axis=0)
            ax.plot(X, mean, label=name)
            ax.fill_between(X, mean - std, mean + std, alpha=0.3)
    ax.set_xlabel("# of tasks")
    ax.set_ylabel("1 - cos(angle)")
    ax.set_title("Angle Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/online_pca_comparison.png", dpi=150)
    print(f"\nPlot saved to figures/online_pca_comparison.png")
    plt.close()  # Close figure to free memory (non-interactive mode)


if __name__ == "__main__":
    import sys
    
    # Run small test first
    result, input_dict = run_test_small()
    
    if result is not None:
        print("\n" + "=" * 60)
        print("Small test PASSED! Running comparison test...")
        print("=" * 60 + "\n")
        
        # Only run comparison if small test passed
        if len(sys.argv) > 1 and sys.argv[1] == "--comparison":
            run_comparison()
        else:
            print("To run comparison test, use: python test_online_pca.py --comparison")

