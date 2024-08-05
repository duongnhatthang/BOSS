# Beyond task diversity: provable representation transfer for sequential multitask linear bandits

## Abstract:
- We study lifelong learning in linear bandits, where a learner interacts with a sequence of linear bandit tasks whose parameters lie in an $m$-dimensional subspace of $\mathbb{R}^d$, thereby sharing a low-rank representation. Current literature typically assumes that the tasks are *diverse*; that is, their parameters uniformly span the $m$-dimensional subspace. This assumption allows the low-rank representation to be learned before all tasks are revealed, which can be unrealistic in real-world applications. In this work, we consider a setting extending beyond the task diversity assumption. We present an algorithm that can efficiently learn and transfer low-rank representations. When facing $N$ tasks each played over $\tau$ rounds, our algorithm achieves a regret guarantee of $\tilde{O}\left (Nm \sqrt{\tau} + N^{\frac{2}{3}} \tau^{\frac{2}{3}} d m^{\frac13} \right)$ under mild assumptions. This improves upon the baseline $\tilde{O} \left (Nd \sqrt{\tau}\right)$ that does not leverage the low-rank structure. We demonstrate empirically on synthetic data that our algorithm outperforms baseline algorithms that rely on the task diversity assumption.

## Experiment results:

**Regret over tasks**             |  **Estimation error of B** | **Estimation error of $\theta_n$**
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/duongnhatthang/Serena/blob/main/figures/new_reg.png)  |  ![](https://github.com/duongnhatthang/Serena/blob/main/figures/new_B.png) |  ![](https://github.com/duongnhatthang/Serena/blob/main/figures/new_theta_smooth.png)

**Estimation error of the angle between $(\theta_n, \hat{\theta}_n)$**             |  **Regret over T, average across all tasks**
:-------------------------:|:-------------------------:
![](https://github.com/duongnhatthang/Serena/blob/main/figures/new_angle_smooth.png)  |  ![](https://github.com/duongnhatthang/Serena/blob/main/figures/new_reg_T.png)


## Installation 
 -  Python 3.10

    ```
    git clone https://github.com/duongnhatthang/Serena.git
    cd Serena
    pip install -r requirements.txt
    ```

## Evaluation 
 -  Interactive [Notebook](https://github.com/duongnhatthang/Serena/blob/main/exp.ipynb)

## License: [Apache 2.0](https://github.com/duongnhatthang/meta-bandit/blob/main/LICENSE)

## Note

The code is built using this [repo](https://github.com/oh-lab/LinearBandit/tree/main).
