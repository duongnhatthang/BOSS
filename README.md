# Representation transfer for sequential multitask linear bandits

This repository is the official implementation of Competitive Ratio and its application in Sequential Representation Learning for Multi-Task Linear Bandit


## Requirements
Python 3.10

```
git clone https://github.com/duongnhatthang/BOSS.git
cd BOSS
pip install -r requirements.txt
```

## Evaluation
Interactive [Notebook](https://github.com/duongnhatthang/BOSS/blob/main/exp.ipynb)

## Results

Our algorithm achieves the following performance on a synthetic dataset:


**Regret over tasks**             |  **Estimation error of B** | **Estimation error of $\theta_n$**
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/duongnhatthang/Serena/blob/main/figures/new_reg.png)  |  ![](https://github.com/duongnhatthang/Serena/blob/main/figures/new_B.png) |  ![](https://github.com/duongnhatthang/Serena/blob/main/figures/new_theta_smooth.png)

**Estimation error of the angle between $(\theta_n, \hat{\theta}_n)$**             |  **Regret over T, average across all tasks**
:-------------------------:|:-------------------------:
![](https://github.com/duongnhatthang/Serena/blob/main/figures/new_angle_smooth.png)  |  ![](https://github.com/duongnhatthang/Serena/blob/main/figures/new_reg_T.png)


## [Competitive Ratio and its application in Sequential Representation Learning for Multi-Task Linear Bandit]

Placeholder


## Contributing

[Apache 2.0](https://github.com/duongnhatthang/meta-bandit/blob/main/LICENSE)

## Note

The code is referencing this [repo](https://github.com/oh-lab/LinearBandit/tree/main).
