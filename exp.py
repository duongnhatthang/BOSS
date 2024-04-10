from evaluation import eval_UCB, eval_TS, eval_PHE
for d in [5, 10, 20]:
    for N in [10, 20]:
        eval_UCB(N=N, d=d)
        eval_TS(N=N, d=d)
        eval_PHE(N=N, d=d)
