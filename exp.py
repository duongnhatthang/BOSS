from evaluation import eval_UCB, eval_TS, eval_PHE
for d in [5, 10, 20]:
    for n_gen_context in [10, 20]:
        eval_UCB(n_gen_context=n_gen_context, d=d)
        eval_TS(n_gen_context=n_gen_context, d=d)
        eval_PHE(n_gen_context=n_gen_context, d=d)
