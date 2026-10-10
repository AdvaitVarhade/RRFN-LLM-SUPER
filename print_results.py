import json

with open('robust_super/results/multiseed_benchmark.json') as f:
    d = json.load(f)

for dom in ['movielens', 'amazon', 'yelp']:
    print('========================================')
    print('DOMAIN:', dom.upper())
    roc = d[dom]['mean_roc_auc']
    ap = d[dom]['mean_avg_precision']
    print(f'ROC-AUC: {roc:.4f}, AP: {ap:.4f}')
    pvals = d[dom]['p_values']
    for model in ['robust', 'vanilla_attacked', 'vanilla_clean', 'uncalibrated']:
        print(f'  Model: {model}')
        stats = d[dom]['summary_stats'][model]
        for m in ['Recall@10', 'nDCG@10', 'RMSE-PC', 'MRMC', 'LTC@10', 'g-LTC@10', 'STER@10', 'Novelty', 'GKPI']:
            if m in stats:
                pval = pvals.get(m, 1.0)
                mean = stats[m]['mean']
                std = stats[m]['std']
                p_str = f'(p={pval:.4f})' if model == 'robust' else ''
                print(f'    {m:<12}: {mean:.4f} +- {std:.4f} {p_str}')
