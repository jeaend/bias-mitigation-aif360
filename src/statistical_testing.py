import pandas as pd
from scipy.stats import wilcoxon

def significance_strength_symbol(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''
    
def perform_wilcoxon(df): 
    # 1) Count dataset-attribute-method combinations, excluded baseline
    combos = []
    for (dataset, attr), group in df.groupby(['Dataset', 'Sensitive Attribute']):
        methods = [m for m in group['Mitigation'].unique() if m.lower() != 'baseline']
        for method in methods:
            pipeline = group[group['Mitigation'] == method]['Pipeline'].iloc[0]
            combos.append({'Dataset': dataset, 'Attribute': attr, 'Method': method, 'Pipeline': pipeline})
    combo_df = pd.DataFrame(combos)
    print(f"Number of (Dataset, Attribute, Method) combos: {len(combo_df)}")
    combo_df

    # 2) Perform Wilcoxon tests for each metric
    df['fold'] = df.groupby(['Dataset', 'Sensitive Attribute', 'Mitigation']).cumcount()
    METRICS = ['accuracy', 'f1_score', 'SPD', 'DI', 'EOD', 'AOD']

    results = []
    for (dataset, attr), sub in df.groupby(['Dataset', 'Sensitive Attribute']):
        for metric in METRICS:
            pivot = sub.pivot(index='fold', columns='Mitigation', values=metric)
            for method in pivot.columns:
                if method.lower() == 'baseline':
                    continue
                x = pivot['baseline'].dropna()
                y = pivot[method].dropna()
                if len(x) == len(y) and len(x) > 0:
                    W, p = wilcoxon(x, y, zero_method='wilcox')
                    pipeline = sub[sub['Mitigation'] == method]['Pipeline'].iloc[0]
                    results.append({
                        'Dataset': dataset,
                        'Attribute': attr,
                        'Pipeline': pipeline,
                        'Method': method,
                        'Metric': metric,
                        'W': W,
                        'p-value': p
                    })

    res_df = pd.DataFrame(results)
    res_df['p_adj'] = (res_df['p-value'] * len(METRICS)).clip(upper=1.0)
    res_df['p-value'] = res_df['p-value'].round(3)
    res_df['p_adj'] = res_df['p_adj'].round(3)
    res_df['significant'] = res_df['p_adj'] < 0.05
    res_df['significance'] = res_df['p_adj'].apply(significance_strength_symbol)

    print(f"Performed {len(res_df)} Wilcoxon tests (6 metrics × {len(combo_df)} combos = {len(res_df)}).")
    
    return res_df
