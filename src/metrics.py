import os
import pandas as pd
import matplotlib.pyplot as plt
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.metrics import accuracy_score, f1_score

COL_BAR    = "#C9C6C6"    
BAND_FAIR  = "#b4dcb4"    
COL_MIT    = "#afadad"    
BAND_BIAS  = "#d89494"    
ALPHA_BAND = 0.4

METRICS = ['accuracy', 'f1_score', 'SPD', 'DI', 'EOD', 'AOD']
AGG_OUT_PATH = '../../reports/agg_metrics.csv'
RAW_OUT_PATH = '../../reports/compas_raw_metrics.csv'
KEYS = ['Dataset', 'Sensitive Attribute', 'Mitigation']

DI_LO, DI_HI = 0.8, 1.25
SPD_MAX = 0.1
EOD_MAX = 0.1
AOD_MAX = 0.1

def compute_metrics(
    test_df,
    y_true,
    y_pred,
    protected: str,
    privileged_value: float = 1.0,
    unprivileged_value: float = 0.0
) -> dict:
    """
    Wraps dataframe + predictions into AIF360 datasets structure and returns a dict of:
      - accuracy, f1_score,
      - statistical_parity_difference, disparate_impact,
      - equal_opportunity_difference, average_odds_difference
    """
    # 1) Original & predicted BinaryLabelDataset
    orig = BinaryLabelDataset(
        favorable_label=1.0, unfavorable_label=0.0,
        df=test_df, label_names=['label'],
        protected_attribute_names=[protected],
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )
    pred = orig.copy(deepcopy=True)
    pred.labels = y_pred.reshape(-1, 1)

    # 2) Outcome‐based metrics
    bldm = BinaryLabelDatasetMetric(
        pred,
        privileged_groups=[{protected: privileged_value}],
        unprivileged_groups=[{protected: unprivileged_value}]
    )
    spd = bldm.statistical_parity_difference()
    di  = bldm.disparate_impact()

    # 3) Error‐based metrics
    cls = ClassificationMetric(
        orig, pred,
        privileged_groups=[{protected: privileged_value}],
        unprivileged_groups=[{protected: unprivileged_value}]
    )
    eod = cls.equal_opportunity_difference()
    aod = cls.average_odds_difference()

    # 4) Performance metrics
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)

    return {
        'accuracy': acc,
        'f1_score': f1,
        'SPD':      spd,
        'DI':       di,
        'EOD':      eod,
        'AOD':      aod
    }


def viz_metrics_2x3(metrics_dataframe_agg, label='Baseline', title=None):
    titles  = [
        'Accuracy',
        'F1 Score',
        'Statistical Parity\nDifference',
        'Disparate\nImpact',
        'Equal Opportunity\nDifference',
        'Average Odds\nDifference'
    ]

    mean_vals = metrics_dataframe_agg.loc['mean', METRICS]
    std_vals  = metrics_dataframe_agg.loc['std',  METRICS]

    fair_bands = {'SPD':(-0.1,0.1), 'DI':(0.8,1.25), 'EOD':(-0.1,0.1), 'AOD':(-0.1,0.1)}
    ylims = {
        'accuracy': (0.0,1.0),
        'f1_score': (0.0,1.0),
        'SPD':      (-1.0,1.0),
        'DI':       (0.0,1.5),
        'EOD':      (-1.0,1.0),
        'AOD':      (-1.0,1.0)
    }
    yticks = {
        'accuracy': [0,0.5,1.0],
        'f1_score': [0,0.5,1.0],
        'SPD':      [-1,-0.5,0,0.5,1],
        'DI':       [0,0.5,1,1.25,1.5],
        'EOD':      [-1,-0.5,0,0.5,1],
        'AOD':      [-1,-0.5,0,0.5,1]
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=False)
    if title:
        fig.suptitle(title, fontsize=14)
    axes = axes.flatten()
    bar_width = 0.3

    for ax, metric, title in zip(axes, METRICS, titles):
        m = mean_vals[metric]
        s = std_vals[metric]
        lo_y, hi_y = ylims[metric]
        y_off = 0.05 * (hi_y - lo_y)

        if metric in ('accuracy','f1_score'):
            ax.axhline(1.0, color='black', linewidth=1)
            ax.bar([''], [m], yerr=[s], capsize=4, color=COL_BAR, width=bar_width)
            ax.text(0, m+y_off, f"{m:.2f}\n±{s:.2f}", ha='center', va='bottom')
        else:
            lo_f, hi_f = fair_bands[metric]
            ax.axhspan(lo_f, hi_f, color=BAND_FAIR, alpha=ALPHA_BAND)
            ax.axhspan(lo_y, lo_f, color=BAND_BIAS, alpha=ALPHA_BAND)
            ax.axhspan(hi_f, hi_y, color=BAND_BIAS, alpha=ALPHA_BAND)

            if metric == 'DI':
                ax.axhline(1, color='black')
            else:
                ax.axhline(0, color='black')

            ax.bar([''], [m], yerr=[s], capsize=4, color=COL_BAR, width=bar_width)
            va = 'bottom' if m >= 0 else 'top'
            y_text = m + y_off if m >= 0 else m - y_off
            ax.text(0, y_text, f"{m:.2f}\n±{s:.2f}", ha='center', va=va)
            ax.text(1.02,(lo_y+lo_f)/2,'Bias\n(priv)', transform=ax.get_yaxis_transform(),
                    ha='left', va='center', color=BAND_BIAS, fontsize=8)
            ax.text(1.02,(lo_f+hi_f)/2,'Fair', transform=ax.get_yaxis_transform(),
                    ha='left', va='center', color=BAND_FAIR, fontsize=10)
            ax.text(1.02,(hi_f+hi_y)/2,'Bias\n(unpriv)', transform=ax.get_yaxis_transform(),
                    ha='left', va='center', color=BAND_BIAS, fontsize=8)

        ax.set_title(title, fontsize=12)
        ax.set_ylim(lo_y, hi_y)
        ax.set_yticks(yticks[metric])
        ax.set_xticks([])
        ax.set_xlabel(label, fontsize=10)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.1, hspace=0.4, wspace=0.3)
    plt.show()

def compare_viz_metrics_2x3(df_base, df_mit, label1='Baseline', label2='Mitigation', title=None):
    titles  = [
        'Accuracy', 'F1 Score', 'Statistical Parity\nDifference',
        'Disparate\nImpact', 'Equal Opportunity\nDifference', 'Average Odds\nDifference'
    ]

    fair_bands = {'SPD':(-0.1,0.1), 'DI':(0.8,1.25), 'EOD':(-0.1,0.1), 'AOD':(-0.1,0.1)}
    ylims = {
        'accuracy': (0.0,1.0), 'f1_score': (0.0,1.0),
        'SPD':(-1.0,1.0), 'DI':(0.0,1.5), 'EOD':(-1.0,1.0), 'AOD':(-1.0,1.0)
    }
    yticks = {
        'accuracy': [0,0.5,1.0], 'f1_score': [0,0.5,1.0],
        'SPD':[-1,-0.5,0,0.5,1], 'DI':[0,0.5,1,1.25,1.5],
        'EOD':[-1,-0.5,0,0.5,1], 'AOD':[-1,-0.5,0,0.5,1]
    }

    fig, axes = plt.subplots(2,3,figsize=(12,8))
    if title:
        fig.suptitle(title, fontsize=14)
    axes = axes.flatten()
    bar_w = 0.4
    gap = 0.05
    x_pos = [-(bar_w/2 + gap/2), (bar_w/2 + gap/2)]

    for ax, metric, ttl in zip(axes, METRICS, titles):
        lo, hi = ylims[metric]
        ax.set_title(ttl)
        ax.set_ylim(lo, hi)
        ax.set_yticks(yticks[metric])

        # fairness bands
        if metric not in ('accuracy','f1_score'):
            lf, hf = fair_bands[metric]
            ax.axhspan(lf, hf, color=BAND_FAIR, alpha=ALPHA_BAND)
            ax.axhspan(lo, lf, color=BAND_BIAS, alpha=ALPHA_BAND)
            ax.axhspan(hf, hi, color=BAND_BIAS, alpha=ALPHA_BAND)
            base_line = 1 if metric=='DI' else 0
            ax.axhline(base_line, color='black')

        # bars
        mb, sb = df_base.loc['mean', metric], df_base.loc['std', metric]
        mm, sm = df_mit.loc['mean', metric], df_mit.loc['std', metric]
        ax.bar(x_pos, [mb, mm], bar_w, yerr=[sb, sm], capsize=4,
               color=[COL_BAR, COL_MIT])

        # annotate each bar
        pad = 0.02*(hi - lo)
        for x, val, err in zip(x_pos, [mb, mm], [sb, sm]):
            y_above = val + err + pad
            if y_above > hi:
                y = val - err - pad
                va = 'top'
            else:
                y = y_above
                va = 'bottom'

            ax.text(
                x, y,
                f"{val:.2f}\n±{err:.2f}",
                ha='center', va=va, fontsize=10,
                clip_on=False
            )

        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    plt.subplots_adjust(wspace=0.4, hspace=0.4, bottom=0.1)
    plt.show()


def save_agg_metrics(dataset_name,mitigation_name, race_agg_df, sex_agg_df):
    agg_dfs = {'race': race_agg_df, 'sex': sex_agg_df}
    rows = []
    for attr, df in agg_dfs.items():
        mean = df.loc['mean', METRICS]
        std  = df.loc['std',  METRICS]
        row = {
            'Dataset': dataset_name,
            'Sensitive Attribute': attr,
            'Mitigation': mitigation_name
        }
        for m in METRICS:
            row[m]     = mean[m]
            row[f'{m}_std'] = std[m]
        rows.append(row)
    agg_df = pd.DataFrame(rows)

    # logic to append if Mitigation + Sensitive Attribute doesnt exist, otherwise update metrics
    if os.path.exists(AGG_OUT_PATH):
        existing = pd.read_csv(AGG_OUT_PATH)
        mask = existing.set_index(KEYS).index.isin(agg_df.set_index(KEYS).index)
        existing = existing[~mask]
        final = pd.concat([existing, agg_df], ignore_index=True)
    else:
        final = agg_df

    final.to_csv(AGG_OUT_PATH, index=False)

def save_raw_metrics(dataset_name, mitigation_name, race_agg_df, sex_agg_df):
    raw_dfs = {
        'race': race_agg_df,
        'sex':  sex_agg_df
    }

    raw_list = []
    for attr, df in raw_dfs.items():
        tmp = df.reset_index(drop=True).copy()
        tmp['Dataset'] = dataset_name
        tmp['Sensitive Attribute'] = attr
        tmp['Mitigation'] = mitigation_name
        raw_list.append(tmp)

    raw_df = pd.concat(raw_list, ignore_index=True)
    front = KEYS
    raw_df = raw_df[front + [c for c in raw_df.columns if c not in front]]

    # if the file exists, drop any old rows with the same (Mitigation, Sensitive Attribute)
    if os.path.exists(RAW_OUT_PATH):
        existing = pd.read_csv(RAW_OUT_PATH)
        mask = existing.set_index(KEYS).index.isin(raw_df.set_index(KEYS).index)
        existing = existing[~mask]
        final = pd.concat([existing, raw_df], ignore_index=True)
    else:
        final = raw_df

    final.to_csv(RAW_OUT_PATH, index=False)

def best_hyperparameter_advdeb(results_df: pd.DataFrame) -> pd.Series:
    """
    From a DataFrame of hyperparameter results (with columns including
    ['acc_mean','SPD_mean','DI_mean', ...]),
    computes a simple fairness score:
        fairness_score = |SPD_mean| + |DI_mean - 1|
    and then selects the row with:
      1) highest acc_mean 
      2) lowest fairness_score
    """
    df = results_df.copy()
    # 1) Compute fairness score
    df['fairness_score'] = df['SPD_mean'].abs() + (df['DI_mean'] - 1.0).abs()

    # 2) Sort by fairness_score ascending, then acc_mean descending
    best = df.sort_values(
        by=['fairness_score', 'acc_mean'],
        ascending=[True, False]
    ).iloc[0]

    return best