from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.metrics import accuracy_score, f1_score

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