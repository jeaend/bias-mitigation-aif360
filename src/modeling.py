from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import reset_default_graph
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
from aif360.algorithms.postprocessing import EqOddsPostprocessing, RejectOptionClassification
from aif360.datasets import BinaryLabelDataset
import pandas as pd

def get_default_model_pipeline():
    return Pipeline([
        ('scaler', RobustScaler()),
        ('clf',    LogisticRegression(solver='liblinear'))
    ])

def train_and_predict(df, feature_cols, train_idx, test_idx, pipeline=None):
    if pipeline is None:
        pipeline = get_default_model_pipeline()   

    # split + extract 
    X_train = df.iloc[train_idx][feature_cols]
    y_train = df.iloc[train_idx]['label']
    X_test  = df.iloc[test_idx][feature_cols]
    y_test  = df.iloc[test_idx]['label']

    # fit & predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # return the test‐DataFrame for metrics
    test_df = df.iloc[test_idx]
    return test_df, y_test, y_pred

################ PREPROCESSING

def reweighing_train_and_predict(
    ds,
    df,
    train_idx,
    test_idx,
    protected,
    privileged_value,
    unprivileged_value,
    pipeline=None
):
    train_bld = ds.subset(train_idx)
    test_bld  = ds.subset(test_idx)

    # Fit & apply REWEIGHING on the training split only
    rw = Reweighing(
        unprivileged_groups=[{protected: unprivileged_value}],
        privileged_groups=[{protected:   privileged_value}]
    )
    rw.fit(train_bld)
    train_transf = rw.transform(train_bld)

    X_tr = train_transf.features
    y_tr = train_transf.labels.ravel()
    w_tr = train_transf.instance_weights.ravel()

    X_te = test_bld.features
    y_te = test_bld.labels.ravel()

    # Train with sample_weight
    ## Adult: w_tr = fnlwgt × reweigh_factor
    ## COMPAS: w_tr = 1 × reweigh_factor (1 is default when weight not explicily set)
    if pipeline is None:
        pipeline = get_default_model_pipeline()
    pipeline.fit(X_tr, y_tr, clf__sample_weight=w_tr)

    y_pred = pipeline.predict(X_te)
    test_df = df.iloc[test_idx]
    return test_df, y_te, y_pred

def disparate_impact_remover_train_and_predict(
    ds,
    df,
    train_idx,
    test_idx,
    protected,
    repair_level=1.0,
    pipeline=None
):
    train_bld = ds.subset(train_idx)
    test_bld  = ds.subset(test_idx)

    direr_train = DisparateImpactRemover(
        repair_level=repair_level,
        sensitive_attribute=protected
    )
    train_transf = direr_train.fit_transform(train_bld)

    direr_test = DisparateImpactRemover(
        repair_level=repair_level,
        sensitive_attribute=protected
    )
    test_transf = direr_test.fit_transform(test_bld)

    X_tr = train_transf.features
    y_tr = train_transf.labels.ravel()

    X_te = test_transf.features
    y_te = test_transf.labels.ravel()

    if pipeline is None:
        pipeline = get_default_model_pipeline()
    pipeline.fit(X_tr, y_tr)

    y_pred = pipeline.predict(X_te)
    test_df = df.iloc[test_idx]

    return test_df, y_te, y_pred

################ INPROCESSING

def adversial_debiasing_train_and_predict(
    df,
    train_idx,
    test_idx,
    protected,
    privileged_value,
    unprivileged_value,
    privileged_groups,
    unprivileged_groups,
    scope_name='adv',
    num_epochs=50,
    batch_size=128,
    adversary_loss_weight=0.1
):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    # 2) Identify feature columns (exclude label & protected)
    feature_cols = [c for c in df.columns if c not in ('label', protected)]

    # 3) Scale numeric features
    scaler = RobustScaler()
    train_vals = scaler.fit_transform(train_df[feature_cols])
    test_vals  = scaler.transform(test_df[feature_cols])

    # 4) Rebuild DataFrames for AIF360
    train_scaled = train_df[[protected, 'label']].copy().reset_index(drop=True)
    train_scaled[feature_cols] = train_vals
    test_scaled  = test_df[[protected, 'label']].copy().reset_index(drop=True)
    test_scaled[feature_cols]  = test_vals

    train_bld = BinaryLabelDataset(
        df=train_scaled,
        label_names=['label'],
        protected_attribute_names=[protected],
        favorable_label=1.0,
        unfavorable_label=0.0,
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )
    test_bld = BinaryLabelDataset(
        df=test_scaled,
        label_names=['label'],
        protected_attribute_names=[protected],
        favorable_label=1.0,
        unfavorable_label=0.0,
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )

    # 6) Train Adversarial Debiasing
    reset_default_graph()
    sess = tf.Session()
    adv = AdversarialDebiasing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        scope_name=scope_name,
        debias=True,
        sess=sess,
        num_epochs=num_epochs,
        batch_size=batch_size,
        adversary_loss_weight=adversary_loss_weight
    )
    adv.fit(train_bld)

    # 7) Predict & extract labels
    pred_bld = adv.predict(test_bld)
    y_test   = test_df['label'].values
    y_pred   = pred_bld.labels.ravel()

    sess.close()
    return test_df, y_test, y_pred

def prejudice_remover_train_and_predict(
    df,
    train_idx,
    test_idx,
    protected: str,
    privileged_value: float,
    unprivileged_value: float,
    eta: float = 25.0
):
    # 1) Split raw DataFrame
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    # 2) Identify feature columns (exclude label and protected attr)
    feature_cols = [c for c in df.columns if c not in ('label', protected)]

    # 3) Scale numeric features
    scaler = RobustScaler()
    train_scaled_vals = scaler.fit_transform(train_df[feature_cols])
    test_scaled_vals  = scaler.transform(test_df[feature_cols])

    # 4) Rebuild scaled DataFrames for AIF360
    train_scaled_df = pd.DataFrame(train_scaled_vals, columns=feature_cols)
    train_scaled_df['label'] = train_df['label'].values
    train_scaled_df[protected] = train_df[protected].values

    test_scaled_df = pd.DataFrame(test_scaled_vals, columns=feature_cols)
    test_scaled_df['label'] = test_df['label'].values
    test_scaled_df[protected] = test_df[protected].values

    train_bld = BinaryLabelDataset(
        df=train_scaled_df,
        label_names=['label'],
        protected_attribute_names=[protected],
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )
    test_bld = BinaryLabelDataset(
        df=test_scaled_df,
        label_names=['label'],
        protected_attribute_names=[protected],
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )

    # 6) Train PrejudiceRemover (in-processing)
    pr = PrejudiceRemover(eta=eta, sensitive_attr=protected)
    pr = pr.fit(train_bld)

    pred_bld = pr.predict(test_bld)

    y_test = test_bld.labels.ravel()
    y_pred = pred_bld.labels.ravel()

    return test_df, y_test, y_pred

################ POSTPROCESSING
def eq_odds_postprocessing_train_and_predict(
    df,
    train_idx,
    test_idx,
    protected: str,
    privileged_value: float,
    unprivileged_value: float,
    seed: int = 42
):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df.drop(columns=['label', protected])
    y_train = train_df['label'].values
    X_test  = test_df.drop(columns=['label', protected])
    y_test  = test_df['label'].values

    pipeline = get_default_model_pipeline()
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred  = pipeline.predict(X_test)

    train_bld = BinaryLabelDataset(
        df=train_df,
        label_names=['label'],
        protected_attribute_names=[protected],
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )
    test_bld  = BinaryLabelDataset(
        df=test_df,
        label_names=['label'],
        protected_attribute_names=[protected],
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )

    train_pred = train_bld.copy(deepcopy=True)
    train_pred.labels = y_train_pred.reshape(-1, 1)
    test_pred  = test_bld.copy(deepcopy=True)
    test_pred.labels  = y_test_pred.reshape(-1, 1)

    # e) Fit Equalized Odds post‐processor
    eq = EqOddsPostprocessing(
        privileged_groups=[{protected: privileged_value}],
        unprivileged_groups=[{protected: unprivileged_value}],
        seed=seed
    )
    eq = eq.fit(train_bld, train_pred)

    post_bld = eq.predict(test_pred)
    y_pred_post = post_bld.labels.ravel()

    return test_df, y_test, y_pred_post

def reject_option_classification_train_and_predict(
    df,
    train_idx,
    test_idx,
    feature_cols,
    protected,
    privileged_value,
    unprivileged_value,
    pipeline=None
):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df[feature_cols]
    y_train = train_df['label'].values
    X_test  = test_df[feature_cols]
    y_test  = test_df['label'].values

    pipeline = get_default_model_pipeline()
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred  = pipeline.predict(X_test)

    train_probs = pipeline.predict_proba(X_train)[:, 1].reshape(-1, 1)
    test_probs  = pipeline.predict_proba(X_test)[:, 1].reshape(-1, 1)

    train_bld = BinaryLabelDataset(
        df=train_df,
        label_names=['label'],
        protected_attribute_names=[protected],
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )
    test_bld = BinaryLabelDataset(
        df=test_df,
        label_names=['label'],
        protected_attribute_names=[protected],
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )

    train_pred = train_bld.copy(deepcopy=True)
    train_pred.labels = y_train_pred.reshape(-1, 1)
    train_pred.scores = train_probs

    test_pred = test_bld.copy(deepcopy=True)
    test_pred.labels = y_test_pred.reshape(-1, 1)
    test_pred.scores = test_probs

    # Apply RejectOptionClassification with defaults
    roc = RejectOptionClassification(
        unprivileged_groups=[{protected: unprivileged_value}],
        privileged_groups=[{protected: privileged_value}]
        # all other parameters are left at their defaults
    )
    roc.fit(train_bld, train_pred)
    post_bld = roc.predict(test_pred)
    y_pred = post_bld.labels.ravel()

    return test_df, y_test, y_pred
