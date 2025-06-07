from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import MetaFairClassifier, PrejudiceRemover
from aif360.algorithms.postprocessing import EqOddsPostprocessing, RejectOptionClassification
from aif360.datasets import BinaryLabelDataset
import pandas as pd
import numpy as np

def get_default_model_pipeline():
    return Pipeline([
        ('scaler', RobustScaler()),
        ('clf',    LogisticRegression(solver='liblinear'))
    ])

def train_and_predict(df, feature_cols, train_idx, test_idx):
    # PREPARE DATA
    X_train = df.iloc[train_idx][feature_cols]
    y_train = df.iloc[train_idx]['label']
    X_test  = df.iloc[test_idx][feature_cols]
    y_test  = df.iloc[test_idx]['label']

    # TRAIN MODEL
    pipeline = get_default_model_pipeline()   
    pipeline.fit(X_train, y_train)

    # GET PREDICTIONS
    y_pred = pipeline.predict(X_test)
    test_df = df.iloc[test_idx]

    return test_df, y_test, y_pred

################ PREPROCESSING

def reweighing_train_and_predict(ds,df,train_idx,test_idx,protected,privileged_value,unprivileged_value):
    # PREPARE DATA
    train_bld = ds.subset(train_idx)
    test_bld  = ds.subset(test_idx)

    ## Fit & apply reweighing on the training split only
    rw = Reweighing(
        unprivileged_groups=[{protected: unprivileged_value}],
        privileged_groups=[{protected: privileged_value}]
    )
    rw.fit(train_bld)
    train_transf = rw.transform(train_bld)

    X_tr = train_transf.features
    y_tr = train_transf.labels.ravel()
    w_tr = train_transf.instance_weights.ravel()

    X_te = test_bld.features
    y_te = test_bld.labels.ravel()

    # TRAIN MODEL with sample_weight
    ## Adult: w_tr = fnlwgt × reweigh_factor
    ## COMPAS: w_tr = 1 × reweigh_factor (1 is default when weight not explicily set)
    pipeline = get_default_model_pipeline()
    pipeline.fit(X_tr, y_tr, clf__sample_weight=w_tr)

    # GET PREDICTIONS
    y_pred = pipeline.predict(X_te)
    test_df = df.iloc[test_idx]

    return test_df, y_te, y_pred

def disparate_impact_remover_train_and_predict(ds,df,train_idx,test_idx,protected):
    # PREPARE DATA
    train_bld = ds.subset(train_idx)
    test_bld  = ds.subset(test_idx)

    ## implementation lacks a separate transform() call 
    ## calling fit_transform() twice (once on train, once on test) 
    ## ensures that each splits features are repaired independently, no data leakage
    direr_train = DisparateImpactRemover(
        repair_level=1.0,
        sensitive_attribute=protected
    )
    train_transf = direr_train.fit_transform(train_bld)

    direr_test = DisparateImpactRemover(
        repair_level=1.0,
        sensitive_attribute=protected
    )
    test_transf = direr_test.fit_transform(test_bld)

    X_tr = train_transf.features
    y_tr = train_transf.labels.ravel()

    X_te = test_transf.features
    y_te = test_transf.labels.ravel()

    # TRAIN MODEL 
    pipeline = get_default_model_pipeline()
    pipeline.fit(X_tr, y_tr)

    # GET PREDICTIONS
    y_pred = pipeline.predict(X_te)
    test_df = df.iloc[test_idx]

    return test_df, y_te, y_pred

################ INPROCESSING
def meta_fair_classifier_train_and_predict(df: pd.DataFrame,train_idx: np.ndarray,test_idx: np.ndarray,protected: str,privileged_value: float,unprivileged_value: float, favorable_label: float,unfavorable_label: float):
    # PREPARE DATA
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ('label', protected)]

    ## Scaling before usign the aif360 model
    scaler     = RobustScaler()
    train_vals = scaler.fit_transform(train_df[feature_cols])
    test_vals  = scaler.transform(test_df[feature_cols])

    ## Rebuild DataFrames for AIF360 (only protected, label, and scaled features)
    train_scaled = train_df[[protected, 'label']].copy().reset_index(drop=True)
    train_scaled[feature_cols] = train_vals
    test_scaled = test_df[[protected, 'label']].copy().reset_index(drop=True)
    test_scaled[feature_cols] = test_vals

    ## Wrap as AIF360 BinaryLabelDataset
    train_bld = BinaryLabelDataset(
        df=train_scaled,
        label_names=['label'],
        protected_attribute_names=[protected],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )
    test_bld = BinaryLabelDataset(
        df=test_scaled,
        label_names=['label'],
        protected_attribute_names=[protected],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )

    # TRAIN MODEL 
    mfc = MetaFairClassifier(
        tau=0.5, 
        sensitive_attr=protected,
        type='sr',   # 'sr' = statistical‐rate (demographic parity)
        seed=42
    )
    mfc.fit(train_bld)

    # GET PREDICTIONS
    pred_bld = mfc.predict(test_bld)
    y_test = test_df['label'].values
    y_pred = pred_bld.labels.ravel()

    return test_df, y_test, y_pred

def prejudice_remover_train_and_predict(df,train_idx,test_idx,protected: str,privileged_value: float,unprivileged_value: float,favorable_label,unfavorable_label):
    # PREPARE DATA
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ('label', protected)]

    ## Scaling before usign the aif360 model
    scaler = RobustScaler()
    train_scaled_vals = scaler.fit_transform(train_df[feature_cols])
    test_scaled_vals  = scaler.transform(test_df[feature_cols])

    ## Rebuild scaled DataFrames for AIF360
    train_scaled_df = pd.DataFrame(train_scaled_vals, columns=feature_cols)
    train_scaled_df['label'] = train_df['label'].values
    train_scaled_df[protected] = train_df[protected].values

    test_scaled_df = pd.DataFrame(test_scaled_vals, columns=feature_cols)
    test_scaled_df['label'] = test_df['label'].values
    test_scaled_df[protected] = test_df[protected].values

    ## Wrap as AIF360 BinaryLabelDataset
    train_bld = BinaryLabelDataset(
        df=train_scaled_df,
        label_names=['label'],
        protected_attribute_names=[protected],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )
    test_bld = BinaryLabelDataset(
        df=test_scaled_df,
        label_names=['label'],
        protected_attribute_names=[protected],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )

    # TRAIN MODEL 
    pr = PrejudiceRemover(
        eta=25.0, 
        sensitive_attr=protected
    )
    pr = pr.fit(train_bld)

    # GET PREDICTIONS
    pred_bld = pr.predict(test_bld)
    y_test = test_bld.labels.ravel()
    y_pred = pred_bld.labels.ravel()

    return test_df, y_test, y_pred

################ POSTPROCESSING
def eq_odds_postprocessing_train_and_predict(df,train_idx,test_idx,protected: str,privileged_value: float,unprivileged_value: float):
    # PREPARE DATA
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df.drop(columns=['label', protected])
    y_train = train_df['label'].values
    X_test  = test_df.drop(columns=['label', protected])
    y_test  = test_df['label'].values

    # TRAIN MODEL 
    pipeline = get_default_model_pipeline()
    pipeline.fit(X_train, y_train)

    # GET PREDICTIONS
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

    eq = EqOddsPostprocessing(
        privileged_groups=[{protected: privileged_value}],
        unprivileged_groups=[{protected: unprivileged_value}],
        seed=42
    )
    eq = eq.fit(train_bld, train_pred)

    post_bld = eq.predict(test_pred)
    y_pred_post = post_bld.labels.ravel()

    return test_df, y_test, y_pred_post

def reject_option_classification_train_and_predict(df,train_idx,test_idx,feature_cols,protected,privileged_value,unprivileged_value):
    # PREPARE DATA
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df[feature_cols]
    y_train = train_df['label'].values
    X_test  = test_df[feature_cols]
    y_test  = test_df['label'].values

    # TRAIN MODEL 
    pipeline = get_default_model_pipeline()
    pipeline.fit(X_train, y_train)

    # GET PREDICTIONS
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

    roc = RejectOptionClassification(
        unprivileged_groups=[{protected: unprivileged_value}],
        privileged_groups=[{protected: privileged_value}]
    )
    roc.fit(train_bld, train_pred)
    post_bld = roc.predict(test_pred)
    y_pred = post_bld.labels.ravel()

    return test_df, y_test, y_pred
