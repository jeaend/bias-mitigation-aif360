from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import reset_default_graph
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
from aif360.datasets import BinaryLabelDataset

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
    # Reset TF graph - start new session (to avoid "Variable … already exists")
    reset_default_graph()
    sess = tf.Session()

    # 2) Split DF
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    # 3) Wrap into AIF360 datasets
    train_bld = BinaryLabelDataset(
        df=train_df,
        label_names=['label'],
        protected_attribute_names=[protected],
        favorable_label=1.0,
        unfavorable_label=0.0,
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )
    test_bld = BinaryLabelDataset(
        df=test_df,
        label_names=['label'],
        protected_attribute_names=[protected],
        favorable_label=1.0,
        unfavorable_label=0.0,
        privileged_protected_attributes=[[privileged_value]],
        unprivileged_protected_attributes=[[unprivileged_value]]
    )

    # 4) Instantiate & train the adversarial debiaser
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

    # 5) Predict & extract labels
    pred_bld = adv.predict(test_bld)
    y_test   = test_df['label'].values
    y_pred   = pred_bld.labels.ravel()

    # 6) Clean up sess
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
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

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

    # Train PrejudiceRemover (η = 25.0 default) 
    pr = PrejudiceRemover(eta=eta, sensitive_attr=protected)
    pr = pr.fit(train_bld)

    pred_bld = pr.predict(test_bld)

    y_test = test_bld.labels.ravel()
    y_pred = pred_bld.labels.ravel()

    return test_df, y_test, y_pred
