from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from aif360.algorithms.preprocessing import Reweighing

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
