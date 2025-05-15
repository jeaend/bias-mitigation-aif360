from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

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