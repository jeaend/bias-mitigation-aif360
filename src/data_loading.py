import pandas as pd
from aif360.datasets import AdultDataset
from src.data_preprocessing import preprocessing_adult

def load_adult_sex():
    ad = AdultDataset(
        protected_attribute_names=['sex'],
        privileged_classes=[['Male']],
        categorical_features=[
            'workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'native-country'
        ],
        features_to_drop=[''],
        instance_weights_name='fnlwgt',
        custom_preprocessing=preprocessing_adult,
        na_values=[]
    )

    # 2) Build a single DataFrame (features + label) 
    df = pd.DataFrame(ad.features, columns=ad.feature_names)
    df['label'] = ad.labels.ravel()
    
    return df


def load_adult_race():
    ad = AdultDataset(
        protected_attribute_names=['race'],
        privileged_classes=[['White']],
        categorical_features=[
            'workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'sex',
            'native-country'
        ],
        features_to_drop=[''],
        instance_weights_name='fnlwgt',
        custom_preprocessing=preprocessing_adult,
        na_values=[]
    )

    # 2) Build a single DataFrame (features + label) 
    df = pd.DataFrame(ad.features, columns=ad.feature_names)
    df['label'] = ad.labels.ravel()
    
    return df