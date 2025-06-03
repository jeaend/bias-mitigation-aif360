import pandas as pd
from aif360.datasets import AdultDataset, CompasDataset
from src.data_preprocessing import preprocessing_adult, preprocessing_compas

def load_adult_sex(custom_preprocessing=preprocessing_adult):
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
        custom_preprocessing=custom_preprocessing,
        na_values=[]
    )

    df = pd.DataFrame(ad.features, columns=ad.feature_names)
    df['label'] = ad.labels.ravel()
    
    df.drop(    
        ['workclass=Other', 'education=College', 'marital-status=Widowed', 'occupation=Unknown', 'relationship=Husband', 'race=Non-White', "native-country=Other"],
        axis=1,
        inplace=True
    )

    return ad, df


def load_adult_race(custom_preprocessing=preprocessing_adult):
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
        custom_preprocessing=custom_preprocessing,
        na_values=[]
    )

    df = pd.DataFrame(ad.features, columns=ad.feature_names)
    df['label'] = ad.labels.ravel()
     
    df.drop(    
        ['workclass=Other', 'education=College', 'marital-status=Widowed', 'occupation=Unknown', 'relationship=Husband', 'sex=Female', "native-country=Other"],
        axis=1,
        inplace=True
    )

    return ad, df

def load_compas_sex(custom_preprocessing=preprocessing_compas):
    ds = CompasDataset(
        protected_attribute_names=['sex'],
        privileged_classes=[['Female']],
        features_to_drop=['age_cat'],
        categorical_features=[
            'race', 'c_charge_degree', 'c_charge_desc'
        ],
        custom_preprocessing=custom_preprocessing
    )
    df = pd.DataFrame(ds.features, columns=ds.feature_names)
    df['label'] = ds.labels.ravel()
    df['race']  = ds.protected_attributes[:, 0]

    df.drop(    
        ['c_charge_desc=Other', 'c_charge_degree=F', 'race=Other'],
        axis=1,
        inplace=True
    )

    return ds, df

def load_compas_race(custom_preprocessing=preprocessing_compas):
    ds = CompasDataset(
        protected_attribute_names=['race'],
        privileged_classes=[['Caucasian']],
        features_to_drop=['age_cat'],
        categorical_features=[
            'sex', 'c_charge_degree', 'c_charge_desc'
        ],
        custom_preprocessing=custom_preprocessing
    )
    df = pd.DataFrame(ds.features, columns=ds.feature_names)
    df['label'] = ds.labels.ravel()
    df['race']  = ds.protected_attributes[:, 0]

    df.drop(    
        ['c_charge_desc=Other', 'sex=Female', 'c_charge_degree=F'],
        axis=1,
        inplace=True
    )

    return ds, df
