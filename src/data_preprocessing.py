def preprocessing_adult(df):
    """
    1) Replace missing markers ('?') with 'Unknown'
    2) Bin education levels into four categories
    3) Bin native-country into Top 5 + 'Other'
    4) Simplify workclass into 4 groups
    5) Binarize race into 'White' vs. 'Non-White'
    """
    # 1) Fill missing
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].replace('?', 'Unknown')

    # 2) Education grouping
    def edu_group(x):
        if x in {
            'Preschool','1st-4th','5th-6th','7th-8th',
            '9th','10th','11th','12th'
        }:
            return '<HS'
        if x == 'HS-grad':
            return 'HS-grad'
        if x in {'Some-college','Assoc-acdm','Assoc-voc'}:
            return 'College'
        return 'Post-grad'
    df['education'] = df['education'].apply(edu_group)

    # 3) Native-country binning
    major_countries = {
        'United-States', 'Mexico', 'Philippines',
        'Canada', 'Germany'
    }
    df['native-country'] = df['native-country'].apply(
        lambda x: x if x in major_countries else 'Other'
    )

    # 4) Workclass simplification
    gov = {'Federal-gov','State-gov','Local-gov'}
    self_emp = {'Self-emp-inc','Self-emp-not-inc'}
    def workclass_group(x):
        if x in gov:
            return 'Government'
        if x in self_emp:
            return 'Self-Employed'
        if x == 'Private':
            return 'Private'
        return 'Other'  # covers Without-pay, Never-worked, Unknown
    df['workclass'] = df['workclass'].apply(workclass_group)

    # 5) Race binarization
    df['race'] = df['race'].apply(lambda x: 'White' if x == 'White' else 'Non-White')

    return df


def preprocessing_compas(df):
    violent = {'assault','battery','murder','manslaughter'}
    property = {'theft','burglary','robbery','arson','trespass'}
    drug = {'possession','traff','deliver','cocaine', 'heroin','marijuana','meth','opioid'}
    alcohol_dui = {'dui','dwi','alcohol','intoxicated'}
    weapons = {'weapon','firearm','gun','deadly'}
    
    def charge_group(x):
        if not isinstance(x, str):
            return 'Other'
        txt = x.lower()
        if any(k in txt for k in violent):
            return 'Violent'
        if any(k in txt for k in property):
            return 'Property'
        if any(k in txt for k in drug):
            return 'Drug'
        if any(k in txt for k in alcohol_dui):
            return 'Alcohol_dui'
        if any(k in txt for k in weapons):
            return 'Weapons'
        return 'Other'
    df['c_charge_desc'] = df['c_charge_desc'].apply(charge_group)

    return df