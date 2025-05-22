# Fairness in ML 

As part of my Master’s thesis, this project conducts a comprehensive comparative evaluation of bias mitigation methods across three stages of the machine learning pipeline leveraging the [IBM AIF360 toolkit](https://aif360.readthedocs.io/en/latest/index.html). 

It applies selected techniques of each pipeline stage:
- **Pre-processing**: Reweighing, Disparate Impact Remover 
- **In-processing**: Adversarial Debiasing, Prejudice Remover  
- **Post-processing**: Equalized Odds Postprocessing, Reject Option Classification

These methods are applied to the UCI Adult Income and ProPublica COMPAS recidivism datasets. For each configuration, the project computes both fairness metrics (Statistical Parity Difference, Disparate Impact, Equal Opportunity Difference, Average Odds Difference) and performance metrics (accuracy, F1-score), generates visual comparisons, and saves reproducible outputs for downstream analysis.

If you are interested in seeing the full thesis (Mitigating Algorithmic Bias in Machine Learning: A Comparative Evaluation of Pre-, In-, and Post-Processing Methods Using AIF360 and Practitioner Perspectives on Fairness), please contact me at endres.jea@gmail.com.

## Project Structure

* **notebooks/**
  Jupyter notebooks for exploration and demonstration of baseline models, mitigation approaches, metrics, divided by datasets

* **src/**
  Four Python modules:
  * `data_preprocessing.py`: implement preprocessing for both datasets
  * `data_loading.py`: load and preprocess the datasets
  * `metrics.py`: compute fairness metrics (SPD, DI, EOD, AOD) and performance metrics (accuracy, F1-score), visualize and save metrics
  * `modeling.py`: orchestrate training and evaluation across pipeline stages (baseline + mitigation methods)

* **reports/**
  Directory for output figures (`plots_*`)  divided by datasets and CSV results (`baseline_agg/`)

* **requirements.txt**
  Project dependencies for reproducible setup

* **.gitignore**
  Specifies files and directories to be ignored by Git

## Prerequisites

* Python 3.11
* [IBM AIF360](https://github.com/Trusted-AI/AIF360) installed
* Dependencies: `pandas`, `scikit-learn`, `matplotlib`

```bash
pip install -r requirements.txt
```

## Data Access 
The datasets were pulled using AIF360 build in function, requiring the following downloads.
Needed directories may vary based on your setup. Once these are downloaded, the scripts should run.

### Adult Income Dataset

```bash
mkdir -p /opt/anaconda3/lib/python3.11/site-packages/aif360/data/raw/adult
cd    /opt/anaconda3/lib/python3.11/site-packages/aif360/data/raw/adult

curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
```

### COMPAS Dataset

```bash
mkdir -p /opt/anaconda3/lib/python3.11/site-packages/aif360/data/raw/compas
cd    /opt/anaconda3/lib/python3.11/site-packages/aif360/data/raw/compas

curl -L https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv \
  -o compas-scores-two-years.csv
```

## Usage Examples
```python
# General setup at begining of the notebook 
dataset_name = 'compas'
mitigation_name   = 'reweighing'
pipeline_stage    = 'preprocessing'   
out_dir_plots    = '../../reports/plots_compas'
```

### Save Metrics
```python
# Raw and aggregated results can then be saved using the functions defined in metrics.py
# 1) Aggregated metrics
save_agg_metrics(
    dataset_name     = dataset_name,
    mitigation_name  = mitigation_name,
    race_agg_df      = compas_race_metrics_agg,
    sex_agg_df       = compas_sex_metrics_agg,
    pipeline_stage   = pipeline_stage
)

# 2) Raw metrics
save_raw_metrics(
    dataset_name    = dataset_name,
    race_raw_df     = compas_race_metrics,
    sex_raw_df      = compas_sex_metrics,
    pipeline_stage  = pipeline_stage
)
```

### Plot Comparison
```python
# Example plot comapring two runs, save to reports and show
plot_title = 'Compas Preprocessing Reweighing: Baseline - Sex'
fig = compare_viz_metrics_2x3(baseline_sex_agg, compas_sex_metrics_agg, 'Baseline', 'Sex', plot_title)
fname    = plot_title.replace(' ', '_').replace('(', '').replace(')', '')
out_path = os.path.join(out_dir_plots, f'{fname}.png')
fig.savefig(out_path)
fig.show()
```

### Baseline Aggregates Exception
All mitigation methods are compared against the dataset's baseline, the baseline results are saved as individual csv's for easy retrieval. This structure matches the structure expectyed by the plotting function.
```python
# COMPAS example 
# storage
compas_race_metrics_agg.to_csv('../../reports/baseline_agg/compas_race_metrics_agg.csv', index=True)
compas_sex_metrics_agg.to_csv('../../reports/baseline_agg/compas_sex_metrics_agg.csv', index=True)

# retrieval
baseline_race_agg = pd.read_csv('../../reports/baseline_agg/compas_race_metrics_agg.csv', index_col=0)
baseline_sex_agg = pd.read_csv('../../reports/baseline_agg/compas_sex_metrics_agg.csv', index_col=0)
```

## General Implementation Structure
Next to the exploratory parts, all implementations follow a three part structure:
```python
# 1) Retrieve data
protected = 'race'
privileged_value   = 0.0
unprivileged_value = 1.0

cd, df = load_compas_race()
feature_cols = [c for c in df.columns if c not in ('label','race')]

# 2) Run experiment and Evaluate
sss = StratifiedShuffleSplit(n_splits=25, test_size=0.2, random_state=42)
results = []

for train_idx, test_idx in sss.split(df, df['label']):
    test_df, y_test, y_pred = reweighing_train_and_predict(
        cd, df,
        train_idx, test_idx,
        protected, privileged_value, unprivileged_value
    )
    m = compute_metrics(
        test_df, y_test, y_pred,
        protected, privileged_value, unprivileged_value
    )
    results.append(m)

# 3) Aggregate results
compas_race_metrics = pd.DataFrame(results)
compas_race_metrics_agg = compas_race_metrics.agg(['mean', 'std'])
print(compas_race_metrics_agg)
```

## License

This project is distributed under the [MIT License](LICENSE).  
See the LICENSE file for full details.
