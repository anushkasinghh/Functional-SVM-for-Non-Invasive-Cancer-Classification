"""
Run blind set evaluation for all 18 PCA configs.
Saves results to eval_result_data/svm_blind_results_all.csv
"""

import sys, pickle
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, '..')
from SVM_implement import SVMBreathClassifier
from sr_preprocessing import extract_sr_window, preprocess_sr, SR_CENTERS

DATA_PATH   = '../../data_processed/breath_data.pkl'
PARAMS_PATH = '../eval_result_data/all_configs_best_params.csv'
OUT_PATH    = '../eval_result_data/svm_blind_results_all.csv'

TASKS = {
    'H_vs_PC':        ['H', 'PC'],
    'H_vs_KC_BC_PC':  ['H', 'KC', 'BC', 'PC'],
}

def clopper_pearson(k, n, alpha=0.05):
    lo = float(beta_dist.ppf(alpha/2,   k,   n-k+1)) if k > 0 else 0.0
    hi = float(beta_dist.ppf(1-alpha/2, k+1, n-k))   if k < n else 1.0
    return round(lo, 4), round(hi, 4)

def get_sr_matrix(df, sr_name):
    center = SR_CENTERS[sr_name]
    rows = []
    for _, row in df.iterrows():
        sr_spec, _ = extract_sr_window(row['intensity_baseline_corrected'],
                                       row['wavenumber'], center=center, window_width=30.0)
        rows.append(preprocess_sr(sr_spec)['preprocessed'])
    return np.array(rows)

def get_concat_matrix(df):
    parts = [get_sr_matrix(df, sr) for sr in sorted(SR_CENTERS.keys())]
    return np.concatenate(parts, axis=1)

# ── Load and split data ───────────────────────────────────────────────────────
with open(DATA_PATH, 'rb') as f:
    df_raw = pickle.load(f)

df_raw['infoP'] = df_raw['infoP'].apply(lambda x: 'H' if x in ['M', 'F', 'H'] else x)
df_raw = df_raw.drop_duplicates(subset='original_filename').reset_index(drop=True)
df_train = df_raw[df_raw['category'] != 'blinddata'].reset_index(drop=True)
df_blind  = df_raw[df_raw['category'] == 'blinddata'].reset_index(drop=True)
print(f"Train: {len(df_train)}  Blind: {len(df_blind)}")

# ── Load params ───────────────────────────────────────────────────────────────
params_df = pd.read_csv(PARAMS_PATH)
pca_params = params_df[(params_df['feature_type'] == 'pca') & (params_df['status'] == 'done')]

clf = SVMBreathClassifier()
rows = []

for _, cfg in pca_params.iterrows():
    task      = cfg['task']
    sr_col    = cfg['sr_col']
    sr_name   = sr_col.replace('_preprocessed', '')
    config_id = cfg['config_id']
    is_concat = (sr_name == 'all')

    task_classes = TASKS[task]
    collapse     = len(task_classes) > 2   # multi-cancer → binary

    # Filter to task classes
    tr = df_train[df_train['infoP'].isin(task_classes)].reset_index(drop=True)
    bl = df_blind[df_blind['infoP'].isin(task_classes)].reset_index(drop=True)

    y_tr = np.where(tr['infoP'] == 'H', 'H', 'cancer') if collapse else tr['infoP'].values
    y_bl = np.where(bl['infoP'] == 'H', 'H', 'cancer') if collapse else bl['infoP'].values

    # Features
    if is_concat:
        X_tr = get_concat_matrix(tr)
        X_bl = get_concat_matrix(bl)
    else:
        X_tr = get_sr_matrix(tr, sr_name)
        X_bl = get_sr_matrix(bl, sr_name)

    params = {
        'sigma':            cfg['best_sigma'],
        'kernel':           cfg['best_kernel'],
        'C':                cfg['best_C'],
        'gamma':            cfg['best_gamma'],
        'degree':           cfg['best_degree'] if cfg['best_kernel'] == 'poly' else None,
        'feature_type':     'pca',
        'n_pca_components': int(cfg['n_pca_components']),
    }

    res = clf.blind_set_evaluation(X_tr, y_tr, X_bl, y_bl, params)
    tp, tn, fp, fn = res['TP'], res['TN'], res['FP'], res['FN']
    mcc = matthews_corrcoef(y_bl, np.where(res['decisions'] > 0,
                            'cancer' if collapse else 'PC', 'H'))
    sens_ci = clopper_pearson(tp, tp + fn)
    spec_ci = clopper_pearson(tn, tn + fp)

    # LOOCV accuracy from params CSV
    loocv_row = pca_params[
        (pca_params['config_id'] == config_id) & (pca_params['task'] == task)
    ]

    # Get LOOCV acc from svm_evaluation_results.csv
    rows.append({
        'config_id':   config_id,
        'task':        task,
        'sr_used':     sr_name,
        'n_blind':     len(y_bl),
        'accuracy':    res['accuracy'],
        'sensitivity': res['sensitivity'],
        'specificity': res['specificity'],
        'mcc':         mcc,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'sens_ci_lo':  sens_ci[0], 'sens_ci_hi': sens_ci[1],
        'spec_ci_lo':  spec_ci[0], 'spec_ci_hi': spec_ci[1],
        'cv_accuracy': cfg['cv_accuracy'],
    })
    print(f"  {config_id}: acc={res['accuracy']:.3f} sens={res['sensitivity']:.3f} spec={res['specificity']:.3f} mcc={mcc:.3f}")

# Merge LOOCV accuracy from evaluation results
eval_df = pd.read_csv('../eval_result_data/svm_evaluation_results.csv')
loocv_df = eval_df[(eval_df['pca'] == 1) & (eval_df['method'] == 'LOOCV')][
    ['config_id', 'task', 'accuracy']
].rename(columns={'accuracy': 'loocv_accuracy'})

out = pd.DataFrame(rows)
out = out.merge(loocv_df, on=['config_id', 'task'], how='left')
out['gap'] = out['loocv_accuracy'] - out['accuracy']
out.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")
print(out[['task', 'sr_used', 'accuracy', 'sensitivity', 'specificity', 'mcc', 'loocv_accuracy', 'gap']].to_string())
