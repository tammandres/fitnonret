"""Check how much the logistic model's precision-recall and positivity curves
vary depending on the train/test split.

The publication (see models.R / plots.py) uses a single temporal 10% test set.
Here we instead create the 10% test sets with repeated 10-fold cross-validation,
fit an unpenalised logistic model (sklearn) with the same dummy coding as
models.R, and overlay the resulting curves - one line per test fold - so the
split-to-split variation is visible.

This script was created with Claude Code to speed up the process,
using other scripts in the repository as a blueprint.
It was manually checked to be valid.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score


data_path = Path(r'Z:\fit_nonreturn_paper_20250417\data')
out_path = Path(r'Z:\fit_nonreturn_paper_20250417\results_fixed')
out_path.mkdir(exist_ok=True)

outcome = 'nonret2_days70'   # 70-day non-return, type 2
n_splits = 10                # 10-fold CV -> 10% test sets
random_state = 0
n_repeats = 10               # Number of repeats for 10-fold CV


# ---- 1. Read data and subset to patients with 70-day fu ----
df = pd.read_csv(data_path / 'first_fit_nonret_fixed.csv')
print(df.shape)

# Keep patients with at least 70-day follow-up
df = df.loc[df.fit_request_date_fu >= 70]

# Drop patients who died before 70-day follow-up without returning a test
# Note: this is for compatibility with older code; the fit_request_date_fu already drops them
df = df.loc[~((df.days_to_death <= 70) & (df.censored == 1))]

df = df.reset_index(drop=True)
print('Analysis set:', df.shape)
assert df[outcome].isna().sum() == 0, 'Outcome should be fully observed at 70-day fu'


# ---- 2. Build design matrix with the same dummy coding as in models.R ----
def build_design(data):
    d = data.copy()

    d['imd_quintile_str'] = d['imd_quintile'].map(
        lambda x: 'Not known' if pd.isna(x) else str(int(round(x))))
    d['request_year_str'] = d['request_year'].astype(int).astype(str)
    d['request_month_str'] = d['request_month'].astype(int).astype(str)

    cat_specs = [
        ('age_group', '18-39', 'age_group'),
        ('ethnicity', 'White', 'ethnicity'),
        ('imd_quintile_str', '5', 'imd_quintile_factor'),
        ('request_year_str', '2017', 'request_year_factor'),
        ('request_month_str', '1', 'request_month_factor'),
    ]

    blocks = [d[['gender_male']].astype(float)]
    for col, ref, prefix in cat_specs:
        dummies = pd.get_dummies(d[col].astype(str), prefix=prefix, prefix_sep='')
        ref_col = prefix + ref
        assert ref_col in dummies.columns, f'Reference level {ref_col} not found'
        dummies = dummies.drop(columns=[ref_col])  # drop reference level (the reference level is implicit in logistic regression)
        blocks.append(dummies.astype(float))

    X = pd.concat(blocks, axis=1)
    return X


X = build_design(df)
y = df[outcome].astype(int).to_numpy()
X_mat = X.to_numpy()
print('Design matrix:', X.shape)
print('Overall non-return rate:', round(y.mean() * 100, 2), '%')


# ---- 3. Helper function for curve data (mirrors summarise_performance in plots.py) ----
def pr_positivity(y_true, y_pred):
    """Return recall (%), precision (%), and percent-predicted-positive (%),
    plus AUC, average precision and prevalence, for one test set."""
    precision, recall, thr = precision_recall_curve(y_true, y_pred)

    # Percent of patients that would test positive at each threshold.
    # precision/recall have length len(thr)+1; the final point (no threshold,
    # recall=0) predicts nobody positive.
    order = np.sort(y_pred)
    n = len(y_pred)
    pos = np.zeros_like(recall, dtype=float)
    pos[:len(thr)] = n - np.searchsorted(order, thr, side='left')
    ppos = pos / n * 100

    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return recall * 100, precision * 100, ppos, auc, ap, y_true.mean()


# ---- 4. Repeated 10-fold CV: fit unpenalised logistic, predict on each fold ----
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

pred_records = []   # out-of-fold predictions
curves = []         # one entry per test fold
metrics = []        # auc / ap per fold

for i, (train_idx, test_idx) in enumerate(rkf.split(X_mat, y)):
    rep = i // n_splits
    fold = i % n_splits

    model = LogisticRegression(penalty=None, solver='lbfgs',
                               max_iter=1000, fit_intercept=True)
    model.fit(X_mat[train_idx], y[train_idx])
    p = model.predict_proba(X_mat[test_idx])[:, 1]

    y_test = y[test_idx]
    recall, precision, ppos, auc, ap, prev = pr_positivity(y_test, p)

    # Percent of patients testing positive at 50% sensitivity (recall = 50%).
    # recall is decreasing, so reverse for np.interp (needs increasing x).
    ppos_at_50 = np.interp(50.0, recall[::-1], ppos[::-1])

    curves.append({'rep': rep, 'fold': fold,
                   'recall': recall, 'precision': precision, 'ppos': ppos,
                   'prevalence': prev})
    metrics.append({'rep': rep, 'fold': fold, 'n_test': len(test_idx),
                    'prevalence': round(prev * 100, 2),
                    'auc': round(auc, 4), 'ap': round(ap, 4),
                    'ppos_at_50sens': round(ppos_at_50, 2)})

    rec = df.loc[test_idx, ['patient_id', 'icen']].copy()
    rec['rep'] = rep
    rec['fold'] = fold
    rec['y_true'] = y_test
    rec['y_pred'] = p
    pred_records.append(rec)

    print(f'rep {rep} fold {fold}: n_test={len(test_idx)}, '
          f'prev={prev*100:.1f}%, AUC={auc:.3f}, AP={ap:.3f}')

predictions = pd.concat(pred_records, axis=0).reset_index(drop=True)
metrics = pd.DataFrame(metrics)

# Save predictions and per-fold metrics
#predictions.to_csv(out_path / 'check_test_set_stability_predictions.csv', index=False)
metrics.to_csv(out_path / 'check_test_set_stability_metrics.csv', index=False)

print('\nAcross folds:')
print('  AUC: mean={:.3f}, sd={:.3f}, range=[{:.3f}, {:.3f}]'.format(
    metrics.auc.mean(), metrics.auc.std(), metrics.auc.min(), metrics.auc.max()))
print('  AP : mean={:.3f}, sd={:.3f}, range=[{:.3f}, {:.3f}]'.format(
    metrics.ap.mean(), metrics.ap.std(), metrics.ap.min(), metrics.ap.max()))
print('  %% positive at 50%% sensitivity: mean={:.2f}, min={:.2f}, max={:.2f}'.format(
    metrics.ppos_at_50sens.mean(), metrics.ppos_at_50sens.min(), metrics.ppos_at_50sens.max()))


# ---- 5. Overlay curves across folds/repeats ----
prev_overall = y.mean()

fig, ax = plt.subplots(1, 2, figsize=(9, 5), tight_layout=True)
ax = ax.flatten()

for j, c in enumerate(curves):
    # Curves for all data splits share the same colour
    # We label them only once to avoid repeating the label across splits
    label = 'CV test folds' if j == 0 else None 

    # Panel A: precision-recall (plot from recall > 1% as below that there is more noise)
    mask = c['recall'] > 1  # recall > 0.01
    ax[0].plot(c['recall'][mask], c['precision'][mask], color='C0', alpha=0.4, linewidth=1, label=label)

    # Panel B: positivity curve (percent testing positive vs sensitivity)
    ax[1].plot(c['recall'], c['ppos'], color='C0', alpha=0.4, linewidth=1, label=label)

# Panel A design
ax[0].grid(alpha=0.5)
ax[0].set_xticks(np.arange(0, 110, 10))
ax[0].set_yticks(np.arange(0, 110, 10))
ax[0].set(xlim=(-5, 105), ylim=(-5, 105),
          xlabel='Percent of nonreturns detected\n(Sensitivity)',
          ylabel='Percent of nonreturns among patients testing positive\n(Positive predictive value)')
ax[0].set_title('A. Precision-recall curves across test folds')
ax[0].legend(frameon=False, fontsize=8)

# Panel B design + reference lines (perfect / random), using overall prevalence
ax[1].plot([0, 100], [0, prev_overall * 100], linestyle='solid', color='C1',
           label='Perfect performance')
ax[1].plot([0, 100], [0, 100], linestyle='dashed', color='red',
           label='Random performance')
ax[1].axvline(50, color='gray', linestyle='dotted', alpha=0.7)  # 50% sensitivity
ax[1].grid(alpha=0.5)
ax[1].set_xticks(np.arange(0, 110, 10))
ax[1].set_yticks(np.arange(0, 110, 10))
ax[1].set(xlim=(-5, 105), ylim=(-5, 105),
          xlabel='Percent of nonreturns detected\n(Sensitivity)',
          ylabel='Percent of patients testing positive')
ax[1].set_title('B. Positivity curves across test folds')
ax[1].legend(frameon=False, fontsize=8)

out_name = f'check_test_set_stability_{outcome}_reps{n_repeats}.png'
plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
print('\nSaved figure:', out_path / out_name)
