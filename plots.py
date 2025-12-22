import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

#plt.style.use('ggplot')
data_path = Path(r'Z:\fit_nonreturn_paper_20250417\data')
out_path = Path(r'Z:\fit_nonreturn_paper_20250417\results')
in_path = out_path

hide_small_km_counts = True

# Helper functions
def summarise_performance(y_true, y_pred, mod_name):
    
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred) 
    m = pd.DataFrame({'outcome': o, 'ap': ap, 'auc': auc}, index=[0])

    precision, recall, thr = precision_recall_curve(y_true, y_pred)
    thr = np.concatenate([thr, [np.nan]])
    pr = pd.DataFrame({'precision': precision, 'recall': recall, 'thr': thr})
    pr['pos'] = pr.apply(lambda x: (y_pred >= x['thr']).sum(), axis=1)
    pr['ppos'] = pr.pos / len(y_pred) * 100
    pr['ppos_perf'] = pr.recall * y_true.mean() * 100
    pr['outcome'] = o

    fig, ax = plt.subplots(1, 2, figsize=(9, 5), tight_layout=True)
    ax = ax.flatten()
    
    mask = pr.recall > 0.01
    ax[0].plot(pr.recall[mask] * 100, pr.precision[mask] * 100, color='C0', label=mod_name)
    ax[0].grid(alpha=0.5)
    ax[0].set_xticks(np.arange(0, 110, 10))
    ax[0].set_yticks(np.arange(0, 110, 10))
    ax[0].set(xlabel='Percent of nonreturns detected\n(Sensitivity)', 
              ylabel='Percent of nonreturns among patients testing positive\n(Positive predictive value)')
    ax[0].legend(frameon=False)
    ax[0].set_title('A. Precision-recall curve')
    ax[0].set(xlim=(-5, 105), ylim=(-5, 105))

    ax[1].plot(pr.recall * 100, pr.ppos, color='C0', label=mod_name)
    ax[1].plot(pr.recall * 100, pr.ppos_perf, linestyle='solid', color='C1', label='Perfect performance')
    ax[1].plot(np.array([0, 100]), np.array([0, 100]), linestyle='dashed', color='red', label='Random performance')
    ax[1].grid(alpha=0.5)
    ax[1].set_xticks(np.arange(0, 110, 10))
    ax[1].set_yticks(np.arange(0, 110, 10))
    ax[1].set(xlabel='Percent of nonreturns detected\n(Sensitivity)', 
              ylabel='Percent of patients testing positive')
    ax[1].legend(frameon=False)
    ax[1].set(xlim=(-5, 105), ylim=(-5, 105))
    ax[1].set_title('B. Positivity curve')

    return m, pr, fig


# ---- Plot logistic odds ratios
#region
df = pd.read_csv(in_path / 'logistic-coef.csv')

outcome = df.outcome.unique()

name_map = {
    'gender_male': 'Male sex (vs female)', 
    'age_group40-49': 'Age 40-49 (vs 18-39)', 
    'age_group50-59': 'Age 50-59 (vs 18-39)',
    'age_group60-69': 'Age 60-69 (vs 18-39)',
    'age_group70-79': 'Age 70-79 (vs 18-39)',
    'age_group80-89': 'Age 80-89 (vs 18-39)',
    'age_group90+': 'Age 90+ (vs 18-39)',
    'ethnicityAsian': 'Ethnicity asian (vs white)', 
    'ethnicityBlack': 'Ethnicity black (vs white)', 
    'ethnicityMixed': 'Ethnicity mixed (vs white)', 
    'ethnicityNot known': 'Ethnicity not known (vs white)', 
    'ethnicityNot stated': 'Ethnicity not stated (vs white)', 
    'ethnicityOther': 'Ethnicity other (vs white)', 
    'imd_quintile_factor1': 'IMD quintile 1 (vs least deprived 5)', 
    'imd_quintile_factor2': 'IMD quintile 2 (vs least deprived 5)', 
    'imd_quintile_factor3': 'IMD quintile 3 (vs least deprived 5)', 
    'imd_quintile_factor4': 'IMD quintile 4 (vs least deprived 5)', 
    'imd_quintile_factorNot known': 'IMD quintile not known (vs least deprived 5)', 
    'request_year_factor2018': 'Request year 2018 (vs 2017)', 
    'request_year_factor2019': 'Request year 2019 (vs 2017)', 
    'request_year_factor2020': 'Request year 2020 (vs 2017)', 
    'request_year_factor2021': 'Request year 2021 (vs 2017)', 
    'request_year_factor2022': 'Request year 2022 (vs 2017)', 
    'request_year_factor2023': 'Request year 2023 (vs 2017)', 
    'request_month_factor2': 'Request month 2 (vs 1)', 
    'request_month_factor3': 'Request month 3 (vs 1)', 
    'request_month_factor4': 'Request month 4 (vs 1)', 
    'request_month_factor5': 'Request month 5 (vs 1)', 
    'request_month_factor6': 'Request month 6 (vs 1)', 
    'request_month_factor7': 'Request month 7 (vs 1)', 
    'request_month_factor8': 'Request month 8 (vs 1)', 
    'request_month_factor9': 'Request month 9 (vs 1)', 
    'request_month_factor10': 'Request month 10 (vs 1)', 
    'request_month_factor11': 'Request month 11 (vs 1)', 
    'request_month_factor12': 'Request month 12 (vs 1)'
}

for o in outcome:
    dfsub = df.loc[df.outcome == o]
    dfsub = dfsub.iloc[::-1]
    dfsub = dfsub.loc[dfsub.Predictor != '(Intercept)']
    dfsub.Predictor = dfsub.Predictor.replace(name_map)

    fig, ax = plt.subplots(figsize=(4, 9))
    dfsub['add_high'] = dfsub['OR upp'] - dfsub['OR']
    dfsub['add_low'] = dfsub['OR'] - dfsub['OR low']
    err = dfsub[['add_high', 'add_low']].abs().transpose()
    ax.errorbar(x=dfsub.OR, y=dfsub.Predictor, xerr=err, ls='none', fmt='', color='black')
    ax.scatter(x=dfsub.OR, y=dfsub.Predictor, s=28, color='black')
    ax.vlines(x=1, ymin=0, ymax=len(dfsub)-1, color='red', linestyle='dotted', alpha=1)
    ax.grid(which='major', alpha=0.25, color='gray')
    logscale = True
    if logscale:
        ax.set_xscale('log', base=2)
        xticks = [1/4, 1/3.5, 1/3, 1/2.5, 1/2, 1/1.5, 1, 1.5, 2]
        xticklabels = ['1/4', None, '1/3', None, '1/2', None, '1', None, '2']
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    ax_len = len(dfsub)-1
    delta = 0.025 * ax_len
    ax.set(ylim=(-delta, ax_len + delta), xlabel='Odds ratio (95% CI)')

    out_name = 'logistic_coef_' + str(o) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')
#endregion

# ---- Plot predictive power of the logistic model
#region
df = pd.read_csv(out_path / 'logistic-pred.csv')
#demo = pd.read_csv(data_path / 'first_fit_nonret.csv')
#demo = demo[['patient_id', 'ethnicity', 'age_group']]

outcome = df.outcome.unique()
metrics = pd.DataFrame()
pr_curve = pd.DataFrame()
for o in outcome:
    print(o)
    dfsub = df.loc[(df.outcome == o) & (df.test_set == 1)]
    print(dfsub.shape[0])
    #dfsub = dfsub.merge(demo, how='left')
    #print(dfsub.shape[0])
    #dfsub = dfsub.loc[dfsub.age_group == '18-39']

    y_true, y_pred = dfsub.y_true, dfsub.y_pred
    m, pr, fig = summarise_performance(y_true, y_pred, 'Logistic regression')
    plt.savefig(out_path / ('logistic_pred_' + str(o) + '.png'), dpi=300, bbox_inches='tight')
    plt.close()

    metrics = pd.concat(objs=[metrics, m], axis=0)
    pr_curve = pd.concat(objs=[pr_curve, pr], axis=0)

metrics = metrics.reset_index(drop=True)
metrics.to_csv(out_path / 'logistic_metrics.csv', index=False)
pr_curve.to_csv(out_path / 'logistic_prcurve.csv', index=False)


df = pd.read_csv(out_path / 'logistic-pred-mult.csv')
#demo = pd.read_csv(data_path / 'first_fit_nonret.csv')
#demo = demo[['patient_id', 'ethnicity', 'age_group']]

outcome = df.outcome.unique()
metrics = pd.DataFrame()
pr_curve = pd.DataFrame()
for o in outcome:
    print(o)
    dfsub = df.loc[(df.outcome == o) & (df.test_set == 1)]
    print(dfsub.shape[0])
    #dfsub = dfsub.merge(demo, how='left')
    #print(dfsub.shape[0])
    #dfsub = dfsub.loc[dfsub.age_group == '18-39']

    y_true, y_pred = dfsub.y_true, dfsub.y_pred
    m, pr, fig = summarise_performance(y_true, y_pred, 'Logistic regression')
    plt.savefig(out_path / ('logistic_pred_mult_' + str(o) + '.png'), dpi=300, bbox_inches='tight')
    plt.close()

    metrics = pd.concat(objs=[metrics, m], axis=0)
    pr_curve = pd.concat(objs=[pr_curve, pr], axis=0)

metrics = metrics.reset_index(drop=True)
metrics.to_csv(out_path / 'logistic_mult_metrics.csv', index=False)
pr_curve.to_csv(out_path / 'logistic_mult_prcurve.csv', index=False)
#endregion

# ---- Plot predictive power of GAM
#region
df = pd.read_csv(out_path / 'gam-pred.csv')
#demo = pd.read_csv(data_path / 'first_fit_nonret.csv')
#demo = demo[['patient_id', 'ethnicity', 'age_group']]

outcome = df.outcome.unique()
metrics = pd.DataFrame()
pr_curve = pd.DataFrame()
for o in outcome:
    print(o)
    dfsub = df.loc[(df.outcome == o) & (df.test_set == 1)]
    dfsub = dfsub[~dfsub.y_pred.isna()]
    print(dfsub.shape[0])
    #dfsub = dfsub.merge(demo, how='left')
    #print(dfsub.shape[0])
    #dfsub = dfsub.loc[dfsub.age_group == '18-39']

    y_true, y_pred = dfsub.y_true, dfsub.y_pred
    m, pr, fig = summarise_performance(y_true, y_pred, 'GAM')
    plt.savefig(out_path / ('gam_pred_' + str(o) + '.png'), dpi=300, bbox_inches='tight')
    plt.close()

    metrics = pd.concat(objs=[metrics, m], axis=0)
    pr_curve = pd.concat(objs=[pr_curve, pr], axis=0)

metrics = metrics.reset_index(drop=True)
metrics.to_csv(out_path / 'gam_metrics.csv', index=False)
pr_curve.to_csv(out_path / 'gam_prcurve.csv', index=False)

#endregion

# ---- Plot predictive power of the Cox model
#region
df = pd.read_csv(out_path / 'cox-pred.csv')
df_logistic = pd.read_csv(out_path / 'logistic-pred.csv')

outcome = df.outcome.unique()
metrics = pd.DataFrame()
pr_curve = pd.DataFrame()
for o in outcome:
    print(o)
    dfsub = df.loc[(df.outcome == o) & (df.test_set == 1)]
    dfsub_logistic = df_logistic.loc[(df_logistic.outcome == o) & (df_logistic.test_set == 1)]

    y_true, y_pred = dfsub.y_true, dfsub.y_pred
    m, pr, fig = summarise_performance(y_true, y_pred, 'Cox proportional hazards')
    plt.savefig(out_path / ('cox_pred_' + str(o) + '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    metrics = pd.concat(objs=[metrics, m], axis=0)
    pr_curve = pd.concat(objs=[pr_curve, pr], axis=0)

    df_compare = dfsub[['patient_id', 'icen', 'y_true', 'y_pred']]\
        .merge(dfsub_logistic[['patient_id', 'icen', 'y_pred']].rename(columns={'y_pred': 'y_pred_logistic'}),
               how='inner')

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df_compare.y_pred_logistic, df_compare.y_pred, alpha=0.5)
    ax.plot([0, 0.5], [0, 0.5], color='red', linestyle='solid', alpha=1)
    ax.set(xlim=(-0.05, 0.55), ylim=(-0.05, 0.55),
           xlabel='Probability of test nonreturn (logistic)',
           ylabel='Proability of test nonreturn (Cox)')
    ax.grid(which='major', alpha=0.5)
    out_name = 'cox_vs_logistic_' + str(o) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')

out_name = 'cox_metrics.csv'
cox_metrics = metrics.reset_index(drop=True)
cox_metrics.to_csv(out_path / out_name, index=False)

out_name = 'cox_prcurve.csv'
pr_curve = pr_curve.reset_index(drop=True)
pr_curve.to_csv(out_path / out_name, index=False)
#endregion

# ---- KM curves
#region
df = pd.read_csv(out_path / 'km_curve.csv')

return_types = df.return_type.unique()
for r in return_types:
    print(r)
    dfsub = df.loc[(df.return_type == r)]
    
    if hide_small_km_counts:
        dfsub = dfsub.loc[dfsub.n_event >= 10]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(dfsub.time, dfsub.surv, where='post')
    ax.fill_between(dfsub.time, dfsub.surv_low, dfsub.surv_upp, step='post', alpha=0.25, facecolor='C0')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticks(np.arange(0, 75, 5))
    ax.set(xlim=[0 - 70*0.05, 70 * 1.05], ylim=[-0.05, 1.05],
           xlabel='Days from FIT test request (t)',
           ylabel='Probability of test return after t days')
    ax.grid(which='major', alpha=0.5)
    out_name = 'km_curve_type' + str(r) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(dfsub.time, 1 - dfsub.surv, where='post')
    ax.fill_between(dfsub.time, 1-dfsub.surv_low, 1-dfsub.surv_upp, step='post', alpha=0.25, facecolor='C0')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticks(np.arange(0, 75, 5))
    ax.set(xlim=[0 - 70*0.05, 70 * 1.05], ylim=[-0.05, 1.05],
           xlabel='Days from FIT test request (t)',
           ylabel='Probability of test return within t days')
    ax.grid(which='major', alpha=0.5)
    out_name = 'survival_curve_type' + str(r) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = {'p14': 'Probability of return within 14 days from request date',
              'p28': 'Probability of return within 28 days from request date',
              'p70': 'Probability of return within 70 days from request date'}
    for col in ['p14', 'p28', 'p70']:
        ax.plot(dfsub.time, dfsub[col], label=labels[col]) #, where='post')
        ax.scatter(dfsub.time, dfsub[col], s=10) #, where='post')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticks(np.arange(0, 75, 5))
    ax.set(xlim=[0 - 70*0.05, 70 * 1.05], ylim=[-0.05, 1.05],
        xlabel='Days from FIT test request (t)',
        ylabel='Probability of test return\nif not yet returned in t days')
    ax.grid(which='major', alpha=0.5)
    ax.legend(frameon=False)

    out_name = 'conditional_curve_type' + str(r) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')


var_map = {'gender_male': 'Gender',
           'age_group': 'Age group',
           'ethnicity': 'Ethnicity',
           'imd_quintile_factor': 'IMD quintile'}
df = pd.read_csv(out_path / 'km_curve_grouped.csv')
variables = df.variable.unique()
len(variables)

return_types = df.return_type.unique()
for r in return_types:
    print(r)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.flatten()

    for j, v in enumerate(variables):
        dfsub = df.loc[(df.return_type == r) & (df.variable == v)]
        if hide_small_km_counts:
            dfsub = dfsub.loc[dfsub.n_event >= 10]

        values = dfsub.value.sort_values().unique()
        for k, val in enumerate(values):
            dfsubsub = dfsub.loc[dfsub.value == val]

            if v == 'gender_male':
                if val == '0':
                    label = 'Not male'
                elif val == '1':
                    label = 'Male'
            else:
                label = val

            ax[j].step(dfsubsub.time, dfsubsub.surv, where='post', label=label, alpha=0.75, color='C' + str(k))
            ax[j].fill_between(dfsubsub.time, dfsubsub.surv_low, dfsubsub.surv_upp, alpha=0.25, 
                               facecolor='C' + str(k), edgecolor=None, step='post')
            ax[j].set_yticks(np.arange(0, 1.1, 0.1))
            ax[j].set_xticks(np.arange(0, 75, 5))
            ax[j].set(xlim=[0 - 70*0.05, 70 * 1.05], ylim=[-0.05, 1.05],
                xlabel='Days from FIT test request (t)',
                ylabel='Probability of test return after t days')
            ax[j].grid(which='major', alpha=0.5)
        ax[j].legend(frameon=False)
        ax[j].set_title(var_map[v])
    
    out_name = 'km_curve_grouped_type' + str(r) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')


    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.flatten()

    for j, v in enumerate(variables):
        dfsub = df.loc[(df.return_type == r) & (df.variable == v)]
        if hide_small_km_counts:
            dfsub = dfsub.loc[dfsub.n_event >= 10]

        values = dfsub.value.sort_values().unique()
        for k, val in enumerate(values):
            dfsubsub = dfsub.loc[dfsub.value == val]

            if v == 'gender_male':
                if val == '0':
                    label = 'Not male'
                elif val == '1':
                    label = 'Male'
            else:
                label = val

            ax[j].step(dfsubsub.time, dfsubsub.st, where='post', label=label, alpha=0.75, color='C' + str(k))
            ax[j].fill_between(dfsubsub.time, 1-dfsubsub.surv_low, 1-dfsubsub.surv_upp, alpha=0.25, 
                               facecolor='C' + str(k), edgecolor=None, step='post')
            ax[j].set_yticks(np.arange(0, 1.1, 0.1))
            ax[j].set_xticks(np.arange(0, 75, 5))
            ax[j].set(xlim=[0 - 70*0.05, 70 * 1.05], ylim=[-0.05, 1.05],
                xlabel='Days from FIT test request (t)',
                ylabel='Probability of test return within t days')
            ax[j].grid(which='major', alpha=0.5)
        ax[j].legend(frameon=False)
        ax[j].set_title(var_map[v])
    
    out_name = 'survival_curve_grouped_type' + str(r) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')


# KM panel curve
df = pd.read_csv(out_path / 'km_curve.csv')
df_return = pd.read_csv(out_path / 'return_time.csv')


return_types = df.return_type.unique()
for r in return_types:
    print(r)
    dfsub = df.loc[(df.return_type == r)]
    if hide_small_km_counts:
        dfsub = dfsub.loc[dfsub.n_event >= 10]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
    ax = ax.flatten()

    rsub = df_return.loc[(df_return.return_type == r) & (df_return.time <= 70)]

    ax[0].plot(rsub.time, rsub.perc)
    ax[0].scatter(rsub.time, rsub.perc, s=10)
    ax[0].set_yticks(np.arange(0, 110, 10))
    ax[0].set_xticks(np.arange(0, 75, 5))
    ax[0].set(xlabel='Days from FIT test request (t)',
           ylabel='Percent of tests returned',
           xlim=(0-70*0.05, 70*1.05), ylim=(-5, 105))
    ax[0].grid(which='major', alpha=0.5)
    ax[0].set_title('A. Return time among test returners')

    ax[1].step(dfsub.time, 1 - dfsub.surv, where='post')
    ax[1].fill_between(dfsub.time, 1-dfsub.surv_low, 1-dfsub.surv_upp, step='post', alpha=0.25, facecolor='C0')
    ax[1].set_yticks(np.arange(0, 1.1, 0.1))
    ax[1].set_xticks(np.arange(0, 75, 5))
    ax[1].set(xlim=[0 - 70*0.05, 70 * 1.05], ylim=[-0.05, 1.05],
           xlabel='Days from FIT test request (t)',
           ylabel='Probability of test return')
    ax[1].grid(which='major', alpha=0.5)
    ax[1].set_title('B. Overall probability of test return')

    labels = {'p14': 'Return within 14 days',
              'p28': 'Return within 28 days',
              'p70': 'Return within 70 days'}
    for col in ['p14', 'p28', 'p70']:
        ax[2].plot(dfsub.time, dfsub[col], label=labels[col]) #, where='post')
        ax[2].scatter(dfsub.time, dfsub[col], s=10) #, where='post')
    ax[2].set_yticks(np.arange(0, 1.1, 0.1))
    ax[2].set_xticks(np.arange(0, 75, 5))
    ax[2].set(xlim=[0 - 70*0.05, 70 * 1.05], ylim=[-0.05, 1.05],
        xlabel='Days from FIT test request (t)',
        ylabel='Probability of test return\nif not yet returned in t days')
    ax[2].grid(which='major', alpha=0.5)
    ax[2].legend(frameon=False)
    ax[2].set_title('C. Conditional probability of test return')

    out_name = 'return_time_panel_type' + str(r) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')


# KM panel curve - on full data (not training data), 
# and restricting the return time curve to 70-day fu for more meaningful estimate
df = pd.read_csv(out_path / 'km_curve_data-full.csv')
df_return = pd.read_csv(out_path / 'return_time_data-full_fu-70.csv')

return_types = df.return_type.unique()
for r in return_types:
    print(r)
    dfsub = df.loc[(df.return_type == r)]
    if hide_small_km_counts:
        dfsub = dfsub.loc[dfsub.n_event >= 10]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
    ax = ax.flatten()

    rsub = df_return.loc[(df_return.return_type == r) & (df_return.time <= 70)]

    ax[0].plot(rsub.time, rsub.perc)
    ax[0].scatter(rsub.time, rsub.perc, s=10)
    ax[0].set_yticks(np.arange(0, 110, 10))
    ax[0].set_xticks(np.arange(0, 75, 5))
    ax[0].set(xlabel='Days from FIT test request (t)',
           ylabel='Percent of tests returned',
           xlim=(0-70*0.05, 70*1.05), ylim=(-5, 105))
    ax[0].grid(which='major', alpha=0.5)
    ax[0].set_title('A. Return time among test returners')

    ax[1].step(dfsub.time, 1 - dfsub.surv, where='post')
    ax[1].fill_between(dfsub.time, 1-dfsub.surv_low, 1-dfsub.surv_upp, step='post', alpha=0.25, facecolor='C0')
    ax[1].set_yticks(np.arange(0, 1.1, 0.1))
    ax[1].set_xticks(np.arange(0, 75, 5))
    ax[1].set(xlim=[0 - 70*0.05, 70 * 1.05], ylim=[-0.05, 1.05],
           xlabel='Days from FIT test request (t)',
           ylabel='Probability of test return')
    ax[1].grid(which='major', alpha=0.5)
    ax[1].set_title('B. Overall probability of test return')

    labels = {'p14': 'Return within 14 days',
              'p28': 'Return within 28 days',
              'p70': 'Return within 70 days'}
    for col in ['p14', 'p28', 'p70']:
        ax[2].plot(dfsub.time, dfsub[col], label=labels[col]) #, where='post')
        ax[2].scatter(dfsub.time, dfsub[col], s=10) #, where='post')
    ax[2].set_yticks(np.arange(0, 1.1, 0.1))
    ax[2].set_xticks(np.arange(0, 75, 5))
    ax[2].set(xlim=[0 - 70*0.05, 70 * 1.05], ylim=[-0.05, 1.05],
        xlabel='Days from FIT test request (t)',
        ylabel='Probability of test return\nif not yet returned in t days')
    ax[2].grid(which='major', alpha=0.5)
    ax[2].legend(frameon=False)
    ax[2].set_title('C. Conditional probability of test return')

    out_name = 'return_time_panel_data-full_panel-a-70-day-fu_type' + str(r) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')


dfsub = df.loc[df.return_type == 2]
dfsub['p_return'] = 1 - dfsub.surv
s = dfsub.loc[dfsub.time.isin([10, 14, 28, 70, 365]), ['time', 'p_return', 'p14', 'p28', 'p70']]
s.p_return = s.p_return.round(3) * 100
s[['p14', 'p28', 'p70']] = s[['p14', 'p28', 'p70']].round(3) * 100
s


# KM panel curve - on full data (not training data), 
# and restricting the return time curve to 70-day fu for more meaningful estimate
# and including years 2023 and 2024
df = pd.read_csv(out_path / 'km_curve_data-full_years-23-24.csv')
df_return = pd.read_csv(out_path / 'return_time_data-full_fu-70_years-23-24.csv')

return_types = df.return_type.unique()
for r in return_types:
    print(r)
    dfsub = df.loc[(df.return_type == r)]
    if hide_small_km_counts:
        dfsub = dfsub.loc[dfsub.n_event >= 10]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
    ax = ax.flatten()

    rsub = df_return.loc[(df_return.return_type == r) & (df_return.time <= 70)]

    ax[0].plot(rsub.time, rsub.perc)
    ax[0].scatter(rsub.time, rsub.perc, s=10)
    ax[0].set_yticks(np.arange(0, 110, 10))
    ax[0].set_xticks(np.arange(0, 75, 5))
    ax[0].set(xlabel='Days from FIT test request (t)',
           ylabel='Percent of tests returned',
           xlim=(0-70*0.05, 70*1.05), ylim=(-5, 105))
    ax[0].grid(which='major', alpha=0.5)
    ax[0].set_title('A. Return time among test returners')

    ax[1].step(dfsub.time, 1 - dfsub.surv, where='post')
    ax[1].fill_between(dfsub.time, 1-dfsub.surv_low, 1-dfsub.surv_upp, step='post', alpha=0.25, facecolor='C0')
    ax[1].set_yticks(np.arange(0, 1.1, 0.1))
    ax[1].set_xticks(np.arange(0, 75, 5))
    ax[1].set(xlim=[0 - 70*0.05, 70 * 1.05], ylim=[-0.05, 1.05],
           xlabel='Days from FIT test request (t)',
           ylabel='Probability of test return')
    ax[1].grid(which='major', alpha=0.5)
    ax[1].set_title('B. Overall probability of test return')

    labels = {'p14': 'Return within 14 days',
              'p28': 'Return within 28 days',
              'p70': 'Return within 70 days'}
    for col in ['p14', 'p28', 'p70']:
        ax[2].plot(dfsub.time, dfsub[col], label=labels[col]) #, where='post')
        ax[2].scatter(dfsub.time, dfsub[col], s=10) #, where='post')
    ax[2].set_yticks(np.arange(0, 1.1, 0.1))
    ax[2].set_xticks(np.arange(0, 75, 5))
    ax[2].set(xlim=[0 - 70*0.05, 70 * 1.05], ylim=[-0.05, 1.05],
        xlabel='Days from FIT test request (t)',
        ylabel='Probability of test return\nif not yet returned in t days')
    ax[2].grid(which='major', alpha=0.5)
    ax[2].legend(frameon=False)
    ax[2].set_title('C. Conditional probability of test return')

    out_name = 'return_time_panel_data-full_panel-a-70-day-fu_years-23-24_type' + str(r) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')


dfsub = df.loc[df.return_type == 2]
dfsub['p_return'] = 1 - dfsub.surv
s = dfsub.loc[dfsub.time.isin([10, 14, 28, 70, 365]), ['time', 'p_return', 'p14', 'p28', 'p70']]
s.p_return = s.p_return.round(3) * 100
s[['p14', 'p28', 'p70']] = s[['p14', 'p28', 'p70']].round(3) * 100
s

#endregion

# ---- Proportion of patients who returned a test within a given time, of all returners
#region
df = pd.read_csv(out_path / 'return_time.csv')

return_types = df.return_type.unique()
for r in return_types:
    print(r)
    dfsub = df.loc[(df.return_type == r) & (df.time <= 70)]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(dfsub.time, dfsub.perc)
    ax.scatter(dfsub.time, dfsub.perc, s=10)
    ax.set_yticks(np.arange(0, 105, 5))
    ax.set_xticks(np.arange(0, 75, 5))
    ax.set(xlabel='Days from FIT test request (t)',
           ylabel='Percent of tests returned among returners',
           xlim=(0-70*0.05, 70*1.05), ylim=(0-100*0.05, 105))
    ax.grid(which='major', alpha=0.5)

    out_name = 'prob_return_among_return_type' + str(r) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')

#endregion

# ---- Plot GAMs
#region

# Plot gam effects on linear predictor scale
df = pd.read_csv(out_path / 'gam-plot_link.csv')

outcome = df.outcome.unique()
variables = df['var'].unique()
name_map = {'age_at_request': 'Age at FIT request',
            'imd_quintile': 'IMD quintile',
            'request_year': 'FIT request year',
            'request_month': 'FIT request month'}

xlabel_map = {'age_at_request': 'Age',
              'imd_quintile': 'IMD quintile',
              'request_year': 'Year',
              'request_month': 'Month'}

xtick_map = {'age_at_request': np.arange(20, 110, 10),
            'imd_quintile': [1, 2, 3, 4, 5],
            'request_year': np.arange(2017, 2025, 2),
            'request_month': np.arange(1, 13, 2)}

for o in outcome:
    print(o)

    dfsub = df.loc[df.outcome == o]
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
    ax = ax.flatten()
    for i, v in enumerate(variables):
        dfsubsub = dfsub.loc[dfsub['var'] == v]

        ax[i].plot(dfsubsub.x, dfsubsub.y) 
        ax[i].fill_between(dfsubsub.x, dfsubsub.y_low, dfsubsub.y_upp, alpha=0.25, edgecolor=None)
        ax[i].set(xlabel=xlabel_map[v], ylabel='Contribution to linear predictor',
                  title=name_map[v])
        ax[i].grid(which='major', alpha=0.5, linestyle='--')
        ax[i].set_xticks(xtick_map[v])

    out_name = 'gam_link_' + str(o) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')

# Plot gam effects on response scale (other predictors at median or selected values, see the R script)
df = pd.read_csv(out_path / 'gam-plot_resp.csv')

outcome = df.outcome.unique()
variables = df['var'].unique()

for o in outcome:
    print(o)

    dfsub = df.loc[df.outcome == o]
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
    ax = ax.flatten()
    for i, v in enumerate(variables):
        dfsubsub = dfsub.loc[dfsub['var'] == v]

        ax[i].plot(dfsubsub.x, dfsubsub.y) 
        ax[i].fill_between(dfsubsub.x, dfsubsub.y_low, dfsubsub.y_upp, alpha=0.25, edgecolor=None)
        ax[i].set(xlabel=xlabel_map[v], ylabel='Probability of test nonreturn',
                  title=name_map[v], ylim=(0, 0.5))
        ax[i].grid(which='major', alpha=0.5, linestyle='--')
        ax[i].set_xticks(xtick_map[v])

    out_name = 'gam_resp_' + str(o) + '.png'
    plt.savefig(out_path / out_name, dpi=300, bbox_inches='tight')


#endregion