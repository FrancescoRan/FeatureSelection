import pandas as pd
from sklearn.datasets import make_classification
from os import getcwd
from sys import path
from scipy.stats import pearsonr, spearmanr, kendalltau
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

path.append(getcwd())

from utils.feature_selection_utils import FeatureSelection
from utils.MultipleTests import MultipleTests


seed    = 42
X, y    = make_classification(n_samples=10000, n_features=100, n_informative=10, n_redundant=40, n_repeated=5, n_classes=2, random_state=seed)
df      = pd.DataFrame(X, columns=['x{i}'.format(i=i) for i in range(0, X.shape[1], 1)])
df['y'] = y

fs = FeatureSelection()

### Threshold based:
df_no_miss    = fs.remove_missing_by_threshold(df=df.drop('y', axis=1), threshold=0.1)
df_no_low_std = fs.remove_low_std_by_threshold(df=df.drop('y', axis=1), threshold=2)


# Univariate

### corr:
corr_threshold = 0.25
corr_type      = pearsonr # spearmanr # pearsonr
df_corr        = fs.correlation(X=df.drop('y', axis=1), target_col=y, corr_type=corr_type)
corr_feat      = df_corr[df_corr[corr_type.__name__].abs() >= corr_threshold]['features'].tolist()
len(corr_feat)

sns.heatmap(df[corr_feat].corr().abs())
df[corr_feat].corr().abs().mean(axis=1).mean()


### MI:
# https://journals.aps.org/pre/pdf/10.1103/PhysRevE.69.066138

mi_threshold = 0.4
df_mi        = fs.mutual_information(X=df, y=df['y'], n_neighbors=3, seed=seed)
df_mi['mi'].hist(bins=50)
features_mi = df_mi['features'].iloc[0:20].tolist()
df[features_mi].corr().abs().mean(axis=1).mean()


# comparison corr vs mi
df_compar = df_corr.merge(df_mi, how='inner', on='features')
df_compar[['pearsonr', 'mi']].abs().corr()
# df_compar[['spearmanr', 'mi']].abs().corr()


### F-stat:
df_f = fs.anova_f_stat(X=df.drop('y', axis=1), y=df['y'], type_='classification')
features_f = df_f['features'].iloc[0:20].tolist()
df[features_f].corr().abs().mean(axis=1).mean()


### Multiple tests
# Mann-Whitney

mt = MultipleTests(df=df,
                 target='y',
                 alpha=0.05,
                 alternative= 'two-sided',
                 test = mannwhitneyu,
                 multiple_test_correction=True)

df_mt = mt.run()

features_mt = df_mt['vars'].iloc[0:20].tolist()
df[features_mt].corr().abs().mean(axis=1).mean()


### recursive cv
# https://link.springer.com/article/10.1023/A:1012487302797

# model          = LogisticRegression(penalty='none', fit_intercept=True, n_jobs=-1)
rf_hyperparams = {'n_estimators': 100, 'max_depth':8, 'max_features':0.8}
model          = RandomForestClassifier

# model = SVC(C=1.0, kernel='linear') # no reg e kernel lineare https://link.springer.com/article/10.1023/B:STCO.0000035301.49549.88
top_k_feat     = 20
df_rfe, rfecv  = fs.recursive_feature_elimination(X=df.drop('y', axis=1),
                                          y=df['y'],
                                          model=model,
                                          hyperparam=rf_hyperparams,
                                          scoring_metric='roc_auc',
                                          n_features_per_step=20,
                                          min_features_to_select=top_k_feat,
                                          verbose=1,
                                          seed=seed,
                                          importance_getter= 'auto' # "coef_"
                                          )

features_rfe = df_rfe['features'].iloc[0:20].tolist()
df[features_rfe].corr().abs().mean(axis=1).mean()



### MRMR + Model
# https://arxiv.org/pdf/1908.05376.pdf

rf_hyperparams = {'n_estimators': 100, 'max_depth':8, 'max_features':0.8}
k_mrmr         = 20
dict_mrmr      = fs.compute_mrmr_and_model(X=df.drop('y', axis=1),
                                            y=df['y'],
                                            type_="classification",
                                            k_mrmr=20,
                                            hyperparam=rf_hyperparams,
                                            relevance_metric='ks', # 'f', 'rf'
                                            scoring_metric='roc_auc',
                                            model=RandomForestClassifier,
                                            # seed=seed
                                           )

features_mrmr = dict_mrmr['mrmr_'+str(k_mrmr)][0]
df[features_mrmr].corr().abs().mean(axis=1).mean()


### Model Based Feature Selection

rf_hyperparams = {'n_estimators': 100, 'max_depth':8, 'max_features':0.8}
perm_imp       = fs.permutation_feature_importance(
                                        X=df.drop('y', axis=1),
                                        y=df['y'],
                                        model= RandomForestClassifier,
                                        hyperparams= rf_hyperparams,
                                        scoring_metric='roc_auc',
                                        n_repeats=10,
                                        seed=seed)

features_perm = perm_imp['features'].iloc[0:20].tolist()
df[features_perm].corr().abs().mean(axis=1).mean()

list(set(features_rfe) & set(features_mrmr))
list(set(features_mrmr) & set(features_perm))
list(set(features_rfe) & set(features_perm))

features_sel_dict = {'corr': corr_feat,
                     'mi': features_mi,
                     'f_stat': features_f,
                     'mt_mw': features_mt,
                     'rfe': features_rfe,
                     'mrmr': features_mrmr,
                     'perm': features_perm}


df_res  = pd.DataFrame(columns=['features'])

for k, v in features_sel_dict.items():

    df_l    = pd.DataFrame(v).rename(columns={0:'features'})
    df_l[k] = 1
    df_res  = df_res.merge(df_l, how='outer', on="features")

df_res = df_res.fillna(0)
sns.heatmap(df_res.select_dtypes(exclude='object'), yticklabels=df_res['features'])








### for DNA MEthylation:
# https://pubmed.ncbi.nlm.nih.gov/22524302/
#