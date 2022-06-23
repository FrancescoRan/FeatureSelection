import sys
from os import getcwd
from sys import path
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Union, Tuple
import pandas as pd
import numpy as np
from time import time
import logging
from sklearn.feature_selection import mutual_info_classif, f_classif, f_regression, RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold, train_test_split, permutation_test_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance

from mrmr import mrmr_classif, mrmr_regression # pip install mrmr_selection

path.append(getcwd())


class FeatureSelection():

    def __init__(self):

        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            filename="FS.log", filemode="w", level=logging.DEBUG, format=log_fmt
        )
        self.logger = logging.getLogger(FeatureSelection.__name__)

        logging.basicConfig(level=logging.DEBUG, format=log_fmt)

        # self.logger.info("Initializing FeatureSelection")
        print("Initializing FeatureSelection")


    ## Quick Model:
    def quick_model(self,
                    df: pd.DataFrame,
                    target: Union[str, pd.Series],
                    model,
                    hyperparams: dict,
                    scoring_metric: str,
                    compute_cv_score: bool = False,
                    seed: int = 42):

        assert isinstance(df, pd.DataFrame), "df must be a pd.DataFrame"

        if isinstance(target, str):
            X = df.drop(target, axis=1)
            y = df[target]

        else:
            try:
                X = df.drop(str(target.name), axis=1)
                y = target
            except KeyError:
                X = df
                y = target

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=seed)

        print("Fitting model:", str(model))
        estimator = model(**hyperparams)
        estimator.fit(X_train, y_train)

        y_pred = estimator.predict(X_test)

        print('Classification Report:\n', classification_report(y_true=y_test, y_pred=y_pred))

        if compute_cv_score:
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
            cv_score = cross_val_score(estimator=estimator, X=X_test, y=y_test, scoring=scoring_metric, cv=cv).mean().round(
                4)
            print("CV Score with kfold=3 and n_repeats=2: ", cv_score)

        else:
            cv_score = None

        return estimator, cv_score


    ## Threshold based:
    def remove_missing_by_threshold(self,
                                     df: pd.DataFrame,
                                     threshold: Union[float, int]) -> pd.DataFrame:

        assert isinstance(threshold, float) or isinstance(threshold, int), "threshold must be either float or int"

        if isinstance(threshold, float):
            missing = (df.isna().sum() / df.shape[0]).round(3).reset_index()

        else:
            missing = df.isna().sum().round(3).reset_index()

        cols_with_missing = missing[missing[0] > threshold]['index'].tolist()

        # self.logger.info("Found {l} columns with more than {t} missing values".format(l=len(cols_with_missing), t=threshold))
        print("Found {l} columns with more than {t} missing values".format(l=len(cols_with_missing), t=threshold))

        if cols_with_missing:
            df = df.drop(cols_with_missing, axis=1)
        else:
            pass

        return df

    def remove_low_std_by_threshold(self,
                                     df: pd.DataFrame, threshold: Union[float,int]) -> pd.DataFrame:

        assert isinstance(threshold, float) or isinstance(threshold, int), "threshold must be float or int"

        std_ = df.std().reset_index()

        cols_low_std = std_[std_[0] < threshold]['index'].tolist()

        # self.logger.info("Found {l} columns with less than {t} standard dev".format(l=len(cols_low_std), t=threshold))
        print("Found {l} columns with less than {t} standard dev".format(l=len(cols_low_std), t=threshold))

        if cols_low_std:
            df = df.drop(cols_low_std, axis=1)
        else:
            pass

        return df


    ### Univariate:
    def correlation(self,
                    X: Union[pd.DataFrame, np.array],
                    target_col: Union[str, pd.Series, np.array],
                    corr_type) -> Union[pd.DataFrame, list]:

            assert corr_type in [pearsonr, spearmanr,
                                 kendalltau], "Please select a valid corr_type [pearsonr, spearmanr, kendalltau]"

            corr_type_name = corr_type.__name__

            # self.logger.info("START: Correlation {c}".format(c=corr_type_name))
            print("START: Correlation {c}".format(c=corr_type_name))
            start_time = time()

            X = X.select_dtypes(exclude='object')

            if isinstance(X, pd.DataFrame) and isinstance(target_col, str):
                corrs = X.apply(lambda x: corr_type(x, X[target_col]))
                corrs.index = [corr_type_name, 'pvals']

            elif isinstance(X, pd.DataFrame) and (isinstance(target_col, pd.Series) or isinstance(target_col, np.ndarray)):
                corrs = X.apply(lambda x: corr_type(x, target_col))
                corrs.index = [corr_type_name, 'pvals']

            elif isinstance(X, np.ndarray) and (isinstance(target_col, pd.Series) or isinstance(target_col, np.ndarray)):
                corrs = [corr_type(col, target_col) for col in X]

            try:
                corrs = corrs.T.reset_index().rename(columns={'index': 'features'}).sort_values(corr_type_name,
                                                                                                ascending=False)
            except:
                pass

            end_time = np.round((time() - start_time) / 60, 2)
            # self.logger.info("END: {m} mins".format(m=end_time))
            print("END: {m} mins".format(m=end_time))

            return corrs

    def mutual_information(self,
                           X: Union[pd.DataFrame, np.array],
                           y: pd.Series,
                           n_neighbors: int = 3,
                           seed: int = 42) -> pd.DataFrame:

        """
        https://journals.aps.org/pre/pdf/10.1103/PhysRevE.69.066138

        """

        # self.logger.info("START: mutual_information")
        print("START: mutual_information")

        start_time = time()
        try:
            target_colname = str(y.name)
        except:
            target_colname = str(pd.Series(y).name)
        try:
            X = X.drop(target_colname, axis=1).select_dtypes(exclude='object')
        except:
            pass

        mi = mutual_info_classif(X=X,
                                 y=y,
                                 n_neighbors=n_neighbors,
                                 random_state=seed)

        mi = pd.DataFrame({'features': X.columns, 'mi': mi}).sort_values(by='mi', ascending=False)

        end_time = (time() - start_time) / 60
        # self.logger.info("END: mutual_information in {m} mins".format(m=end_time))
        print("END: mutual_information in {m} mins".format(m=end_time))

        return mi

    def anova_f_stat(self,
                     X: pd.DataFrame,
                     y: pd.Series,
                     type_: str):

        assert type_ in ['classification', 'regression'], "Please insert either calssification or regression for type_"

        # self.logger.info("START: anova_f_stat")
        print("START: anova_f_stat")
        start_time = time()

        if type_ == 'classification':
            f_stat, pvals = f_classif(X=X, y=y)
        else:
            f_stat, pvals = f_regression(X=X, y=y)

        df_anova = pd.DataFrame({'features': X.columns, 'f_stat':f_stat, 'pval':pvals}).sort_values('pval', ascending=True)

        end_time = (time() - start_time) / 60
        # self.logger.info("END: anova_f_stat in {m} mins".format(m=end_time))
        print("END: anova_f_stat in {m} mins".format(m=end_time))

        return df_anova


    ### Recursive:
    def recursive_feature_elimination(self,
                                      X: pd.DataFrame,
                                      y: pd.Series,
                                      model,
                                      hyperparam: dict,
                                      scoring_metric: str,
                                      n_features_per_step: int = 1,
                                      min_features_to_select: int = 1,
                                      verbose: int = 0,
                                      seed: int = 42,
                                      compute_cv_score: bool = False,
                                      importance_getter:str = 'auto'
                                      ) -> Tuple[pd.DataFrame, RFECV]:
        """
        https://link.springer.com/article/10.1023/A:1012487302797

        """
        print("START: recursive_feature_elimination")
        start_time = time()
        
        estimator = model(**hyperparam)
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        rfecv = RFECV(estimator=estimator,
                      step=n_features_per_step,
                      min_features_to_select=min_features_to_select,
                      cv=skf,
                      scoring=scoring_metric,
                      verbose=verbose,
                      n_jobs=-1,
                      importance_getter=importance_getter)
        rfecv.fit(X, y)

        df_rfe = pd.DataFrame({'features': X.columns,
                               'ranking_rfe': rfecv.ranking_}).sort_values('ranking_rfe', ascending=True)

        selected_features = df_rfe['features'].iloc[0:min_features_to_select].tolist()

        print("Performace using top {k} features from RFECV:".format(k=min_features_to_select))
        estimator, cv_score = self.quick_model(df=X[selected_features],
                                               target=y,
                                               model=model,
                                               hyperparams=hyperparam,
                                               scoring_metric=scoring_metric,
                                               compute_cv_score=compute_cv_score,
                                               seed=seed)
        end_time = np.round((time() - start_time) / 60, 2)
        print("END: recursive_feature_elimination in {m} mins".format(m=end_time))

        return df_rfe, rfecv

    def compute_mrmr_and_model(self,
                                X: pd.DataFrame,
                                y: pd.Series,
                                type_: str,
                                k_mrmr: int,
                                hyperparam: dict,
                                relevance_metric: str = 'f',
                                scoring_metric: str = 'roc_auc',
                                model=KNeighborsClassifier,
                                compute_cv_score: bool = False,
                                seed: int = 42) -> dict:

        print("START: compute_mrmr_and_model K={k}".format(k=k_mrmr))
        start_time = time()

        assert type_ in ['classification', 'regression'], "Please insert a valid type_ in ['classification', 'regression']"
        # assert target_col in df.columns, "target_col must be in df"

        mrmr_params = {"X":X,
                     "y":y,
                     "K":k_mrmr,
                     "relevance":relevance_metric,
                     "redundancy":'c',
                     "return_scores":True,
                     "n_jobs":-1}

        if type_ == "classification":
            mrmr_ = mrmr_classif(**mrmr_params)
        else:
            mrmr_ = mrmr_regression(**mrmr_params)

        selected_features = mrmr_[0]

        estimator, cv_score = self.quick_model(df=X,
                                               target=y,
                                               model=model,
                                               hyperparams=hyperparam,
                                               scoring_metric=scoring_metric,
                                               compute_cv_score=compute_cv_score,
                                               seed=seed)

        if compute_cv_score:
            print('CV Score {k} selected features'.format(k=k_mrmr), scoring_metric, cv_score)

        out = {'estimator': estimator, 'mrmr_{k}'.format(k=k_mrmr): mrmr_, '{m}_cv_score_{k}'.format(m=model.__name__, k=k_mrmr): cv_score}

        end_time = np.round((time() - start_time) / 60, 2)
        print("END: compute_mrmr_and_knn {m} mins".format(m=end_time))

        return out


    ### Model Based Feature Selection
    def permutation_feature_importance(self,
                                        X: pd.DataFrame,
                                        y: pd.Series,
                                        model,
                                        hyperparams: dict,
                                        scoring_metric: str,
                                        n_repeats: int = 5,
                                        seed: int = 42) -> pd.DataFrame:
        """
        http://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf

        """

        print("START: permutation_feature_importance, repeating {n} permutations per feature, using {m}".format(n=str(n_repeats), m=model.__name__))
        start_time = time()

        try:
            X = X.drop(y.name, axis=1)
        except KeyError:
            pass

        estimator = model(**hyperparams)
        estimator.fit(X, y)
        # skf       = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

        perm_out = permutation_importance(X=X,
                                          y=y,
                                          estimator=estimator,
                                          scoring=scoring_metric,
                                          n_repeats=n_repeats,
                                          n_jobs=-1,
                                          random_state=seed)

        df_perm_out = pd.DataFrame(
                                    {'features': X.columns,
                                     'impurity_imp': estimator.feature_importances_,
                                     'perm_imp_mean': perm_out['importances_mean'],
                                     'perm_imp_std': perm_out['importances_std']}
                                    ).sort_values('perm_imp_mean', ascending=False)

        end_time = np.round((time() - start_time) / 60, 2)
        print("END: permutation_feature_importance {m} mins".format(m=end_time))

        return df_perm_out


