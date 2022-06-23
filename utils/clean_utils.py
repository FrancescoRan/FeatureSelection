import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class CleanUtils:
    
    def __init__(self) -> None:
        pass

    
    def _check_len_nans(df : pd.DataFrame, 
                        axis : int, 
                        threshold : float = 0.3, 
                        plot_hist : bool = False,
                        return_df : bool = False) -> pd.DataFrame:
    
        assert threshold >= 0 and threshold <= 1, "threshold must be in [0,1]"
        
        df_na        = df.isna().sum(axis)/df.shape[0]
        list_na_vars = list(df_na[df_na >= threshold].index)
        
        if plot_hist:
            plt.hist(df_na, log = True, bins = 60)
            plt.show()
        
        print(df_na.describe())
        
        df_na_cat = np.select(condlist = [
                                df_na == 0.0,   
                                df_na <= 0.0001,
                                df_na <= 0.005,
                                df_na <= 0.01,
                                df_na <= 0.05,
                                df_na <= 0.10,
                                df_na <= 0.2,
                                df_na > 0.2], 
                            choicelist = ['No Missing',
                                            '<= 0.01%',
                                            '<= 0.5%',
                                            '<= 1%',
                                            '<= 5%',
                                            '<= 10%',           
                                            '<= 20%',
                                            '> 20%']
                            )
        
        print(pd.Series(df_na_cat).value_counts())
        
        if return_df:
            return df_na[df_na >= threshold]
    

    def _impute_mean(df: pd.DataFrame) -> pd.DataFrame:
        ## su betas la media performa praticamente uguale alla soluzione migliore:
        ## https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03592-5
        
        df_na            = df.isna().sum()
        list_na_vars     = list(df_na[df_na > 0].index) 
        df[list_na_vars] = df[list_na_vars].apply(lambda x: x.fillna(x.mean(skipna = True)), axis = 0)
        
        return df