from collections import defaultdict
from typing import (List, Tuple, Type)
from math import isnan
from copy import deepcopy
import numpy as np
import pandas as pd
#from pandas.api.types import (is_string_dtype, is_numeric_dtype)
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm       
import bhad.utils as utils
from bhad.model import BHAD
from bhad.utils import discretize

class Explainer:

    def __init__(self, bhad_obj : Type['BHAD'], discretize_obj : Type['discretize'], verbose : bool = True):
        """
        Create model explanations per observation 
        Args:
            bhad_obj (sklearn estimator): fitted bhad class instance 
                                         that cointains all relevant attributes
            discretize_obj (sklearn transformer): fitted discretize class instance
            verbose (bool, optional): [description]. Defaults to True.
        """

        # Check if objects are properly fitted sklearn objects:
        #--------------------------------------------------------
        check_is_fitted(bhad_obj) ; check_is_fitted(discretize_obj)
        if verbose : 
            print("--- BHAD Model Explainer ---\n")
            print('Using fitted BHAD and fitted discretizer.')    
        self.verbose = verbose
        self.avf = bhad_obj
        self.disc = discretize_obj
        assert len(self.avf.numeric_features_ + self.avf.cat_features_) > 0, f'\nAt least one numeric or categorical column has to be specified in {self.avf} explicitly!'

    def __del__(self):
        class_name = self.__class__.__name__

    def __repr__(self):
        return f"Explainer(bhad_obj = {self.avf}, discretize_obj = {self.disc})"

    def fit(self)-> 'Explainer':
        self.feature_distr_, self.modes_, self.cdfs_ = self.calculate_references()
        return self

    def calculate_references(self)-> Tuple[dict, dict, dict]:
        """
        Calculate marginal frequency distr. and empirical cdfs per feature, 
        used in model explanation. df_orig must be the train set.
        This will be shown in the explanation as a reference why an obseravtion
        may be anomalous compared to its empirical distribution
        
        Input:
        ------
        df_original : dataframe incl. numeric features with original values before discretization;
                      will be used to estimate c.d.f. of continous feature
        """
        df_orig = deepcopy(self.disc.df_orig_)     # train set
        cols = df_orig.columns

        # Compute margins for each feature:
        #-------------------------------------
        feat_info, modes, cdfs = dict(), dict(), dict()
        for c in cols:    
            #print(f'Column {c} is non-numeric: {is_string_dtype(df_orig[c])}')
            if c in self.avf.numeric_features_:
                cdfs[c] = ECDF(df_orig[c].tolist())   # fit empirical cdf to the non-discretized numeric orig. values
                feat_info[c] = 'not available'
                modes[c] = 'not available'
            elif c in self.avf.cat_features_:    
                cdfs[c] = 'not available'
                val_index = self.avf.enc_.dummy_names_index[c]
                counts = pd.DataFrame(self.avf.freq_[val_index], index=self.avf.enc_.dummy_names_by_feat[c], columns = ['pmf'])
                pmfs = counts/np.sum(counts['pmf'].values)         # rel. freq./estimate pmf
                feat_info[c] = pmfs                                # per feature, i.e. column
                single = feat_info.get(c).pmf           
                modes[c] = single.idxmax(axis=0, skipna=True)      # take argmax to get the x.value of the mode
            else:
                raise ValueError(f'Column {c} missing in provided num./cat. lists! Please check your arguments.')

            if isinstance(modes[c], pd._libs.interval.Interval):
                 modes[c] = round(modes[c].mid,4)
        if self.verbose: 
            print("Marginal cdfs estimated using train set of shape {}".format(df_orig.shape))         
        return feat_info, modes, cdfs
    
    
    def _make_explanation_string(self, names_i : List[str], values_i : List[float])-> List[str]:
        """
        Create local explanation as a string with most relevant features per obseravtion. 
        State their relative position in the respective marginal density/mass function.

        Args:
            names_i (List[str]): _description_
            values_i (List[float]): _description_

        Returns:
            List[str]: Features in the order of rel. importance for obs. i (starting with most important)
        """
        # Convert techy names to human friendly names
        #---------------------------------------------------
        tec2biz = defaultdict(str, {names: names for names in self.disc.df_orig_.columns})     # here both are the same; but can be easily modified    
        names, values = [], []
        for name, val in zip(names_i, values_i):  

                # Numeric features: 
                #-------------------
                if name in self.avf.numeric_features_:
                    # filter out individual numeric values with NaNs from individual explanation
                    if isinstance(val, (int, float)) and ~(isnan(val) | np.isnan(val)):
                        ecdf = self.cdfs_[name]   
                        # Evaluate 1D estimated cdf step function:
                        try: 
                            names.append(tec2biz[name]+' (Cumul.perc.: '+str(round(ecdf(val),3))+')')
                        except Exception as ex:
                            print(ex)
                            names.append(name+' (Cumul.perc.: '+str(round(ecdf(val),3))+')')
                        values.append(str(round(val,2)))
                        
                # Categorical features: 
                #-----------------------
                elif name in self.avf.cat_features_:
                    search_index = np.array(self.feature_distr_[name].index.tolist())
                    comp = str(val) == search_index
                    # If no matching level has been found use 'Others' category and its pr.mass:
                    if not any(comp):
                        comp_aux = (self.avf.enc_.oos_token_ == search_index)
                        row = np.where(comp_aux)[0][0]
                    else:
                        row = np.where(comp)[0][0]
                    pmf = self.feature_distr_[name].iloc[row,:].pmf
                    names.append(tec2biz[name]+' (Perc.: '+str(round(pmf,3))+')')
                    values.append(val)
                else:
                    print(name,"neither numeric nor categorical!")
        return utils.paste(names, values, sep=': ', collapse="\n") 
    
    
    #@utils.timer
    def get_explanation(self, thresholds : float = None, nof_feat_expl : int = 5)-> pd.DataFrame:
        """ 
        Find most infrequent feature realizations based on the BHAD output.
        Motivation: the BHAD anomaly score is simply the unweighted average of the log probabilities
        per feature level/bin (categ. + discretized numerical). Therefore the levels which lead an observation to being
        outlierish are those with (relatively) infrequent counts.
        
        Parameters
        ----------
        nof_feat_expl: max. number of features to report (e.g. the 5 most infrequent features per obs.)
        thresholds:    list of threshold values per feature between [0,1], referring to rel. freq.
        
        Returns:
        --------
        df_original: original dataset + additional column with explanations
        """
        assert hasattr(self, 'feature_distr_'), 'Fit explainer first!'
        df_orig = deepcopy(self.disc.df_orig[self.avf.df_.columns])   # raw data (no preprocessing/binning) to get the original values of features (not the discretized/binned versions)
        if self.verbose : 
            print("Create local explanations for {} observations.".format(df_orig.shape[0])) 
        if thresholds is None:
            self.expl_thresholds = [.2]*self.avf.df_.shape[1]
        else:
            self.expl_thresholds = thresholds

        nof_feat_expl = max(nof_feat_expl, 1)     # use at least one feature for explanation    
        n = self.avf.f_mat.shape[0]               # sample size current sample
        n_ = self.avf.f_mat_.shape[0]             # sample size train set; used to convert to rel. freq.
        index_row, index_col = np.nonzero(self.avf.f_mat) 
        nz_freq = self.avf.f_mat[index_row, index_col].reshape(n,-1)    # non-zero frequencies
        ac = np.array(self.avf.df_.columns.tolist())                    # feature names
        names = np.tile(ac, (n, 1))
        i = np.arange(len(nz_freq))[:, np.newaxis]                      # set new x-axis 
        j = np.argsort(nz_freq, axis=1)                              # sort freq. per row and return indices
        nz = pd.DataFrame(nz_freq, columns = self.avf.df_.columns)   # absolute frequencies/counts
        df_relfreq = nz/n_                                           # relative marginal frequencies
        df_filter = np.zeros(list(df_relfreq.shape), dtype=bool)     # initialize; take only 'significantly' anomalous values
        cols = list(df_relfreq.columns)             # all column names
        #--------------------------------------------------------------------------
        # 'Identify' outliers, with relative freq. below threshold
        # (=decision rule)
        # Note: smallest (here) 5 features do not necesserily need to have anomalous values
        # Once identified we calculate a baseline/reference for the user
        # for numeric: use the ECDF; for categorical: mode of the pmf 
        # (see calculate_references() fct above)
        #--------------------------------------------------------------------------
        for z, col in enumerate(cols):
            # to handle distr. with few categories
            if not any(df_relfreq[col].values <= self.expl_thresholds[z]):
                self.expl_thresholds[z] = min(min(df_relfreq[col].values),.8)    # to exclude minima = 1.0 (-> cannot be outliers!)   
            
            df_filter[:,z] = df_relfreq[col].values <= self.expl_thresholds[z]   

        df_filter_twist = df_filter[i,j]      # sorted filter of 'relevance'
        df_orig_twist = df_orig.values[i,j]  # sorted orig. values
        orig_names_twist = names[i,j]            # sorted names

        # Over all observation (rows) in df:
        #--------------------------------------
        for obs in tqdm(range(n)):
            names_i = orig_names_twist[obs, df_filter_twist[obs,:]].tolist()
            values_i = df_orig_twist[obs, df_filter_twist[obs,:]].tolist()
            assert len(names_i) == len(values_i), 'Lengths of lists names_i and values_i do not match!'
            values_str = list(map(str, values_i))
            
            if len(names_i) > nof_feat_expl:
                names_i = names_i[:nof_feat_expl]
                values_str = values_str[:nof_feat_expl]
            if len(names_i)*len(values_str) > 0 :
               df_orig.loc[obs, 'explanation'] = self._make_explanation_string(names_i, values_i)  
            else:   
               df_orig.loc[obs, 'explanation'] = None   
        return  df_orig