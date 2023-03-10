import os, sys
import numpy as np
import pandas as pd
from math import floor, ceil, isnan
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.utils.validation import check_is_fitted
from copy import deepcopy
from tqdm.auto import tqdm       # progress bar
from bhad.utils


class explainer:

    def __init__(self, avf_obj, discretize_obj, verbose : bool = True):
        """
        Create model explanations per observation/claim using the BAVF
        algorithm as global approximation for the ensemble model.
        Args:
            avf_obj (sklearn estimator): fitted avf class instance 
                                         that cointains al relevant attributes
            discretize_obj (sklearn transformer): fitted discretize class instance
            verbose (bool, optional): [description]. Defaults to True.
        """

        # Check if objects are properly fitted sklearn objects:
        #--------------------------------------------------------
        check_is_fitted(avf_obj) ; check_is_fitted(discretize_obj)
        if verbose : print('Using fitted BAVF and Discretizer instances.')    
        self.verbose = verbose
        self.avf = avf_obj
        self.disc = discretize_obj

    def __del__(self):
        class_name = self.__class__.__name__

    def fit(self):
        self.feature_distr_, self.modes_, self.cdfs_ = self.calc_categorical_margins()
        return self

    def calc_categorical_margins(self):
        """
        Calculate marginal frequency distr. and empirical cdfs per feature, 
        used in model explanation. df_orig must be the train set.
        Input:
        df_original : dataframe inlc. numeric features with original values before discretization;
                      will be used to estimate c.d.f. of continous feature
        """
        df_orig = deepcopy(self.disc.df_orig_)
        cols = df_orig.columns

        # Compute margins for each feature:
        #-------------------------------------
        feat_info, modes, cdfs = dict(), dict(), dict()
        for c in cols:    
            cdfs[c] = ECDF(df_orig[c].tolist())
            val_index = self.avf.enc_.dummy_names_index[c]
            counts = pd.DataFrame(self.avf.freq_[val_index], index=self.avf.enc_.dummy_names_by_feat[c], columns = ['pmf'])
            pmfs = counts/np.sum(counts['pmf'].values)         # rel. freq.
            feat_info[c] = pmfs
            single = feat_info.get(c).pmf           
            modes[c] = single.idxmax(axis=0, skipna=True)      # take argmax to get the x.value of the mode

            if isinstance(modes[c], pd._libs.interval.Interval):
                 modes[c] = round(modes[c].mid,2)
        if self.verbose : print("Marginal cdfs estimated using train set of shape {}".format(df_orig.shape))         
        return feat_info, modes, cdfs
    
    
    def _make_explanation_string(self, names_i, values_i):
        
        # Convert techy names to business friendly names in Filtered-MFT xls:
        #----------------------------------------------------------------------
        config_post = {} #config_meta.post['post_processing']['template']   # get postprocessor.yaml template
        feat_names_tech = list(config_post.keys())               # techy names
        tec2biz = {na : config_post.get(na,{})[na] for na in feat_names_tech}   # create techy-to-business names mapping dict 
        
        names, values = [], []
        for name, val in zip(names_i, values_i):   
            if ~(isnan(val) | np.isnan(val)):    # filter out individual values with NaNs from individual explanation
                if name in self.avf.numeric_features_:
                    ecdf = self.cdfs_[name]   
                    # Evaluate 1D estimated cdf step function:
                    try: 
                        names.append(tec2biz[name]+' (Cumul.perc.: '+str(round(ecdf(val),2))+')')
                    except Exception as ex:
                        print(ex)
                        names.append(name+' (Cumul.perc.: '+str(round(ecdf(val),2))+')')
                        
                    values.append(str(round(val,2)))
                elif name in self.avf.cat_features_:
                    search_index = np.array(self.feature_distr_[name].index.tolist())
                    comp = str(val) == search_index
                    # If no matching level has been found use 'Others' category and its pr.mass:
                    if ~any(comp):
                        comp_aux = (self.avf.enc_.oos_token_ == search_index)
                        row = np.where(comp_aux)[0][0]
                    else:
                        row = np.where(comp)[0][0]
                    pmf = self.feature_distr_[name].iloc[row,:].pmf
                    names.append(tec2biz[name]+' (Perc.: '+str(round(pmf,2))+')')
                    values.append(val)
                else:
                    print(name,"neither numeric nor categorical!")
        return utils.paste(names, values, sep=': ', collapse="\n") 
    
    
    @utils.timer
    def explain_avf(self, thresholds = None, nof_feat_expl = 5):
        """ 
        Find most infrequent feature realizations based on the AVF output.
        Motivation: the AVF anomaly score is simply the unweighted average of the absolute frequencies
        per feature level (categ. + discretized numerical). Therefore the levels which lead an observation to being
        outlierish are those with (relatively) infrequent counts. 
        
        Parameters
        ----------
        nof_feat_expl: max. number of features to report (e.g. the 5 most infrequent features per obs.)
        thresholds:    list of threshold values per feature between [0,1], referring to rel. freq.
        
        Returns:
        --------
        df_original + string column vector with feature realisations 
        """
        df_orig = deepcopy(self.disc.df_orig)   # raw data (no preprocessing/binning) to get the original values of features (not the discretized/binned versions)
        if self.verbose : print("Make explanation for {} observations.".format(df_orig.shape[0])) 
        if thresholds is None:
            self.expl_thresholds = [.2]*self.avf.df_.shape[1]
        else:
            self.expl_thresholds = thresholds
            
        n = self.avf.f_mat.shape[0]          # sample size current sample
        n_ = self.avf.f_mat_.shape[0]        # sample size train set; used to convert to rel. frequ.
        index_row, index_col = np.nonzero(self.avf.f_mat) 
        nz_freq = self.avf.f_mat[index_row, index_col].reshape(n,-1)          # non-zero frequencies
        ac = np.array(self.avf.df_.columns.tolist())         # feature names
        names = np.tile(ac, (n, 1))
        i = np.arange(len(nz_freq))[:, np.newaxis]          # set new x-axis 
        j = np.argsort(nz_freq, axis=1)                  # sort freq. per row and return indices
        nz = pd.DataFrame(nz_freq, columns = self.avf.df_.columns)   # absolute frequencies/counts
        df_relfreq = nz/n_                  # relative marginal frequencies
        df_filter = np.zeros(list(df_relfreq.shape), dtype=bool)      # initialize
        cols = df_relfreq.columns             # all columns
        
        # Identify outliers, with relative frequ. below threshold
        #----------------------------------------------------------
        for z, col in enumerate(cols):
            if not any(df_relfreq[col].values <= self.expl_thresholds[z]):
                self.expl_thresholds[z] = min(min(df_relfreq[col].values),.8)    # to exclude minima = 1.0 (-> cannot be outliers!)     

            # This is a quick & dirty fix in case you have included LoB as a control variable
            # do not use it in the model explanation 
            if '_lob' in col: self.expl_thresholds[z] = -1000    # set its threshold to an impossible value vs. (0,1), hence will never be picked up

            df_filter[:,z] = df_relfreq[col].values <= self.expl_thresholds[z]   

        df_filter_twist = df_filter[i,j]
        df_orig_twist = df_orig.values[i,j]
        orig_names = names[i,j]

        # Over all observation (rows) in df:
        #--------------------------------------
        for obs in tqdm(range(n)):
            names_i = orig_names[obs, df_filter_twist[obs,:]].tolist()
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

        return  df_orig, self.expl_thresholds