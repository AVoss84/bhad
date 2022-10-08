from sklearn.base import BaseEstimator, OutlierMixin
#from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from copy import deepcopy
import warnings, utils


class BHAD(BaseEstimator, OutlierMixin):
    """
    Bayesian Histogram-based Anomaly Detector (BHAD), see [1] for details. 

    Parameters
    ----------
    contamination : float, optional (default=0.1)
        The amount (fraction) of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function.

    Attributes
    ----------
    threshold : float
        The outlier score threshold calculated based on the score distribution 
        and the provided contamination argument.
    value_counts : dictionary of str keys and pandas.Series values
        A dictionary with column names as the keys and pandas.Series as the values. 
        Each pandas.Series has the value count of each observation in the respective 
        column. The observations (str) are the indices in the pandas.Series object 
        and the counts (int) are the respective values.

    Methods
    -------
    decision_function(self, X[, y])
        Outlier scores of X based on algorithm centered 
        around threshold value. 

    fit(self, X[, y])
        Fit the model. 

    fit_predict(self, X[, y])
        Performs fit on X and returns outlier labels for X. This function is 
        sklearn compatible.

    predict(self, X)
        Predict if a particular sample is an outlier (-1 label) or not (1 label). 

    score_samples(self, X)
        Score of the samples as the outlier score calculated by summing the counts 
        of each feature level in the dataset.
        Although this function is consistent with the sklearn OutlierMixin class, 
        it can not be used with sklearn pipelines (as is the case with other sk-learn 
        outlier detector classes).

    Reference:
    ------------
    [1] Vosseler, A. (2022): Unsupervised insurance fraud prediction based on anomaly detector ensembles, Risks, 10 (132)
    """

    def __init__(self, contamination = 0.01, alpha = 1/2, exclude_col = [], append_score = False, verbose : bool = True):
        
        self.contamination = contamination                   # outlier proportion in the dataset
        self.alpha = alpha                              # uniform Dirichlet prior concentration parameter used for each feature
        self.verbose = verbose
        self.append_score = append_score
        self.exclude_col = exclude_col               # list with column names in X of columns to exclude for computation of the score
        super(BHAD, self).__init__()


    def __del__(self):
        class_name = self.__class__.__name__


    def _fast_bhad(self, X : pd.DataFrame):
      """
      -------
      Input:
      -------
      X:            design matrix as pandas df with all features (must all be categorical, 
                    since one-hot enc. will be applied! Otherwise run discretize() first.)
      append_score: Should anomaly score be appended to X?
      """  
      assert isinstance(X, pd.DataFrame)
      selected_col = X.columns[~X.columns.isin(self.exclude_col)] 
      if len(self.exclude_col)>0:
         print("Features",self.exclude_col, 'excluded.')  
        
      df = deepcopy(X[selected_col]) 
      self.df_shape = df.shape  
      self.columns = df.select_dtypes(include='object').columns.tolist()  # use only categorical (including discretized numerical)
    
      if len(self.columns)!= self.df_shape[1] :
          warnings.warn('Not all features in X are categorical!!')
      self.df = df
      #self.enc = OneHotEncoder(handle_unknown='ignore', dtype = int)
      self.enc = utils.onehot_encoder(prefix_sep='__')
      self.enc.fit(df)    # training phase
             
      self.df_one = self.enc.transform(df)   # apply one-hot encoder to categorical -> sparse dummy matrix
      assert all(np.sum(self.df_one, axis=1) == df.shape[1]), 'Row sums must be equal to number of features!!'
      self.columns_onehot_ = self.enc.get_feature_names()   # keep this for postprocessing/explainability later
      if self.verbose : print("Matrix dimension after one-hot encoding:", self.df_one.shape)  
     
      self.alphas = np.array([self.alpha]*self.df_one.shape[1])        # Dirichlet concentration parameters; aka pseudo counts
      self.freq = self.df_one.sum(axis=0)                                 # suff. statistics of multinomial likelihood
      self.log_pred = np.log((self.alphas + self.freq)/np.sum(self.alphas + self.freq))  # log posterior predictive probabilities for single trial / multinoulli

      # Duplicate list of marg. freq. in an array for elementwise multiplication  
      # i.e. Repeat counts for each obs. i =1...n
      #---------------------------------------------------------------------------   
      # Keep frequencies for explanation later         
      a = np.tile(self.freq, (self.df_shape[0], 1))     # Repeat counts for each obs. i =1...n
      a_bayes = np.tile(self.log_pred, (self.df_shape[0], 1))

      # Keep only nonzero matrix entries 
      # (via one-hot encoding matrix), i.e. freq. for respective entries
      # Assign each obs. the overall category count
      #-------------------------------------------------------------------
      self.f_mat = self.df_one * np.array(a)                    # keep only nonzero matrix entries
      f_mat_bayes = self.df_one * np.array(a_bayes)

      # Calculate outlier score for each row (observation), see equation (5) in paper.
      out = pd.Series(np.apply_along_axis(np.sum, 1, f_mat_bayes), index=df.index)    
      if self.append_score:  
         out = pd.concat([df, pd.DataFrame(out, columns = ['outlier_score'])], axis=1)
      return out    
    

    def fit(self, X, y=None):
        """
        Apply the BHAD and calculate the outlier threshold value.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples. X values should be of type str, or castable to 
            str (e.g. catagorical).
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : BHAD object
        """
        if self.verbose : print("\nConstruct Bayesian Histogram-based Anomaly Detector (BHAD).")
        self.scores = self._fast_bhad(X)
    
        if self.append_score:  
            self.threshold = np.nanpercentile(self.scores['outlier_score'].tolist(), q=100*self.contamination)
        else: 
            self.threshold = np.nanpercentile(self.scores.tolist(), q=100*self.contamination)
        if self.verbose : print("BHAD completed.")

        self.scores_ = self.scores          
        self.threshold_ = self.threshold
        self.freq_ = self.freq
        self.f_mat_ = self.f_mat
        self.df_one_ = self.df_one 
        self.X_ = X
        self.df_ = self.df
        self.xindex_fitted_ = X.index
        self.enc_ = self.enc
        return self

    
    def score_samples(self, X):
        """
        Outlier score calculated by summing the counts 
        of each feature level in the dataset.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples. X values should be of type str, or easily castable 
            to str (e.g. categorical).

        Returns
        -------
        scores : numpy.array, shape (n_samples,)
            The outlier score of the input samples centered arount threshold 
            value.
        """
        df = deepcopy(X)
        self.df_one = self.enc_.transform(df)#.toarray()   # apply fitted one-hot encoder to categorical -> sparse dummy matrix
        assert all(np.sum(self.df_one, axis=1) == df.shape[1]), 'Row sums must be equal to number of features!!'

        self.freq_updated_ = self.freq_ + self.df_one.sum(axis=0)      # update suff. stat with abs. freq. of new data levels
        #freq_updated = np.log(np.exp(self.freq) + self.df_one + alpha)    # multinomial-dirichlet
        self.log_pred = np.log((self.alphas + self.freq_updated_)/np.sum(self.alphas + self.freq_updated_))   # log posterior predictive probabilities for single trial / multinoulli
        self.columns_onehot = self.enc_.columns_#.get_feature_names(df_str_cols)
        self.f_mat = self.freq_updated_ * self.df_one           # get level specific counts for X, e.g. test set
        f_mat_bayes = self.log_pred * self.df_one  
        self.scores = pd.Series(np.apply_along_axis(np.sum, 1, f_mat_bayes), index=X.index) 

        # If you already have it from fit then just output it:
        if hasattr(self, 'X_') and (len(self.xindex_fitted_) == X.shape[0]):
            self.f_mat = deepcopy(self.f_mat_)
            return self.scores_
        else:    
            return self.scores
    
    
    def decision_function(self, X):
        """
        Outlier score centered around the threshold value. Outliers are scored 
        negatively (<= 0) and inliers are scored positively (> 0).

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples. X values should be of type str, or easily castable 
            to str (e.g. categorical).

        Returns
        -------
        scores : numpy.array, shape (n_samples,)
            The outlier score of the input samples centered arount threshold 
            value.
        """
        score = self.score_samples(X).to_numpy()
        # center scores; divide into outlier and inlier (-/+)
        return score - self.threshold_
    
    
    def predict(self, X):
        """
        Returns labels for X.

        Returns -1 for outliers and 1 for inliers.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            The input samples. X values should be of type str, or easily castable 
            to str (e.g. categorical).

        Returns
        -------
        scores : array, shape (n_samples,)
            The outlier labels of the input samples.
            -1 means an outlier, 1 means an inlier.
        """
        self.anomaly_scores = self.decision_function(X)            # get centered anomaly scores
        outliers = np.asarray(-1*(self.anomaly_scores <= 0).astype(int))
        inliers = np.asarray((self.anomaly_scores > 0).astype(int))
        return outliers + inliers

