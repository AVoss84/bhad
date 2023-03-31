from bhad.utils import (Discretize, mvt2mixture)
from bhad.model import BHAD
import numpy as np
import matplotlib.pyplot as plt
#from importlib import reload
from sklearn.datasets import fetch_openml

X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

X.head(2)


#reload(model)

X_cleaned = X.drop(['body', 'cabin', 'name', 'ticket', 'boat'], axis=1).dropna()
y_cleaned = y[X_cleaned.index]

X_cleaned.info(verbose=True)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))


num_cols = list(X_train.select_dtypes(include=['float', 'int']).columns) 

cat_cols = list(X_train.select_dtypes(include=['object', 'category']).columns)

from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ('discrete', Discretize(nbins = None, verbose = False)),     
    ('model', BHAD(contamination = 0.01, num_features = num_cols, cat_features = cat_cols))
])

y_pred_train = pipe.fit_predict(X_train)



from bhad.explainer import Explainer

local_expl = Explainer(pipe.named_steps['model'], pipe.named_steps['discrete']).fit()

local_expl.get_explanation(nof_feat_expl = 5, append = False)   # individual explanations

local_expl.global_feat_imp                                      # global explanation

disc = pipe.named_steps['discrete']
avf = pipe.named_steps['model']

#--------------------------------------------------------------------------------

df_orig = deepcopy(disc.df_orig[avf.df_.columns])   # raw data (no preprocessing/binning) to get the original values of features (not the discretized/binned versions)

expl_thresholds = [.2]*avf.df_.shape[1]
nof_feat_expl = 5
nof_feat_expl = max(nof_feat_expl, 1)     # use at least one feature for explanation   

# Use statistics in f_mat 
# based on data in predict:
#-----------------------------
n = avf.f_mat.shape[0]               # sample size current sample
n_ = avf.f_mat_.shape[0]             # sample size train set; used to convert to rel. freq.
index_row, index_col = np.nonzero(avf.f_mat) 
nz_freq = avf.f_mat[index_row, index_col].reshape(n,-1)    # non-zero frequencies
ac = np.array(avf.df_.columns.tolist())                    # feature names
names = np.tile(ac, (n, 1))
i = np.arange(len(nz_freq))[:, np.newaxis]                      # set new x-axis 
j = np.argsort(nz_freq, axis=1)                              # sort freq. per row and return indices
nz = pd.DataFrame(nz_freq, columns = avf.df_.columns)   # absolute frequencies/counts
df_relfreq = nz/n_                                           # relative marginal frequencies
df_filter = np.zeros(list(df_relfreq.shape), dtype=bool)     # initialize; take only 'significantly' anomalous values
cols = list(df_relfreq.columns)  

df_filter
#-------------------------------------------------------------------------------

from copy import deepcopy
import pandas as pd

df_orig = deepcopy(disc.df_orig_)     # train set
cols = df_orig.columns
cols
feat_info, modes, cdfs = dict(), dict(), dict()

c = 'age'
c = "sibsp"
c = 'parch'
c = "fare"
c = "home.dest"
c = "sex"

cdfs[c] = 'not available'
val_index = avf.enc_.dummy_names_index[c]

x = df_orig[c].tolist()
siglev = 0.1
emp_ci = np.quantile(x, q=[siglev/2, 1 - siglev/2], interpolation="higher")
emp_ci

#-------------------------------------------------------------
counts = pd.DataFrame(avf.freq_[val_index], index=avf.enc_.dummy_names_by_feat[c], columns = ['pmf'])
pmfs = counts/np.sum(counts['pmf'].values)         # rel. freq./estimate pmf
feat_info[c] = pmfs                                # per feature, i.e. column
feat_info[c] = (pmfs.rank(method="max").sort_values(by=['pmf'])/pmfs.shape[0]) 
single = feat_info.get(c).pmf           
modes[c] = single.idxmax(axis=0, skipna=True)      # take argmax to get the x.value of the mode

#---------------------------------------------------
x = pmfs.values.squeeze()
x[x > 0]


siglev = 0.1
emp_ci = np.quantile(x[x > 0], q=[siglev/2, 1 - siglev/2], interpolation="higher")
emp_ci

lower_point, upper_point = emp_ci[0], emp_ci[1] 
lower_point

if lower_point < upper_point:
    np.logical_or(x < lower_point, x > upper_point)


df_orig[c].describe()

#-----------------------------------------------------





modes
feat_info

pmfs
cumsum = (pmfs.rank(method="max").sort_values(by=['pmf'])/pmfs.shape[0])
cumsum

df = pd.DataFrame(data={'pmf': [4, 2, 4, 8, 3, 3]}, index = ['cat', 'penguin', 'dog', 'spider', 'snake', 'dino'])
pmfs = df/df.sum()[0]
pmfs

pmfs.rank(method="max").quantile(0.5, interpolation="lower")

pmfs.rank(method="max").sort_values(by=['pmf']) #.cumsum()

pmfs.rank(method="max").sort_values(by=['pmf']) 

cumsum = (pmfs.rank(method="max").sort_values(by=['pmf'])/pmfs.shape[0])

val = "cat"
cumsum.loc[val,'pmf']


#(pmfs.rank(method="max").sort_values(by=['pmf'])/pmfs.shape[0]).quantile(0.5, interpolation="lower") 


# Compute margins for each feature:
#-------------------------------------
feat_info, modes, cdfs = dict(), dict(), dict()
for c in cols:    
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

