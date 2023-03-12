
from bhad.utils import (discretize, mvt2mixture)
from bhad.model import BHAD
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from sklearn.datasets import fetch_openml


X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True, parser="pandas")


X_cleaned = X.drop(['body', 'cabin', 'name', 'ticket', 'boat'], axis=1).dropna()
y_cleaned = y[X_cleaned.index]

X_cleaned.info(verbose=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

numeric_cols = list(X_train.select_dtypes(include=['float', 'int']).columns) 
cat_cols = list(X_train.select_dtypes(include=['object', 'category']).columns)
print(len(cat_cols+numeric_cols))

from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ('discrete' , discretize(nbins = None, verbose = True)),     
    ('model', BHAD(contamination = 0.01, numeric_features = numeric_cols, cat_features = cat_cols))
])

y_pred_train = pipe.fit_predict(X_train)


from bhad import explainer

reload(explainer)

local_expl = explainer.Explainer(pipe.named_steps['model'], pipe.named_steps['discrete']).fit()

df_train, _ = local_expl.get_explanation()
df_train.shape

X_train.dtypes

df_train #.head(2)


r = iter(range(100))

obs = next(r)

X_train.loc[obs,:]

print(df_train.explanation.loc[obs])



for obs, ex in enumerate(df_train.explanation.values):
    if (obs % 10) == 0:
        print(f'\nObs. {obs}:\n', ex)


#####################################################################
######################################
from copy import deepcopy
import pandas as pd

disc = pipe.named_steps['discrete']
avf = pipe.named_steps['model']
#----------------------------------------------------------------------

df_orig = deepcopy(disc.df_orig[avf.df_.columns])   # raw data (no preprocessing/binning) to get the original values of features (not the discretized/binned versions)

expl_thresholds = [.2]*avf.df_.shape[1]
    
n = avf.f_mat.shape[0]          # sample size current sample
n_ = avf.f_mat_.shape[0]        # sample size train set; used to convert to rel. frequ.
index_row, index_col = np.nonzero(avf.f_mat)
nz_freq = avf.f_mat[index_row, index_col].reshape(n,-1)          # non-zero frequencies

ac = np.array(avf.df_.columns.tolist())         # feature names

names = np.tile(ac, (n, 1))
i = np.arange(len(nz_freq))[:, np.newaxis]          # set new x-axis 
#i = np.arange(len(nz_freq))[:,None]
j = np.argsort(nz_freq, axis=1)                  # sort freq. per row and return indices
nz = pd.DataFrame(nz_freq, columns = avf.df_.columns)   # absolute frequencies/counts

nz_freq
nz_freq[i,j]

X_train.columns


df_relfreq = nz/n_                  # relative marginal frequencies
# Take only 'significantly' anomalous values 
df_filter = np.zeros(list(df_relfreq.shape), dtype=bool)      # initialize
cols = list(df_relfreq.columns)             # all columns

df_relfreq
cols

cols == df_orig.columns
df_orig[cols]

#--------------------------------------------------------------------------
# 'Identify' outliers, with relative freq. below threshold
# (=decision rule)
# Note: smallest (here) 5 features do not necesserily need to have anomalous values
# Once identified we calculate a baseline/reference for the user
# for numeric: use the ECDF; for categorical: mode of the pmf 
# (see calculate_references() fct above)
#--------------------------------------------------------------------------
for z, col in enumerate(cols):
    # handle distr. with few categories
    if not any(df_relfreq[col].values <= expl_thresholds[z]):
        expl_thresholds[z] = min(min(df_relfreq[col].values),.8)    # to exclude minima = 1.0 (-> cannot be outliers!)   
        print(expl_thresholds)

    df_filter[:,z] = df_relfreq[col].values <= expl_thresholds[z]   

df_relfreq
df_filter
print(expl_thresholds)

obs = 0

j[obs,:]

#nz_freq[obs,:]
df_relfreq.loc[obs,:].values

pd.DataFrame(cols).T
df_orig
j

j[obs]
df_orig.values[0,:]
df_orig.values[0,j[obs]]


#-------------------------------------------
df_filter_twist = df_filter[i,j]
df_orig_twist = df_orig.values[i,j]
orig_names_twist = names[i,j]    # sorted; passt!!!!!!!!!!!
orig_names_twist #[obs,:]
#------------------------------------------------

pd.DataFrame(cols).T

#df_filter_twist[obs,:]
df_orig_twist[obs,:]
j[obs,:]
df_orig.loc[obs,:]
df_relfreq.loc[obs,:]


#df_orig_twist
#df_orig.values

names_i = orig_names_twist[obs, :].tolist()
#names_i = orig_names_twist[obs, df_filter_twist[obs,:]].tolist()
names_i

values_i = df_orig_twist[obs, :].tolist()
#values_i = df_orig_twist[obs, df_filter_twist[obs,:]].tolist()
values_i




nof_feat_expl = 5


# Over all observation (rows) in df:
#--------------------------------------
for obs in range(n):
    names_i = orig_names_twist[obs, df_filter_twist[obs,:]].tolist()
    values_i = df_orig_twist[obs, df_filter_twist[obs,:]].tolist()
    assert len(names_i) == len(values_i), 'Lengths of lists names_i and values_i do not match!'
    values_str = list(map(str, values_i))
    
    if len(names_i) > nof_feat_expl:
        names_i = names_i[:nof_feat_expl]
        values_str = values_str[:nof_feat_expl]
    if len(names_i)*len(values_str) > 0 :
        df_orig.loc[obs, 'explanation'] = local_expl._make_explanation_string(names_i, values_i)  
    else:   
        df_orig.loc[obs, 'explanation'] = None   

df_orig.head()

df_orig.explanation.values[0]

j[0]
