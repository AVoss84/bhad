from bhad import model
import bhad.utils as utils
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

seed = 42  
outlier_prob_true = .01         # probab. for outlier ; should be consistent with contamination rate in your model
k = 20                          # feature dimension 
N = 2*10**3                     # sample size

# Specify first and second moments for each component  
bvt = utils.mvt2mixture(thetas = {'mean1' : np.full(k,-1), 'mean2' : np.full(k,.5), 
                                'Sigma1' : np.eye(k)*.4, 'Sigma2' : np.eye(k)*.1, 
                                'nu1': 3.*k, 'nu2': 3.*k}, seed = seed, gaussian = False)

# Get latent draws and observations:
#------------------------------------
y_true, dataset = bvt.draw(n_samples = N, k = k, p = outlier_prob_true)

print(dataset.shape)


from bhad.utils import (Discretize, mvt2mixture)
from bhad.model import BHAD
import numpy as np
import matplotlib.pyplot as plt
#from importlib import reload
from sklearn.datasets import fetch_openml

X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True, parser="pandas")

X.head(2)

dataset = X.drop(['body', 'cabin', 'name', 'ticket', 'boat'], axis=1).dropna()
y_true = y[dataset.index]


#reload(bhad)
#-------------------------------------
# disc = util.discretize(nbins = None, verbose = False)

# df = disc.fit_transform(dataset)

# print(dataset.shape)
# print(df.shape)

# pipe = bhad.BHAD(contamination = 0.01)
# pipe.fit(df)   

# #yhat = pipe.fit_predict(dataset)   
# scores = pipe.decision_function(df)
# scores

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset, y_true, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))





from bhad.utils import (Discretize, mvt2mixture)
from bhad.model import BHAD
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from sklearn.pipeline import Pipeline
from copy import deepcopy

numeric_cols = list(X_train.select_dtypes(include=['float', 'int']).columns) 
cat_cols = list(X_train.select_dtypes(include=['object', 'category']).columns)

print(len(cat_cols+numeric_cols))

pipe = Pipeline(steps=[
    ('discrete', Discretize(nbins = None, verbose = False)),     
    ('model', BHAD(contamination = 0.01, numeric_features = numeric_cols, cat_features = cat_cols))
])

y_pred_train = pipe.fit_predict(X_train)   
scores_train = pipe.decision_function(X_train) 

y_pred_test = pipe.predict(X_test)
scores_test = pipe.decision_function(X_test)

#---------------------------------------------
# disc = utils.Discretize(nbins = None, verbose = False)
# X_tilde = disc.fit_transform(X_train)

# bm = model.BHAD(contamination = 0.01)

# model.fit(X_tilde)   

# y_pred_train = model.fit_predict(X_tilde)   
# scores_train = model.decision_function(X_tilde) 
# scores_train

# X_tilde_test = disc.transform(X_test)
# y_pred_test = model.predict(X_tilde_test)   
# scores_test = model.decision_function(X_tilde_test) 
# scores_test

#-------------------------------------

from bhad import explainer
import pandas as pd

reload(explainer)

local_expl = explainer.Explainer(pipe.named_steps['model'], pipe.named_steps['discrete']).fit()

df_train = local_expl.get_explanation()
#------------------------------------------------------------------------------------------------------

avf = pipe.named_steps['model']
disc = pipe.named_steps['discrete']

df_orig = deepcopy(disc.df_orig[avf.df_.columns])   # raw data (no preprocessing/binning) to get the original values of features (not the discretized/binned versions)
expl_thresholds = [.2]*avf.df_.shape[1]
nof_feat_expl = 5

nof_feat_expl = max(nof_feat_expl, 1)     # use at least one feature for explanation    
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
cols = list(df_relfreq.columns)             # all column names

ranks = np.argsort(j, axis=1)                                # array with ranks for each observ./cell 
avg_ranks = np.mean(ranks, axis=0)                           # avg. rank per feature 
index_sorted_ranks = np.argsort(avg_ranks)

global_feat_imp = list(np.array(cols)[index_sorted_ranks])

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
    if not any(df_relfreq[col].values <= expl_thresholds[z]):
        expl_thresholds[z] = min(min(df_relfreq[col].values),.8)    # to exclude minima = 1.0 (-> cannot be outliers!)   
    
    df_filter[:,z] = df_relfreq[col].values <= expl_thresholds[z]   

df_filter_twist = df_filter[i,j]      # sorted filter of 'relevance'
df_orig_twist = df_orig.values[i,j]  # sorted orig. values
orig_names_twist = names[i,j]            # sorted names

df_filter_twist
df_orig_twist
orig_names_twist
j

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

obs
names_i
values_i


tester = nz_freq #[:10,:6]
len(tester)
tester.shape


i = np.arange(len(tester))[:, np.newaxis]                      # set new x-axis 
j = np.argsort(tester, axis=1)                              # sort freq. per row and return indices

ranks = np.argsort(j, axis=1) 

tester
j
ranks

avg_ranks = np.mean(ranks, axis=0)
avg_ranks

ranks_global_feat_imp = pd.DataFrame(avg_ranks, index=cols, columns=['avg ranks']).sort_values(by=['avg ranks'], ascending=True)
ranks_global_feat_imp

# index_sorted_ranks = np.argsort(avg_ranks)
# index_sorted_ranks


# global_feat_imp = list(np.array(cols)[index_sorted_ranks])
# global_feat_imp

#list(ranks_global_feat_imp.index)