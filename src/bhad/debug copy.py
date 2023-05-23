from bhad import model
import bhad.utils as utils
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload


seed = 42  
outlier_prob_true = .01         # probab. for outlier ; should be consistent with contamination rate in your model
k = 30                          # feature dimension 
N = 2*10**4                     # sample size

# Specify first and second moments for each component  
bvt = utils.mvt2mixture(thetas = {'mean1' : np.full(k,-1), 'mean2' : np.full(k,.5), 
                                'Sigma1' : np.eye(k)*.4, 'Sigma2' : np.eye(k)*.1, 
                                'nu1': 3.*k, 'nu2': 3.*k}, seed = seed, gaussian = False)

# Get latent draws and observations:
#------------------------------------
y_true, dataset = bvt.draw(n_samples = N, k = k, p = outlier_prob_true)

print(dataset.shape)


reload(model)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset, y_true, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))


reload(utils)

from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    # if nbins = None, this will automatically select the optimal bin numbers 
    # based on the MAP estimate (but will make computation slower!)
    ('discrete' , utils.Discretize(nbins = None, verbose = False)),      # step only needed if continous features are present
    ('model', model.BHAD(contamination = 0.01))
])

y_pred_train = pipe.fit_predict(X_train)   
scores_train = pipe.decision_function(X_train) 

y_pred_test = pipe.predict(X_test)
scores_test = pipe.decision_function(X_test)


np.all(y_pred_test == y_pred_test_gpt)
#---------------------------------------------

reload(utils)

disc = utils.Discretize(nbins = None, verbose = False)
X_tilde = disc.fit_transform(X_train)
X_tilde_test = disc.transform(X_test)
#-------------------------------------------------------

reload(utils)

oh = utils.onehot_encoder(prefix_sep='__')
enc = oh.fit(X_tilde)

#df_one = oh.transform(X_tilde).toarray()  

df_one_gpt = oh.transform(X_tilde).toarray()  

np.all(df_one == df_one_gpt)

#----------------------------------------------------------
import pandas as pd
from copy import deepcopy

# Own:
@utils.timer
def transform(self, X: pd.DataFrame):
    
    #check_is_fitted(self)        # Check if fit had been called
    self.selected_col = X.columns[~X.columns.isin(self.exclude_col)]

    # If you already have it from fit then just output it
    if hasattr(self, 'X_') and self.X_.equals(X):
        return self.dummyX_

    df = deepcopy(X[self.selected_col])
    # ohm_aux = df.apply(lambda row: self._single_row(my_tuple = row, df_columns = df.columns), axis=1)
    # ohm = np.stack(ohm_aux.values)
    # return ohm

    ohm = np.zeros((df.shape[0],len(self.columns_)))
    for r, my_tuple in enumerate(df.itertuples(index=False)):    # loop over rows (slow)
        my_index = []
        for z, col in enumerate(df.columns):              # loop over columns
            raw_level_list = list(self.value2name_[col].keys())
            mask = my_tuple[z] == np.array(raw_level_list)
            if any(mask): 
                index = np.where(mask)[0][0]
                dummy_name = self.value2name_[col][raw_level_list[index]]
            else:
                dummy_name = col + self.prefix_sep_ + self.oos_token_
            my_index.append(self.names2index_[dummy_name])
        targets = np.array(my_index).reshape(-1)
        ohm[r,targets] = 1    
    return ohm

#-------------------------------------------------------------

# ChatGPT:
@utils.timer
def transform_gpt(self, X: pd.DataFrame):

    self.selected_col = X.columns[~X.columns.isin(self.exclude_col)]

    # If you already have it from fit then just output it
    if hasattr(self, 'X_') and self.X_.equals(X):
        return self.dummyX_

    df = deepcopy(X[self.selected_col])

    ohm = np.zeros((df.shape[0], len(self.columns_)))

    for col in df.columns:
        raw_level_list = np.array(list(self.value2name_[col].keys()))
        mask = np.isin(df[col].values, raw_level_list)

        masked_values = df[col].values[mask]
        dummy_names = np.array([self.value2name_[col][value] for value in masked_values])
        oos_dummy_name = col + self.prefix_sep_ + self.oos_token_

        my_index = np.array([self.names2index_.get(dummy_name, self.names2index_[oos_dummy_name]) for dummy_name in dummy_names])

        ohm[np.arange(df.shape[0])[mask], my_index] = 1

    return ohm

#----------------------------------------------------------

df_one = transform(self = enc, X = X_tilde)

df_one_gpt = transform_gpt(self = enc, X = X_tilde)

df_one_gpt_test = transform_gpt(self = enc, X = X_tilde_test)

np.all(df_one == df_one_gpt)
#np.equal(df_one, df_one_gpt)

df_one.sum(axis=1)
df_one_gpt.sum(axis=1)

#--------------------------------------------------------------

df_one_gpt.sum(axis=1)

np.all(df_one_gpt.sum(axis=1) > 0)
#-------------------------------------------------------------------

disc = utils.Discretize(nbins = None, verbose = False)

X_tilde = disc.fit_transform(X_train)

oh = utils.onehot_encoder(prefix_sep='__')
enc = oh.fit(X_tilde)

X_tilde_test = disc.transform(X_test)



X = X_tilde_test
self = enc

self.selected_col = X.columns[~X.columns.isin(self.exclude_col)]


df = deepcopy(X[self.selected_col])
df.shape
ohm = np.zeros((df.shape[0], len(self.columns_)))

#for col in df.columns:
cols = iter(df.columns)

col = next(cols)
print(col)

raw_level_list = np.array(list(self.value2name_[col].keys()))
raw_level_list.shape
df[col].values.shape


# for v in df[col].values:
#     print(np.any(v == raw_level_list))

raw_level_list

mask = np.isin(df[col].values, raw_level_list)
mask

sum(mask)
mask.shape

mask

masked_values = df[col].values[mask]
masked_values

masked_values.shape



for z, m in zip(np.arange(len(mask)), mask):
    print(z, m)
    if m:
        self.value2name_[col][df[col].values[z]]
    else:
        oos_dummy_name = col + self.prefix_sep_ + self.oos_token_


dummy_names = np.array([self.value2name_[col][df[col].values[z]] if m else oos_dummy_name for z, m in zip(np.arange(len(mask)), mask)])
dummy_names.shape

#dummy_names = np.array([self.value2name_[col][value] for value in masked_values])
dummy_names

len(dummy_names)
df.shape

oos_dummy_name = col + self.prefix_sep_ + self.oos_token_
oos_dummy_name


my_index = np.array([self.names2index_[dummy_name] for dummy_name in dummy_names])

#my_index = np.array([self.names2index_.get(dummy_name, oos_dummy_name) for dummy_name in dummy_names])
my_index.shape

ohm[np.arange(df.shape[0]), my_index] = 1

ohm.sum(axis=1)
np.all(ohm.sum(axis=1) == 3)