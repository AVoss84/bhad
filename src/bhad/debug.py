from bhad import model
import bhad.utils as utils
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload


seed = 42  
outlier_prob_true = .01         # probab. for outlier ; should be consistent with contamination rate in your model
k = 30                          # feature dimension 
N = 2*10**3                     # sample size

# Specify first and second moments for each component  
bvt = utils.mvt2mixture(thetas = {'mean1' : np.full(k,-1), 'mean2' : np.full(k,.5), 
                                'Sigma1' : np.eye(k)*.4, 'Sigma2' : np.eye(k)*.1, 
                                'nu1': 3.*k, 'nu2': 3.*k}, seed = seed, gaussian = False)

# Get latent draws and observations:
#------------------------------------
y_true, dataset = bvt.draw(n_samples = N, k = k, p = outlier_prob_true)

print(dataset.shape)


reload(model)

# from sklearn.pipeline import Pipeline

# pipe = Pipeline(steps=[
#     # if nbins = None, this will automatically select the optimal bin numbers 
#     # based on the MAP estimate (but will make computation slower!)
#     ('discrete' , util.discretize(nbins = None, verbose = False)),      # step only needed if continous features are present
#     ('model', bhad.BHAD(contamination = 0.01))
# ])

# yhat = pipe.fit_predict(dataset)
# #pipe.score_samples(dataset)   
# scores = pipe.decision_function(dataset)
# scores

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
#---------------------------------------------


disc = utils.Discretize(nbins = None, verbose = False)
X_tilde = disc.fit_transform(X_train)

reload(utils)

oh = utils.onehot_encoder()
enc = oh.fit(X_tilde)

ohm = oh.transform(X_tilde)
ohm

X = X_tilde

selected_col = X.columns[~X.columns.isin(enc.exclude_col)]

df = X[enc.selected_col]

df.head()

ohm = np.zeros((df.shape[0],len(enc.columns_)))

loop1 = enumerate(df.itertuples(index=False))

r, my_tuple = next(loop1)
r, my_tuple
type(my_tuple)

loop2 = enumerate(df.columns)

z, col = next(loop2)
z, col

raw_level_list = list(enc.value2name_[col].keys())
raw_level_list

mask = my_tuple[z] == np.array(raw_level_list)
mask


def single_row(self, x, df_columns):
    """Run over all columns for single row"""
    my_index = []
    ohm = np.zeros((len(enc.columns_)), dtype=np.int8)
    for z, col in enumerate(df_columns): 

        raw_level_list = list(self.value2name_[col].keys())
        mask = x[z] == np.array(raw_level_list)
        if any(mask): 
            index = np.where(mask)[0][0]
            dummy_name = self.value2name_[col][raw_level_list[index]]
        else:
            dummy_name = col + self.prefix_sep_ + self.oos_token_
        my_index.append(self.names2index_[dummy_name])

    targets = np.array(my_index).reshape(-1)
    ohm[targets] = 1
    return ohm


## loop over rows
ohm_aux = df.apply(lambda row: test(x = row, df_columns = df.columns, self = enc), axis=1)
ohm = np.stack(ohm_aux.values)
ohm


##
ohm = np.zeros((df.shape[0],len(enc.columns_)))
targets = []
for r, my_tuple in enumerate(df.itertuples(index=False)): 
    print(r, test(x = my_tuple, df_columns = df.columns, self = enc))
    tar = test(x = my_tuple, df_columns = df.columns, self = enc)
    targets.append(tar)
    ohm[r,tar] = 1
ohm
ohm.sum(axis=1)






for r, my_tuple in enumerate(df.itertuples(index=False)): 
    print(my_tuple)


for r, my_tuple in enumerate(df.itertuples(index=False)):    # loop over rows (slow)
    my_index = []
    for z, col in enumerate(df.columns):              # loop over columns
        raw_level_list = list(enc.value2name_[col].keys())
        mask = my_tuple[z] == np.array(raw_level_list)
        if any(mask): 
            index = np.where(mask)[0][0]
            dummy_name = enc.value2name_[col][raw_level_list[index]]
        else:
            dummy_name = col + enc.prefix_sep_ + enc.oos_token_
        my_index.append(enc.names2index_[dummy_name])
    targets = np.array(my_index).reshape(-1)
    ohm[r,targets] = 1    


def helper(enc):

    my_index = []
    for z, col in enumerate(df.columns):              # loop over columns
        raw_level_list = list(enc.value2name_[col].keys())
        mask = my_tuple[z] == np.array(raw_level_list)
        if any(mask): 
            index = np.where(mask)[0][0]
            dummy_name = enc.value2name_[col][raw_level_list[index]]
        else:
            dummy_name = col + enc.prefix_sep_ + enc.oos_token_
        my_index.append(enc.names2index_[dummy_name])



""" model = bhad_old.BHAD(contamination = 0.01)

y_pred_train = model.fit_predict(X_tilde)   
scores_train = model.decision_function(X_tilde) 
scores_train

X_tilde_test = disc.transform(X_test)
y_pred_test = model.predict(X_tilde_test)   
scores_test = model.decision_function(X_tilde_test) 
scores_test

#---------------------------------------------
reload(bhad)
#-------------------------------------

pipe = bhad.BHAD(contamination = 0.01, verbose=False)

y_pred_train = pipe.fit_predict(X_train)   
scores_train = pipe.decision_function(X_train)

y_pred_test = pipe.predict(X_test)
scores_test = pipe.decision_function(X_test)
scores_test
#------------------------------------------------

import pandas as pd
from copy import deepcopy 

X_tilde = pipe.disc.fit_transform(X_test)

df = X_tilde
df
X = deepcopy(X_test)        

df_one = pipe.enc_.transform(df).toarray()   # apply fitted one-hot encoder to categorical -> sparse dummy matrix
assert all(np.sum(pipe.df_one, axis=1) == df.shape[1]), 'Row sums must be equal to number of features!!'

df_one

# Update suff. stat with abs. freq. of new data points/levels
freq_updated_ = pipe.freq_ + pipe.df_one.sum(axis=0)      
#freq_updated = np.log(np.exp(self.freq) + self.df_one + alpha)    # multinomial-dirichlet

# Log posterior predictive probabilities for single trial / multinoulli
log_pred = np.log((pipe.alphas + freq_updated_)/np.sum(pipe.alphas + pipe.freq_updated_))   
f_mat = freq_updated_ * df_one           # get level specific counts for X, e.g. test set
f_mat_bayes = log_pred * df_one  
scores = pd.Series(np.apply_along_axis(np.sum, 1, f_mat_bayes), index=X.index) 
scores """