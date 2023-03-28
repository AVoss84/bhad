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


reload(model)

from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ('discrete' , utils.discretize(nbins = None, verbose = False)),      # step only needed if continous features are present
    ('model', model.BHAD(contamination = 0.01))
])

y_pred_train = pipe.fit_predict(X_train)
#pipe.score_samples(dataset)   
scores_train = pipe.decision_function(X_train)
scores_train






reload(model)
reload(utils)

bm = model.BHAD(contamination = 0.01, nbins = None, verbose=True)

y_pred_train = bm.fit_predict(X_train)   
#scores_train = bm.decision_function(X_train)
scores_train = bm.anomaly_scores





from sklearn.pipeline import Pipeline

reload(bhad_old)
reload(util)

pipe = Pipeline(steps=[
    # if nbins = None, this will automatically select the optimal bin numbers 
    # based on the MAP estimate (but will make computation slower!)
    ('discrete' , util.discretize(nbins = None, verbose = False)),      # step only needed if continous features are present
    ('model', bhad_old.BHAD(contamination = 0.01))
])

y_pred_train = pipe.fit_predict(X_train)   
scores_train = pipe.decision_function(X_train) 

y_pred_test = pipe.predict(X_test)
scores_test = pipe.decision_function(X_test)
#---------------------------------------------
disc = utils.discretize(nbins = None, verbose = False)
X_tilde = disc.fit_transform(X_train)

model = model.BHAD(contamination = 0.01)

model.fit(X_tilde)   

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

#y_pred_test = pipe.predict(X_test)
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
scores