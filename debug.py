import bhad
import utils as util
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

seed = 42  
outlier_prob_true = .01         # probab. for outlier ; should be consistent with contamination rate in your model
k = 30                          # feature dimension 
N = 2*10**4                     # sample size

# Specify first and second moments for each component  
bvt = util.mvt2mixture(thetas = {'mean1' : np.full(k,-1), 'mean2' : np.full(k,.5), 
                                'Sigma1' : np.eye(k)*.4, 'Sigma2' : np.eye(k)*.1, 
                                'nu1': 3.*k, 'nu2': 3.*k}, seed = seed, gaussian = False)

# Get latent draws and observations:
#------------------------------------
y_true, dataset = bvt.draw(n_samples = N, k = k, p = outlier_prob_true)

print(dataset.shape)


reload(bhad)

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

reload(bhad)
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

reload(bhad)
#-------------------------------------

pipe = bhad.BHAD(contamination = 0.01, verbose=False)

y_pred_train = pipe.fit_predict(X_train)   

scores_train = pipe.decision_function(X_train)

y_pred_test = pipe.predict(X_test)