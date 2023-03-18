from unicodedata import category
from bhad.utils import (discretize, mvt2mixture)
from bhad.model import BHAD
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import bhad.explainer as expl
import random

reload(expl)

seed = 42  
outlier_prob_true = .01         # probab. for outlier ; should be consistent with contamination rate in your model

k = 33                          # feature dimension 
N = 10**3                     # sample size

# Specify first and second moments for each component  
bvt = mvt2mixture(thetas = {'mean1' : np.full(k,-1), 'mean2' : np.full(k,.5), 
                                'Sigma1' : np.eye(k)*.4, 'Sigma2' : np.eye(k)*.1, 
                                'nu1': 3.*k, 'nu2': 3.*k}, seed = seed, gaussian = False)

# Get latent draws and observations:
#------------------------------------
y_true, dataset = bvt.draw(n_samples = N, k = k, p = outlier_prob_true)

print(dataset.shape)

# dataset['var30'] = np.array(random.choices(['A', 'B', 'C'], k=N))
# dataset

dataset.info(verbose=True)

dataset.select_dtypes(include=['float', 'int'])
dataset.select_dtypes(include=['object', 'category'])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset, y_true, test_size=0.33, random_state=42)


print(X_train.shape)
print(X_test.shape)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

#------------------------------------------------

# from sklearn.pipeline import Pipeline


# x_trans = discretize(nbins = 20).fit_transform(dataset)

# x_trans.head()

# x_trans.info(verbose=True)

# x_train = np.array([["A1","B1","C1"],["A2","B1","C2"]])
# x_test = np.array([["A1","B2","C2"]]) # As you can see, "B2" is a new attribute for column B

# from sklearn.preprocessing import OneHotEncoder

#df = pd.DataFrame(x_train)
# df = x_trans

# oos_token = 'infrequent'     #'OTHERS'
# unique_categories_ = [df[var].unique().tolist() + [oos_token] for var in df.columns]
# #unique_categories_

# enc = OneHotEncoder(handle_unknown='infrequent_if_exist', dtype = int, categories = unique_categories_)
# #enc = OneHotEncoder(handle_unknown='ignore', dtype = int, categories = [['A1', 'A2'], ['B1', 'B2'], ['C1', 'C2']])

# x_train

# x_train_dummy = enc.fit_transform(x_train)
# x_train_dummy.toarray()

# enc.categories_

# x_test
# x_test_dummy = enc.transform(x_test)
# x_test_dummy.toarray()


# x_dummy = enc.fit_transform(x_trans)
# x_dummy.shape

# enc = util.onehot_encoder(prefix_sep='__') 

# x_dummy = enc.fit_transform(x_trans)
# x_dummy.shape


# bh = BHAD(contamination = 0.01)
# bh
# print(bh.__dict__)

# bh.fit(x_trans)

#dataset.dtypes

#reload(bhad)


#num = list(dataset.columns)
# num.remove('var30')

disco = discretize(nbins = None, verbose = False)

xtrans = disco.fit_transform(X_train)
xtrans

[str(v) for v in disco.save_binnings_['var0'].tolist()] + ['infrequent']

X_train.columns

#unique_categories_ = [X_train[var].unique().tolist() + ['infrequent'] for var in X_train.columns]

unique_categories_ = [[v for v in disco.save_binnings_[var].tolist()] + ['infrequent'] for var in X_train.columns]

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='infrequent_if_exist', dtype = int, categories = unique_categories_)
df_one = enc.fit_transform(xtrans).toarray()  
df_one.shape
np.sum(df_one, axis=1)
df_one[0,:]

from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ('discrete' , discretize(nbins = None, verbose = False)),     
    ('model', BHAD(contamination = 0.01, numeric_features = list(X_train.columns)))
])


#pipe.fit(X_train)

y_pred_train = pipe.fit_predict(X_train)     
#scores_train = pipe.decision_function(X_train)

bh = pipe.named_steps['model']
bh.df_one.shape
bh.df_one_.shape

y_pred_test = pipe.predict(X_test)
#scores_test = pipe.decision_function(X_test)


from bhad import explainer

reload(explainer)

local_expl = explainer.Explainer(pipe.named_steps['model'], pipe.named_steps['discrete'])

expl = local_expl.fit()

df_orig, expl_thresholds = local_expl.get_explanation()

df_orig.shape

for obs, ex in enumerate(df_orig.explanation.values):
    if (obs % 1000) == 0:
        print(f'\nObs. {obs}:\n', ex)

