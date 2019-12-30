fst, snd = 8, 0
n_split=5
epochs=200000
batch_size=800000
class_weight = {0: 1., 1: 50.}
optimizer = 'adam'
gpus=0

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.utils import to_categorical 
import arff, numpy as np
import pandas as pd
import arff, numpy as np
import _pickle as pickle
import sklearn
from sklearn.model_selection import cross_val_score
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
def read_arff(filename, target='class'):
    data = arff.load(open(filename, 'r'), encode_nominal=True)
    df = pd.DataFrame(data['data'],
                      columns=[attr[0] for attr in data['attributes']])
    return read_data_frame(df, filename, target)

def as_column_vector(vec):
    return vec.reshape(len(vec), 1)

def uuid(x):
    return '1'

def read_data_frame(df, filename, target='class'):
    Y = target and as_column_vector(df.pop(target).values.astype('float'))
    X = df.values.astype('float')  # Do not call this before setting Y!
    name = filename.split('/')[-1] + '_' + uuid(pickle.dumps((X, Y)))
    return X,Y

dataset = read_arff('cla.arff', 'time')
X, Y = dataset[0], to_categorical(dataset[1].ravel())

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(2, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer=optimizer)
	return model

# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
