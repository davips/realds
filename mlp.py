fst, snd = 8, 0
n_splits=5
epochs=200000
batch_size=800000
#class_weight = {0: 1., 1: 50.}
optimizer = 'adam'

import arff, numpy as np
import pandas as pd
import arff, numpy as np
import _pickle as pickle
import sklearn
from sklearn.model_selection import cross_val_score

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
sorted(sklearn.metrics.SCORERS.keys())

#from sklearn.neural_network import MLPRegressor as MLP
#mlp = MLP(hidden_layer_sizes=(20,10,5))

from sklearn.neural_network import MLPClassifier as MLP

hidden_layer_sizes=(fst, ) if snd==0 else (fst, snd)
mlp = MLP(max_iter=epochs, hidden_layer_sizes=hidden_layer_sizes, batch_size=batch_size)

print("evaluating")
## scores = cross_val_score(mlp, dataset[0], dataset[1].ravel(), scoring='max_error', cv=10, n_jobs=-1)
scores = cross_val_score(mlp, dataset[0], dataset[1].ravel(), cv=n_splits, n_jobs=-1)
print(scores)
