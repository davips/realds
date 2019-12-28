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

dataset = read_arff('output.arff', 'time')
sorted(sklearn.metrics.SCORERS.keys())

from sklearn.neural_network import MLPRegressor as MLP
mlp = MLP(hidden_layer_sizes=(8,))

#from sklearn.neural_network import MLPClassifier as MLP
#mlp = MLP(hidden_layer_sizes=(8,))

#print("evaluating")
## scores = cross_val_score(mlp, dataset[0], dataset[1].ravel(), scoring='max_error', cv=10, n_jobs=-1)
#scores = cross_val_score(mlp, dataset[0], dataset[1].ravel(), cv=10, n_jobs=-1)
#print(scores)
