n_split=3 
epochs=50
batch_size=1000000

import tensorflow as tf
from keras.models import Sequential 
from keras.utils import to_categorical 


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
X, Y = dataset[0], to_categorical(dataset[1].ravel())
	
def create_model():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(100, input_shape=(12,), activation = 'relu'))
  model.add(tf.keras.layers.Dense(50, activation = 'relu'))
  model.add(tf.keras.layers.Dense(2, activation = 'softmax'))
  model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
  return model

from sklearn.model_selection import KFold 
res = []
for train_index,test_index in KFold(n_split).split(X): 
  x_train,x_test=X[train_index],X[test_index] 
  y_train,y_test=Y[train_index],Y[test_index] 
   
  model=create_model() 
  model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size) 
   
  res.append(model.evaluate(x_test, y_test))

print(res)
#model.fit(x_train, to_categorical(y_train), epochs=10, batch_size=393889, validation_split=0.33)  
