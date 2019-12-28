n_split=10
epochs=10000
batch_size=1000000
class_weight = {0: 1., 1: 50.}

import tensorflow as tf
from keras.models import Sequential 
from keras.utils import to_categorical 
import arff, numpy as np
import pandas as pd
import arff, numpy as np
import _pickle as pickle
import sklearn
from sklearn.model_selection import cross_val_score
from keras import backend as K
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

dataset = read_arff('output.arff', 'time')
X, Y = dataset[0], to_categorical(dataset[1].ravel())
	
def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def create_model():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(100, input_shape=(6,), activation = 'relu'))
  model.add(tf.keras.layers.Dense(50, activation = 'relu'))
  model.add(tf.keras.layers.Dense(2, activation = 'softmax'))
  model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy', f1_m] )
  return model

from sklearn.model_selection import KFold 
res = []
for train_index,test_index in KFold(n_split, shuffle=False).split(X): 
  x_train,x_test=X[train_index],X[test_index] 
  y_train,y_test=Y[train_index],Y[test_index] 
   
  model=create_model() 
  model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weight) 
   
  res.append(model.evaluate(x_test, y_test))

print(res)
#model.fit(x_train, to_categorical(y_train), epochs=10, batch_size=393889, validation_split=0.33)  
