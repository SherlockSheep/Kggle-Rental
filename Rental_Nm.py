import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import Adam

data_train = pd.read_json("/Users/yanyangma/Documents/pyLearning/Data_Mining/rental_data/train.json")
interest_level = pd.get_dummies(data_train['interest_level'])
data_train = pd.concat([data_train,interest_level],axis=1)
label_num_map={'high':2,'medium':1,'low':0}
data_train['label'] = data_train['interest_level'].apply(lambda x:label_num_map[x])
m_high_dic ={}
m_medium_dic ={}
m_low_dic ={}
for i in xrange(0,data_train['bathrooms'].count()):
    m_high_dic[data_train['manager_id'].values[i]] = 0;
    m_medium_dic[data_train['manager_id'].values[i]] = 0;
    m_low_dic[data_train['manager_id'].values[i]] = 0;

for i in xrange(0,data_train['bathrooms'].count()):
    if(data_train['interest_level'].values[i]=='high'):
        m_high_dic[data_train['manager_id'].values[i]] = m_high_dic[data_train['manager_id'].values[i]] + 1;
    elif(data_train['interest_level'].values[i]=='medium'):
        m_medium_dic[data_train['manager_id'].values[i]] = m_medium_dic[data_train['manager_id'].values[i]] + 1;
    else:
        m_low_dic[data_train['manager_id'].values[i]] = m_low_dic[data_train['manager_id'].values[i]] + 1;
        
data_train['m_high'] = 0
data_train['m_medium'] = 0
data_train['m_low'] = 0
data_train['p_num'] = 0
data_train['d_len'] = 0
data_train['f_num'] = 0;
for i in xrange(0,data_train['bathrooms'].count()):
    data_train['m_high'].values[i] = m_high_dic[data_train['manager_id'].values[i]]
    data_train['m_medium'].values[i] = m_medium_dic[data_train['manager_id'].values[i]]
    data_train['m_low'].values[i] = m_low_dic[data_train['manager_id'].values[i]]
    data_train['p_num'].values[i] = len(data_train['photos'].values[i])
    data_train['f_num'].values[i] = len(data_train['features'].values[i])
    data_train['d_len'].values[i] = len(data_train['description'].values[i])

data_train['is_train'] = np.random.uniform(0, 1, len(data_train)) <= .75
train,test = data_train[data_train['is_train']==True], data_train[data_train['is_train']==False]
train_set_high = train[train['interest_level']=='high'].sample(n=50000,replace=True)
train_set_low = train[train['interest_level']=='low'].sample(n=50000,replace=True)
train_set_medium = train[train['interest_level']=='medium'].sample(n=50000,replace=True)
train_set = pd.concat([train_set_high,train_set_low,train_set_medium])

features = ['bedrooms','bathrooms','price','latitude','longitude','m_high','m_medium','m_low','p_num','d_len','f_num']
targets = ['high','medium','low']

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=len(features)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

x = data_train[features].values
y = data_train[targets].values

model.fit(x, y,
          epochs=20,
          batch_size=128)

