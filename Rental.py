import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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
clf = RandomForestClassifier(n_jobs=2,n_estimators=100)
y = data_train['label']
clf.fit(data_train[features],y)

#test
preds = clf.predict(test[features])
pd.crosstab(test['label'], preds, rownames=['actual'], colnames=['preds'])
#

data_test = pd.read_json("/Users/yanyangma/Documents/pyLearning/Data_Mining/rental_data/test.json")
data_test['m_high'] = 0
data_test['m_medium'] = 0
data_test['m_low'] = 0
data_test['p_num'] = 0
data_test['d_len'] = 0
data_test['f_num'] = 0

for i in xrange(0,data_test['bathrooms'].count()):
    if(m_high_dic.has_key(data_test['manager_id'].values[i])):
        data_test['m_high'].values[i] = m_high_dic[data_test['manager_id'].values[i]]
        data_test['m_medium'].values[i] = m_medium_dic[data_test['manager_id'].values[i]]
        data_test['m_low'].values[i] = m_low_dic[data_test['manager_id'].values[i]]
        data_test['p_num'].values[i] = len(data_test['photos'].values[i])
        data_test['f_num'].values[i] = len(data_test['features'].values[i])
        data_test['d_len'].values[i] = len(data_test['description'].values[i])

data_test['label'] = clf.predict(data_test[features])
probs = clf.predict_proba(data_test[features])
result = pd.DataFrame(data_test['listing_id'])
probst = [[r[col] for r in probs] for col in range(len(probs[0]))]
result['high'] = probst[2]
result['medium'] = probst[1]
result['low'] = probst[0]
result.to_csv('/Users/yanyangma/Documents/pyLearning/Data_Mining/rental_data/result.csv')

'''Finding argument:Best:n_estimators:290,max_depth:68,score:0.747487437186
s = 0
c = 0
bi = 0
bj = 0
for i in xrange(1,500):
	#for j in xrange(1,100):
		#print("Round:"+str(c)+",n_estimators:"+str(i)+",max_depth"+str(j))
		c = c+1
		j=68
		print(c)
		clf = RandomForestClassifier(n_jobs=2,n_estimators=i)
		y = train_set['label']
		clf.fit(train_set[features],y)
		#test
		snew = clf.score(data_train[features],data_train['label'])
		if(snew>=s):
			s = snew
			bi = i
			bj = j
			print("n_estimators:"+str(i)+",max_depth"+str(j)+",score:"+str(s))
		#

print("Best"+"n_estimators:"+str(bi)+",max_depth"+str(bj)+",score:"+str(s))
'''
