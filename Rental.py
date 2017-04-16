import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


data_train = pd.read_json("/Users/yanyangma/Documents/pyLearning/Data_Mining/rental/train.json")
interest_level = pd.get_dummies(data_train['interest_level'])
data_train = pd.concat([data_train,interest_level],axis=1)
label_num_map={'high':2,'medium':1,'low':0}
data_train['label'] = data_train['interest_level'].apply(lambda x:label_num_map[x])

data_train['is_train'] = np.random.uniform(0, 1, len(data_train)) <= .75
train,test = data_train[data_train['is_train']==True], data_train[data_train['is_train']==False]
train_set_high = data_train[data_train['interest_level']=='high'].sample(n=10000,replace=True)
train_set_low = data_train[data_train['interest_level']=='low'].sample(n=10000,replace=True)
train_set_medium = data_train[data_train['interest_level']=='medium'].sample(n=10000,replace=True)
train_set = pd.concat([train_set_high,train_set_low,train_set_medium])
features = data_train.columns[:2].append(data_train.columns[13:14]).append(data_train.columns[10:11]).append(data_train.columns[8:9])

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

'''Output
data_test = pd.read_json("/Users/yanyangma/Documents/pyLearning/Data_Mining/rental/test.json")
data_test['interest_level'] = clf.predict(data_test[features])
probs = clf.predict_proba(data_test[features])
result = pd.DataFrame(data_test['listing_id'])
probst = [[r[col] for r in probs] for col in range(len(probs[0]))]
result['high'] = probst[0]
result['low'] = probst[1]
result['medium'] = probst[2]
'''