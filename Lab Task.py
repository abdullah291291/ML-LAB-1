#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv('ECG200_TRAIN.csv',delimiter = "  ",header = None)


# In[5]:


col1 = df[0]
col1.value_counts()
df = df.sort_values(by=0)
df


# In[6]:


df1 = df[:31]
df2 = df[31:]
# df = df.reindex(np.random.permutation(df.index)) #shuffling


# In[7]:


fol1 = pd.concat([df1.iloc[0:3],df2.iloc[0:7]])
fol2 = pd.concat([df1.iloc[3:6],df2.iloc[7:14]])
fol3 = pd.concat([df1.iloc[6:9],df2.iloc[14:21]])
fol4 = pd.concat([df1.iloc[9:12],df2.iloc[21:28]])
fol5 = pd.concat([df1.iloc[12:15],df2.iloc[28:35]])
fol6 = pd.concat([df1.iloc[15:18],df2.iloc[35:42]])
fol7 = pd.concat([df1.iloc[18:21],df2.iloc[42:49]])
fol8 = pd.concat([df1.iloc[21:24],df2.iloc[49:56]])
fol9 = pd.concat([df1.iloc[24:27],df2.iloc[56:63]])
fol10 = pd.concat([df1.iloc[27:30],df2.iloc[63:69]])


# In[8]:


train1 = pd.concat([fol2,fol3,fol4,fol5,fol6,fol7,fol8,fol9,fol10])
test1 = fol1
label1_train = train1[0]
label1_test = test1[0]
train1 = train1.drop(0,axis = 1)
test1 = test1.drop(0,axis = 1)

train2 = pd.concat([fol1,fol3,fol4,fol5,fol6,fol7,fol8,fol9,fol10])
test2 = fol2
label2_train = train2[0]
label2_test = test2[0]
train2 = train2.drop(0,axis = 1)
test2 = test2.drop(0,axis = 1)

train3 = pd.concat([fol2,fol1,fol4,fol5,fol6,fol7,fol8,fol9,fol10])
test3 = fol3
label3_train = train3[0]
label3_test = test3[0]
train3 = train3.drop(0,axis = 1)
test3 = test3.drop(0,axis = 1)

train4 = pd.concat([fol2,fol3,fol1,fol5,fol6,fol7,fol8,fol9,fol10])
test4 = fol4
label4_train = train4[0]
label4_test = test4[0]
train4 = train4.drop(0,axis = 1)
test4 = test4.drop(0,axis = 1)

train5 = pd.concat([fol2,fol3,fol4,fol1,fol6,fol7,fol8,fol9,fol10])
test5 = fol5
label5_train = train5[0]
label5_test = test5[0]
train5 = train5.drop(0,axis = 1)
test5 = test5.drop(0,axis = 1)

train6 = pd.concat([fol2,fol3,fol4,fol5,fol1,fol7,fol8,fol9,fol10])
test6 = fol6
label6_train = train6[0]
label6_test = test6[0]
train6 = train6.drop(0,axis = 1)
test6 = test6.drop(0,axis = 1)

train7 = pd.concat([fol2,fol3,fol4,fol5,fol6,fol1,fol8,fol9,fol10])
test7 = fol7
label7_train = train7[0]
label7_test = test7[0]
train7 = train7.drop(0,axis = 1)
test7 = test7.drop(0,axis = 1)

train8 = pd.concat([fol2,fol3,fol4,fol5,fol6,fol7,fol1,fol9,fol10])
test8 = fol8
label8_train = train8[0]
label8_test = test8[0]
train8 = train8.drop(0,axis = 1)
test8 = test8.drop(0,axis = 1)

train9 = pd.concat([fol2,fol3,fol4,fol5,fol6,fol7,fol8,fol1,fol10])
test9 = fol9
label9_train = train9[0]
label9_test = test9[0]
train9 = train9.drop(0,axis = 1)
test9 = test9.drop(0,axis = 1)

train10 = pd.concat([fol2,fol3,fol4,fol5,fol6,fol7,fol8,fol9,fol1])
test10 = fol10
label10_train = train10[0]
label10_test = test10[0]
train10 = train10.drop(0,axis = 1)
test10 = test10.drop(0,axis = 1)


# # Applying the MODEL

# In[9]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[10]:


model.fit(train1,label1_train)
score1 = model.score(test1,label1_test)

model.fit(train2,label2_train)
score2 = model.score(test2,label2_test)

model.fit(train3,label3_train)
score3 = model.score(test3,label3_test)

model.fit(train4,label4_train)
score4 = model.score(test4,label4_test)

model.fit(train5,label5_train)
score5 = model.score(test5,label5_test)

model.fit(train6,label6_train)
score6 = model.score(test6,label6_test)

model.fit(train7,label7_train)
score7 = model.score(test7,label7_test)

model.fit(train8,label8_train)
score8 = model.score(test8,label8_test)

model.fit(train9,label9_train)
score9 = model.score(test9,label9_test)

model.fit(train10,label10_train)
score10 = model.score(test10,label10_test)

Score = np.average([score1,score2,score3,score4,score5,score6,score7,score8,score9,score10])


# In[11]:


Score #Average Accuracy


# # USING train_test_split

# In[18]:


from sklearn.model_selection import train_test_split
X = df.drop(0,axis = 1)
y = df[0]
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[19]:


model.fit(X_train,y_train)


# In[20]:


model.score(X_test,y_test)


# In[ ]:





# In[ ]:




