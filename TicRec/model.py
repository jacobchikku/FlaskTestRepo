
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
nltk.download('all')
import numpy as np
import pickle


# In[2]:


dfTickets = pd.read_csv('Userreport.csv',encoding='cp1252' )
dfTickets


# In[3]:


def cleanDataset(dataset, columnsToClean, regexList):
    for column in columnsToClean:
        for regex in regexList:
            dataset[column] = removeString(dataset[column], regex)
    return dataset


def removeString(data, regex):
    return data.str.lower().str.replace(regex, ' ')


# In[4]:


def getRegexList():
    regexList = []

    regexList += ['^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$']
    regexList += ['[\w\d\-\_\.]+ @ [\w\d\-\_\.]+']
    regexList += ['[^a-zA-Z]']

    return regexList


# In[5]:


columnsToClean = ['Short description']


# In[6]:


dfTickets =cleanDataset(dfTickets, columnsToClean, getRegexList())


# In[7]:


dfTickets


# In[8]:


dfTickets.groupby('Classification').describe()


# In[9]:


list = [' SAS Filesystem/Quota',' SAS Database',' SAS Service Requests',' SAS Configuration',' SAS Others',' SAS - UC4 Job Issues',' SAS AVC Client Issue',' SAS Access Issue',' SAS Backup',' SAS Database',' SAS GPFS',' SAS Linux Server',' SAS Network Issue',' SAS Performance',' SAS Platform Access/Availability',' SAS Portal issue',' SAS Service Requests',' SAS Server Issue',' SAS Service Requests',' sas configuration']


# In[10]:


dfTickets = dfTickets.loc[dfTickets['Classification'].isin(list)]


# In[11]:


dfTickets


# In[12]:


dfTickets.groupby('Classification').describe()


# In[13]:


from nltk.corpus import stopwords


# In[14]:


import string
def text_process(mess):
    """
    1. Remove all stopwords
    2. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]



# In[15]:


dfTickets['Short description'].head(50).apply(text_process)


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(dfTickets['Short description'])

# Print total number of vocab words
vocab_size=len(bow_transformer.vocabulary_)


# In[17]:


dfTickets_bow = bow_transformer.transform(dfTickets['Short description'])
dfTickets_bow


# In[18]:


print('Shape of Sparse Matrix: ', dfTickets_bow.shape)
print('Amount of Non-Zero occurences: ', dfTickets_bow.nnz)


# In[19]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(dfTickets_bow)


# In[20]:


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['sas']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['failure']])


# In[21]:


messages_tfidf = tfidf_transformer.transform(dfTickets_bow)
print(dfTickets_bow.shape)


# In[22]:


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(dfTickets['Short description'], dfTickets['Classification'], test_size=0.3)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# In[23]:


# #deeplearning
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# # Initialising the ANN



# model = Sequential()

# model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_shape=(1100,)))
# # Adding the second hidden layer
# model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))# Adding the output layer
# model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
# # Compiling the ANN
# classifier = model.compile(optimizer = 'adam', loss = 'mean_squared_error')
# model.summary()
    
# pipeline = Pipeline([
#     ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
#     ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
#     ('model',model)
# ])


# In[24]:


#Ignore the upcoming code


# In[25]:


# clf = KerasRegressor(build_fn=create_model,verbose=0)

# scaler = StandardScaler()

# param_grid = {
#     'clf__optimizer':['rmsprop','adam','adagrad'],
#     'clf__epochs':[4,8],
#     'clf__dropout':[0.1,0.2],
#     'clf__kernel_initializer':['glorot_uniform','normal','uniform']
# }

# pipeline = Pipeline([
#     ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
#     ('tfidf', TfidfTransformer())
#     'clf',clf
# ])
# grid = GridSearchCV(pipeline, cv=3, param_grid=param_grid)
# grid.fit(msg_train, label_train)


# In[26]:


# #training with lstm
# classifier.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# history = classifier.fit(msg_train,label_train,
#                     batch_size=50,
#                     epochs=3,
#                     verbose=1,
#                     validation_split=0.1)


# In[27]:


# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline

# pipeline = Pipeline([
#     ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
#     ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
# ])


# In[28]:


# 


# In[29]:


# 


# In[68]:


# label_test=label_test.reset_index()['Classification']


# In[31]:


# from sklearn.metrics import classification_report
# print(classification_report(prediction,label_test))


# In[62]:


#  parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3)}


# In[32]:


pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline = pipeline.fit(msg_train, label_train)


# In[36]:


pipeline.fit(msg_train,label_train)
prediction = pipeline.predict(msg_test)


# In[38]:


import joblib


# In[39]:


joblib.dump(prediction, 'model.pkl')


# In[31]:


# from sklearn.model_selection import GridSearchCV, train_test_split
# gs_clf = GridSearchCV(pipeline, parameters, n_jobs=-1)

# gs_clf = gs_clf.fit(msg_train, label_train)


# In[56]:


xt = ["cannot open file"]


# In[57]:


prd = pipeline.predict(xt)


# In[58]:


prb = pipeline.predict_proba(xt)


# In[59]:


print(str(prd[0]))


# In[51]:




