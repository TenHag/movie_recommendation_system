#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


credits =pd.read_csv('tmdb_5000_credits.csv')
movies=pd.read_csv('tmdb_5000_movies.csv')


# In[3]:


credits.head()


# In[4]:


movies.head()


# In[5]:


movies=movies.merge(credits,on='title')


# In[6]:


movies.head()


# In[7]:


movies=movies[['movie_id','title','genres','keywords','overview','cast','crew']]


# In[8]:


movies.info()


# In[9]:


movies.isnull().sum()


# In[10]:


movies.dropna(inplace=True)


# In[11]:


movies.duplicated().sum()


# In[12]:


import ast


# In[13]:


def ajax(obj):
    line=[]
    for i in ast.literal_eval(obj):
        line.append(i['name'])
    return line


# In[14]:


movies['genres']=movies['genres'].apply(ajax)


# In[15]:


movies


# In[16]:


movies['keywords']=movies['keywords'].apply(ajax)


# In[17]:


movies


# In[18]:


def fener(obj):
    tadic=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            tadic.append(i['name'])
            counter=counter+1
        else:
            break
    return tadic


# In[19]:


movies['cast']=movies['cast'].apply(fener)


# In[20]:


movies


# In[21]:


def gala(haha):
    df=[]
    for i in ast.literal_eval(haha):
        if i['job']=='Director':
            df.append(i['name'])
            break
    return df


# In[22]:


movies['crew']=movies['crew'].apply(gala)


# In[23]:


movies


# In[24]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[25]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[26]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['crew']+movies['cast']


# In[27]:


movies.head()


# In[28]:


new_df=movies[['movie_id','title','tags']]


# In[29]:


new_df


# In[30]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[31]:


new_df


# In[32]:


import nltk


# In[33]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[34]:


def jfc(meme):
    rb=[]
    for i in meme.split():
        rb.append(ps.stem(i))
    return " ".join(rb)


# In[35]:


new_df['tags']=new_df['tags'].apply(jfc)


# In[36]:


new_df.head()


# In[37]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[38]:


new_df.head()


# In[39]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:




# In[40]:


gk=CountVectorizer(max_features=5000,stop_words='english')


# In[41]:


vectors =gk.fit_transform(new_df['tags']).toarray()
vectors


# In[43]:


gk.get_feature_names_out()


# In[44]:


from sklearn.metrics.pairwise import cosine_similarity 


# In[45]:


sim =cosine_similarity(vectors)


# In[46]:


sorted(list(enumerate(sim[0])), reverse =True, key= lambda x:x[1])[1:6]


# In[47]:


def recommend(movie):
    inox=new_df[new_df['title']==movie].index[0]
    dist=sim[inox]
    movie_list =sorted(list(enumerate(dist)), reverse =True, key= lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        
    return


# In[49]:


recommend('Avatar')


# In[52]:


recommend('Iron Man')


# In[53]:


import pickle


# In[56]:


pickle.dump(new_df, open('movies.pkl','wb'))


# In[58]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[59]:


pickle.dump(sim,open('similarity.pkl','wb'))


# In[ ]:




