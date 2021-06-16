#!/usr/bin/env python
# coding: utf-8

# Objective:
# Build a recommendation system to recommend products to
# customers based on the their previous ratings for other
# products.

# Steps:
#     
# 1.Read and explore the given dataset. (Rename
# column/add headers, plot histograms, find data
# characteristics) - (2.5 Marks)
# 2. Take a subset of the dataset to make it less sparse/ denser.
# ( For example, keep the users only who has given 50 or
# more number of ratings ) - (2.5 Marks)
# 3. Split the data randomly into train and test dataset. ( For
# example, split it in 70/30 ratio) - (2.5 Marks)
# 4. Build Popularity Recommender model. - (20 Marks)
# 5. Build Collaborative Filtering model. - (20 Marks)
# 6. Evaluate both the models. ( Once the model is trained on
# the training data, it can be used to compute the error
# (RMSE) on predictions made on the test data.) - (7.5 Marks)
# 7. Get top - K ( K = 5) recommendations. Since our goal is to
# recommend new products for each user based on his/her
# habits, we will recommend 5 new products. - (7.5 Marks)
# 8. Summarise your insights. - (7.5 marks)

# In[1]:


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
import warnings; warnings.simplefilter('ignore')


# # Read and explore the given dataset. (Rename column/add headers, plot histograms, find data characteristics)

# In[2]:


electronicsData=pd.read_csv('ratings_Electronics.csv')


# In[3]:


electronicsData.head()


# In[4]:


#(Rename column/add headers)
columns=['userid', 'itemid', 'ratings','timestamp']
electronicsData=pd.read_csv('ratings_Electronics.csv',names=columns)


# In[5]:


electronicsData.head()


# In[6]:


#EDA
electronicsData.info()


# In[7]:


electronicsData.drop('timestamp',axis=1,inplace=True)# timestamp is not needed


# In[8]:


electronicsData.head()


# In[9]:


electronicsData['ratings'].describe().transpose()

#Insights
1. We can see that minimum rating provided is 1
2. And maximum rating which is provided is 5.
3. We will not be having outliers in such type of cases.
# In[10]:


electronicsData['ratings'].value_counts()# we can se the counts for each rating


# In[11]:


#checking missing values:
electronicsData.isnull().sum()


# We can see that there is no missing data

# In[12]:


#plot histograms
electronicsData['ratings'].plot.hist()


# Insight:
# There are many users who has given rating 5.
# 

# In[13]:


#Checking unique values in users and items
print('Number of unique users  ', electronicsData['userid'].nunique())
print('Number of unique items  ', electronicsData['itemid'].nunique())


# We can see that 4201696 users rated for 476002 items

# # Take a subset of the dataset to make it less sparse/ denser. 
# ( For example, keep the users only who has given 50 or more number of ratings )

# In[14]:


countsuser = electronicsData['userid'].value_counts()
countsuser.head()


# In[15]:


ratings_explicit = electronicsData[electronicsData['userid'].isin(countsuser[countsuser >= 50].index)]


# In[16]:


ratings_explicit.head()


# Here we can see that data has been reduced from 7824482 entries to 125871 enteries

# # Generating matrix table from explicit ratings table

# In[17]:


ratings_matrix = ratings_explicit.pivot(index='userid', columns='itemid', values='ratings').fillna(0)
userid = ratings_matrix.index
itemid = ratings_matrix.columns
print(ratings_matrix.shape)
ratings_matrix.head()


# # Split the data randomly into train and test dataset.
# ( For
# example, split it in 70/30 ratio)

# In[18]:


train_data, test_data = train_test_split(ratings_explicit, test_size = 0.3, random_state=0)
train_data.head()


# In[19]:


print('Shape of training data: ',train_data.shape)
print('Shape of testing data: ',test_data.shape)


# # Build Popularity Recommender model

# In[20]:


train_data.groupby('itemid')['ratings'].mean().sort_values(ascending=False).head()  


# In[21]:


train_data.groupby('itemid')['ratings'].count().sort_values(ascending=False).head()  


# In[22]:


ratings_mean_count = pd.DataFrame(train_data.groupby('itemid')['ratings'].mean()) 


# In[23]:


ratings_mean_count['rating_counts'] = pd.DataFrame(train_data.groupby('itemid')['ratings'].count())  


# In[24]:


ratings_mean_count.head()  


# In[25]:


#Getting the top 5 recommendations 
ratings_explicit.head()


# In[26]:


#  predictions
def recommendation(userid):     
    user_recommendation = ratings_explicit
          
    #Add user_id column for which the recommendations are being generated 
    user_recommendation['useid'] = userid 
      
    #Bring user_id column to the front 
    cols = user_recommendation.columns.tolist() 
    cols = cols[-1:] + cols[:-1] 
    user_recommendation = user_recommendation[cols] 
          
    return user_recommendation


# In[27]:


find_recom = [1,1000,7567]#user id entered.
for i in find_recom:
    print("The list of recommendations for the userId: %d\n" %(i))
    print(recommendation(i))    
    print("\n") 


# We will get same recommendations for all users on popularity baseis.

# # 5.Build Collaborative Filtering model. - (20 Marks)

# In[28]:


electronicsData=pd.read_csv('ratings_Electronics.csv')


# In[29]:


columns=['userid', 'itemid', 'ratings','timestamp']
electronicsData=pd.read_csv('ratings_Electronics.csv',names=columns)


# In[30]:


ratings_explicit = electronicsData[electronicsData['userid'].isin(countsuser[countsuser >= 50].index)]


# In[31]:


ratings_explicit.reset_index(inplace=True)
ratings_explicit.drop('index',axis=1)


# In[32]:


ratings_explicit.to_csv('file_ratings_explicit.csv')


# In[33]:


#from surprise import SVD
#from surprise import Dataset
#from surprise import accuracy
#from surprise.model_selection import train_test_split
#from surprise import Reader

# Load the movielens-100k dataset (download it if needed),
#reader = Reader(line_format='IDX userid itemid ratings', sep=',')
#reader = Reader(line_format='user item rating', sep=',')


#data=Dataset.load_from_file('ratings_explicit',reader=reader)

#
#data = Dataset.load_from_file('file_ratings_explicit.csv',reader=reader)

#reader = Reader(line_format='user item rating timestamp', sep=',')

#data=Dataset.load_from_file('file_ratings_explicit.csv',reader=reader)


# sample random trainset and testset
# test set is made of 25% of the ratings.
#trainset, testset = train_test_split(data, test_size=.25)

# We'll use the famous SVD algorithm.
#algo = SVD()
#algo.fit(trainset)
# Train the algorithm on the trainset, and predict ratings for the testset
#algo.fit(trainset)
#predictions = algo.test(testset)

# Then compute RMSE
#accuracy.rmse(predictions)


# In[ ]:


# sample random trainset and testset
# test set is made of 25% of the ratings.

#trainset, testset = train_test_split(data, test_size=.30)

# We'll use the famous SVD algorithm.
#algo = SVD()
#algo.fit(trainset)
# Train the algorithm on the trainset, and predict ratings for the testset
#algo.fit(trainset)
#predictions = algo.test(testset)

# Then compute RMSE
#accuracy.rmse(predictions)


# In[34]:


electronics = pd.concat([train_data, test_data]).reset_index()
electronics.head()


# In[35]:


# Matrix 
Table = electronics.pivot(index = 'userid', columns ='itemid', values = 'ratings').fillna(0)
Table.head()


# In[36]:


print('Shape of the pivot table: ', Table.shape)


# In[37]:


Table['user_index'] = np.arange(0, Table.shape[0], 1)
Table.head()


# In[38]:


Table.set_index(['user_index'], inplace=True)

Table.head()


# # SVD

# In[40]:


# Singular Value Decomposition
U, sigma, Vt = svds(Table, k = 10)


# In[41]:


# Constructing diagonal array in SVD
sigma = np.diag(sigma)
print('Diagonal matrix: \n',sigma)


# In[42]:


#Predicted ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
# Convert predicted ratings to dataframe
pred = pd.DataFrame(predicted_ratings, columns = Table.columns)
pred.head()


# In[45]:


#Ratingwise Predictions

def recommend(userid, Table, pred, num_recommendations):
    # index starts at 0  
    user_idx = userid-1 
    # Get and sort the user's ratings
    sorted_user_ratings = Table.iloc[user_idx].sort_values(ascending=False)
    #sorted_user_ratings
    sorted_user_predictions = Table.iloc[user_idx].sort_values(ascending=False)
    #sorted_user_predictions
    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
    temp.index.name = 'Recommended Items'
    temp.columns = ['user_ratings', 'user_predictions']
    temp = temp.loc[temp.user_ratings == 0]   
    temp = temp.sort_values('user_predictions', ascending=False)
    print('\nBelow are the recommended items for user(user_id = {}):\n'.format(userid))
    print(temp.head(num_recommendations))


# # Getting Top K recommendation

# In[46]:


userid = 1
num_recommendations = 5
recommend(userid, Table, pred, num_recommendations)    


# # RMSE

# In[48]:


pred.mean().head()


# In[49]:


rmse_df = pd.concat([ratings_matrix.mean(), pred.mean()], axis=1)
rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']
print(rmse_df.shape)
rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)
rmse_df.head()


# # Insight

# Popularity based recommendation system recommends the same set of 5 products to every user and hence based on popularity of the product irrespective of user past choices.
# 
# Collaborative Filtering is a personalised recommender system based on the past behavior of the user and irrespective of the popularity of the items.

# In[ ]:




