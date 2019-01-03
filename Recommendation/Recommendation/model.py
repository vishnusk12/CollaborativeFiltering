# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:42:33 2018

@author: Vishnu
"""

import pandas as pd 
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
# from sklearn.decomposition import TruncatedSVD

path = os.getcwd() + '/Recommendation/Recommendation'

data_path = path + '/data/'

df_cuisine = pd.read_csv(data_path + "chefmozcuisine.csv")
df_cuisine.dropna()
df_cuisine.Rcuisine = df_cuisine.Rcuisine.str.replace('_', ' ')
df_cuisine.Rcuisine = df_cuisine.Rcuisine.str.replace('-', ' ')
df_user = pd.read_csv(data_path + "usercuisine.csv")
df_user.dropna()
df_place = pd.read_csv(data_path + "geoplaces2.csv")
df_place.dropna()
df_place['address'].replace('?', 'Calle Mezquite Fracc Framboyanes', inplace=True)
df_place['city'].replace('?', 'victoria', inplace=True)
df_place['state'].replace('?', 'Tamaulipas', inplace=True)
df_place['country'].replace('?', 'Mexico', inplace=True)
df_place['country'].replace('mexico country', 'Mexico', inplace=True)
df_place['country'].replace('mexico', 'Mexico', inplace=True)
df_rating = pd.read_csv(data_path + "rating_final.csv")
df_rating.dropna()
df_rating = df_rating.groupby(['placeID']).agg([np.average])
df_rating.columns = ['rating', 'food_rating', 'service_rating']
df_rating['placeID'] = df_rating.index
unique_id = list(df_rating['placeID'].unique())
cuisines = df_cuisine[df_cuisine['placeID'].isin(unique_id)]
df_merged = pd.merge(df_rating, df_cuisine, on='placeID')
df = pd.merge(df_merged, df_user, on='Rcuisine')
userItemRatingMatrix = pd.pivot_table(df, values='food_rating',
                                      index=['Rcuisine'], 
                                      columns=['userID']).fillna(0)

# userItemRatingMatrix = pd.pivot_table(df, values='food_rating',
#                                         index=['userID'], 
#                                         columns=['Rcuisine']).fillna(0)

unique_cuisines = list(userItemRatingMatrix.index.values)

def TopRestaurents(cuisine):
    places = list(df_cuisine.loc[df_cuisine['Rcuisine'] == str(cuisine).title(), 'placeID'])
    restaurents = df_rating[df_rating['placeID'].isin(places)]
    top_restaurents = list(restaurents.sort_values(by=['food_rating'], ascending=False)[:5]['placeID'])
    address = df_place[df_place['placeID'].isin(top_restaurents)]
    address['fullAddress'] = address['name'] + ', ' + address['city'] + ', ' + address['state'] + ', ' + address['country']
    top_rated_restaurents = list(address['fullAddress'])
    dict_restaurents = {}
    if len(top_rated_restaurents) > 1:
        dict_restaurents['message_rest'] = 'The top ' + str(len(top_rated_restaurents)) + ' restaurents serving ' + str(cuisine).title() + ' cuisine are:'
        dict_restaurents['restaurents'] = top_rated_restaurents
    elif len(top_rated_restaurents) == 1:
        dict_restaurents['message_rest'] = 'The top restaurent serving ' + str(cuisine).title() + ' cuisine is:'
        dict_restaurents['restaurents'] = top_rated_restaurents
    else:
        dict_restaurents['message_rest'] = "Sorry.. We don't have the data to recommend restaurents serving your prefered cuisine."
        dict_restaurents['restaurents'] = top_rated_restaurents
    return dict_restaurents

# Memory-Based Collaborative Filtering

def SimilarCuisines(cuisine):
    userItemRatingCSRMatrix = csr_matrix(userItemRatingMatrix.values)
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(userItemRatingCSRMatrix)
    query_index = userItemRatingMatrix.index.get_loc(str(cuisine).title())
    distances, indices = knn.kneighbors(userItemRatingMatrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)
    list_cuisines = []
    for i in range(0, len(distances.flatten())):
        list_cuisines.append(userItemRatingMatrix.index[indices.flatten()[i]])
    dict_cuisines = {}
    if len(list_cuisines) > 0:
        dict_cuisines['message_cuisine'] = 'Users who searched ' + str(cuisine).title() + ' cuisine may also like: '
        if str(cuisine).title() not in list_cuisines:
            dict_cuisines['cuisines'] = list_cuisines
        else:
            list_cuisines.remove(str(cuisine).title())
            dict_cuisines['cuisines'] = list_cuisines
    else:
        dict_cuisines['message_cuisine'] = 'No Matches'
        dict_cuisines['cuisines'] = list_cuisines
    return dict_cuisines

# Model-Based Collaborative Filtering

# def SimilarCuisines(cuisine):
#     X = userItemRatingMatrix.values.T
#     SVD = TruncatedSVD(n_components=12, random_state=17)
#     matrix = SVD.fit_transform(X)
#     corr = np.corrcoef(matrix)
#     cuisines = userItemRatingMatrix.columns
#     cuisine_list = list(cuisines)
#     ind = cuisine_list.index(str(cuisine).title())
#     corr_ind  = corr[ind]
#     dict_cuisines = {}
#     dict_cuisines['message_cuisine'] = 'Users similar to you have also searched for: '
#     dict_cuisines['cuisines'] = list(cuisines[(corr_ind<1.0) & (corr_ind>0.3)])[:5]
#     return dict_cuisines
