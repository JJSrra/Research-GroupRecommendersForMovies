import numpy as np
import pandas as pd
import evaluation
import collections

def predict_ranking_from_group(group, movies, movie_ratings):
    preference_matrix = get_preference_matrix(group, movies, movie_ratings)


def get_preference_matrix(users, movies, movie_ratings):
    preference_matrix = np.array([])
    for user in users:
        user_ratings = []
        for movie in movies:
            user_ratings.append(movie_ratings[user][movie])

        user_ranking = evaluation.sort_movies_by_ranking(user_ratings, movies)
        preference_matrix = np.append(preference_matrix, user_ranking)

    return preference_matrix.reshape(len(users), len(movies))

def get_user_status(preference_matrix, user_index):
    evaluated_movies = preference_matrix.shape[1]
    user_status = most_preferred_score(preference_matrix, user_index)

    if evaluated_movies > 1:
        user_status += preferred_score(preference_matrix, user_index)
    if evaluated_movies > 2:
        user_status += least_preferred_score(preference_matrix, user_index)

    return user_status

def most_preferred_score(preference_matrix, user_index):
    most_preferred_item = preference_matrix[user_index][0]
    num_users = preference_matrix.shape[0]

    sum_same_most_preferred_item = collections.Counter(preference_matrix[:,0])[most_preferred_item]
    return sum_same_most_preferred_item / num_users

def preferred_score(preference_matrix, user_index):
    preferred_item = preference_matrix[user_index][1]
    num_users = preference_matrix.shape[0]

    sum_same_preferred_item = collections.Counter(preference_matrix[:,1])[preferred_item]
    return (sum_same_preferred_item * 0.5) / num_users

def least_preferred_score(preference_matrix, user_index):
    least_preferred_item = preference_matrix[user_index][2]
    num_users = preference_matrix.shape[0]

    sum_same_least_preferred_item = collections.Counter(preference_matrix[:,2])[least_preferred_item]
    return (sum_same_least_preferred_item * 0.25) / num_users