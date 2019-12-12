import numpy as np
import evaluation
import collections

def predict_ranking_from_group(group, movies, movie_ratings):
    preference_matrix = get_preference_matrix(group, movies, movie_ratings)
    user_status_list = [get_user_status(preference_matrix, user) for user in range(0, len(group))]
    highest_status_user = np.argsort(-np.array(user_status_list))[0] # Negated for descending order
    return preference_matrix[highest_status_user]
    
def get_preference_matrix(users, movies, movie_ratings):
    preference_matrix = np.empty([len(users), len(movies)])
    for i in range(0, len(users)):
        user_ratings = []
        for movie in movies:
            user_ratings.append(0.0 if movie not in movie_ratings[users[i]].keys() else movie_ratings[users[i]][movie])

        user_ranking = evaluation.sort_movies_by_ranking(np.array(user_ratings), movies)
        preference_matrix[i] = user_ranking

    return preference_matrix.astype(int)

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