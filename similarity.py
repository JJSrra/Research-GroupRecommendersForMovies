import numpy as np
import evaluation
import baseline

def predict_individual_rating_for_movie(user, group, movie, movie_ratings, pearson_matrix):
    dim = len(pearson_matrix) # Rembember that User IDs start at 1 but we need a 0 row/column, so this is 1 more
    available_neighbors = [user for user in range(1, dim) if user not in group]

    nearest_neighbors = baseline.obtain_nearest_neighbors(user, available_neighbors, 10, pearson_matrix)
    return baseline.predict_rating(user, nearest_neighbors, movie, movie_ratings, pearson_matrix[user])

def predict_ranking_from_group(group, movies, movie_ratings, pearson_matrix):
    profile_user = get_profile_user_for_group(group, pearson_matrix)
    ranking = get_ranking_from_profile_user(profile_user, group, movies, movie_ratings, pearson_matrix)
    
    return ranking

def get_profile_user_for_group(group, pearson_matrix):
    avg_similarity = []
    for user in group:
        other_users = list(filter(lambda group_user : group_user != user, group))
        avg_similarity.append(np.mean(pearson_matrix[user][other_users]))

    profile_user = np.argsort(-np.array(avg_similarity))[0]
    
    return group[profile_user]

def get_ranking_from_profile_user(profile_user, group, movies, movie_ratings, pearson_matrix):
    ratings = []

    for movie in movies:
        ratings.append(predict_individual_rating_for_movie(profile_user, group, movie, movie_ratings, pearson_matrix))

    ranking_order = np.argsort(-np.array(ratings))

    return np.array(movies)[ranking_order]
    
