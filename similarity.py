import numpy as np
import evaluation

def predict_ranking_from_group(group, movies, movie_ratings, pearson):
    ranking = []
    profile_user = get_profile_user_for_group(group, movie_ratings, pearson)
    ranking = get_ranking_from_profile_user(profile_user, movies, movie_ratings)
    
    return ranking

def get_profile_user_for_group(group, movie_ratings, pearson):
    avg_similarity = []
    for user in group:
        other_users = list(filter(lambda group_user : group_user != user, group))
        avg_similarity.append(np.mean(pearson[user][other_users]))

    profile_user = np.argsort(-np.array(avg_similarity))[0]
    
    return group[profile_user]

def get_ranking_from_profile_user(profile_user, movies, movie_ratings):
    ratings = []

    for movie in movies:
        ratings.append(0.0 if movie not in movie_ratings[profile_user].keys() else movie_ratings[profile_user][movie])

    ranking_order = np.argsort(-np.array(ratings))

    return np.array(movies)[ranking_order]
    
