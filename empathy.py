import numpy as np
import evaluation

def predict_ranking_from_group(group, movies, movie_ratings):
    ratings_matrix = get_ratings_matrix(group, movies, movie_ratings)
    print(ratings_matrix)

def get_ratings_matrix(users, movies, movie_ratings):
    ratings_matrix = np.empty([len(users), len(movies)])
    
    for i in range(0, len(users)):
        user_ratings = []
        for movie in movies:
            user_ratings.append(0.0 if movie not in movie_ratings[users[i]].keys() else movie_ratings[users[i]][movie])

        ratings_matrix[i] = user_ratings

    return ratings_matrix
