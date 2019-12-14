import numpy as np
import evaluation

def predict_ranking_from_group(group, movies, movie_ratings):
    ranking = []
    weights = np.repeat(1/len(group), len(group))
    ratings_matrix = get_ratings_matrix(group, movies, movie_ratings)

    chosen_movie, movies, ratings_matrix, weights = choose_movie_and_update_weights(
        ratings_matrix, movies, group, weights, movie_ratings)

    ranking.append(chosen_movie)

def get_ratings_matrix(users, movies, movie_ratings):
    ratings_matrix = np.empty([len(users), len(movies)])

    for i in range(0, len(users)):
        user_ratings = []
        for movie in movies:
            user_ratings.append(0.0 if movie not in movie_ratings[users[i]].keys() else movie_ratings[users[i]][movie])

        ratings_matrix[i] = user_ratings

    return ratings_matrix


def choose_movie_and_update_weights(ratings_matrix, movies, group, weights, movie_ratings):
    weighted_matrix = (ratings_matrix.T * weights).T
    final_ratings = np.apply_along_axis(np.sum, 0, weighted_matrix)

    chosen_movie_index = np.argsort(-final_ratings)
    chosen_movie = movies[chosen_movie_index]
    movies = np.delete(movies, chosen_movie_index)

    ratings_matrix = np.delete(ratings_matrix, chosen_movie_index, 1)

    # Update weights

    return chosen_movie, movies, ratings_matrix, weights
    
