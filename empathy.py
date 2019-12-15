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

    chosen_movie_index = np.argsort(-final_ratings)[0]
    chosen_movie = movies[chosen_movie_index]
    movies = np.delete(movies, chosen_movie_index)

    # Update weights
    happiness_order = np.argsort(-ratings_matrix[:,chosen_movie_index])

    # Get users-1 fractions of 20 (from -10% to +10%), so users can be assigned to one of the edges and
    # obtain their weight update value. I.e: 5 users. 20/4 = 5. -0.1 ... -0.05 ... 0.0 ... 0.05 ... 0.1
    # The user that liked the movie the most will reduce their weight in 0.1, the second one that liked
    # the movie the most will reduce their weight in 0.05... And viceversa for the users that disliked it

    update_weight_range = 0.2
    weight_step = update_weight_range / (len(group) - 1)

    for i in range(0, len(weights)):
        weights[happiness_order[i]] += -(update_weight_range/2) + i*weight_step

    weights = weights.clip(0,1)

    ratings_matrix = np.delete(ratings_matrix, chosen_movie_index, 1)

    return chosen_movie, movies, ratings_matrix, weights
    
