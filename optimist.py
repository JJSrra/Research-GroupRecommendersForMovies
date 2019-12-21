import numpy as np
import evaluation
import baseline

def predict_individual_rating_for_movie(user, group, movie, movie_ratings, pearson_matrix):
    dim = len(pearson_matrix) # Rembember that User IDs start at 1 but we need a 0 row/column, so this is 1 more
    available_neighbors = [user for user in range(1, dim) if user not in group]

    nearest_neighbors = baseline.obtain_nearest_neighbors(user, available_neighbors, 10, pearson_matrix)
    return baseline.predict_rating(user, nearest_neighbors, movie, movie_ratings, pearson_matrix[user])

def predict_ranking_from_group(group, movies, movie_ratings, pearson_matrix):
    ranking = []
    weights = get_optimist_weights(group, movie_ratings)
    ratings_matrix = get_ratings_matrix(group, movies, movie_ratings, pearson_matrix)
    ratings_matrix = (ratings_matrix.T * weights).T

    while len(movies) > 1:
        chosen_movie, movies, ratings_matrix = choose_movie(
            ratings_matrix, movies, group, movie_ratings)

        ranking.append(chosen_movie)

    ranking.append(movies[0])
    return ranking

def get_optimist_weights(group, movie_ratings):
    avg_ratings = []
    for user in group:
        avg_ratings.append(0.0 if not movie_ratings[user].keys() else sum(movie_ratings[user].values()) / len(movie_ratings[user].keys()))

    total_avg = np.sum(avg_ratings)
    weights = np.apply_along_axis(lambda rating: rating/total_avg, 0, avg_ratings)
    return weights

def get_ratings_matrix(users, movies, movie_ratings, pearson_matrix):
    ratings_matrix = np.empty([len(users), len(movies)])

    for i in range(0, len(users)):
        user_ratings = []
        for movie in movies:
            user_ratings.append(predict_individual_rating_for_movie(users[i], users, movie, movie_ratings, pearson_matrix))

        ratings_matrix[i] = user_ratings

    return ratings_matrix

def choose_movie(ratings_matrix, movies, group, movie_ratings):
    final_ratings = np.apply_along_axis(np.sum, 0, ratings_matrix)

    chosen_movie_index = np.argsort(-final_ratings)[0]
    chosen_movie = movies[chosen_movie_index]
    movies = np.delete(movies, chosen_movie_index)

    ratings_matrix = np.delete(ratings_matrix, chosen_movie_index, 1)

    return chosen_movie, movies, ratings_matrix
    
