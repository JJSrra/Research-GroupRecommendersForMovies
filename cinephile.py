import numpy as np
import evaluation
import baseline

def predict_individual_rating_for_movie(user, group, movie, movie_ratings, pearson_matrix):
    dim = len(pearson_matrix) # Rembember that User IDs start at 1 but we need a 0 row/column, so this is 1 more
    available_neighbors = [user for user in range(1, dim) if user not in group]

    nearest_neighbors = baseline.obtain_nearest_neighbors(user, available_neighbors, 10, pearson_matrix)
    return baseline.predict_rating(user, nearest_neighbors, movie, movie_ratings, pearson_matrix[user])

def predict_ranking_from_group(group, movies, train_movie_ratings, test_movie_ratings, pearson_matrix):
    ranking = []
    weights = get_cinephile_weights(group, train_movie_ratings)
    ratings_matrix = get_ratings_matrix(group, movies, test_movie_ratings, pearson_matrix)
    ratings_matrix = (ratings_matrix.T * weights).T

    while len(movies) > 1:
        chosen_movie, movies, ratings_matrix = choose_movie(ratings_matrix, movies, group)
        ranking.append(chosen_movie)

    ranking.append(movies[0])
    return ranking

def get_cinephile_weights(group, movie_ratings):
    movies_seen = []
    for user in group:
        movies_seen.append(0 if not movie_ratings[user].keys() else len(movie_ratings[user]))

    total_movies = np.sum(movies_seen)
    weights = np.apply_along_axis(lambda num_movies: num_movies/total_movies, 0, movies_seen)

    return weights

def get_ratings_matrix(users, movies, movie_ratings, pearson_matrix):
    ratings_matrix = np.empty([len(users), len(movies)])

    for i in range(0, len(users)):
        user_ratings = []
        for movie in movies:
            user_ratings.append(predict_individual_rating_for_movie(users[i], users, movie, movie_ratings, pearson_matrix))

        ratings_matrix[i] = user_ratings

    return ratings_matrix

def choose_movie(ratings_matrix, movies, group):
    final_ratings = np.apply_along_axis(np.sum, 0, ratings_matrix)

    chosen_movie_index = np.argsort(-final_ratings)[0]
    chosen_movie = movies[chosen_movie_index]
    movies = np.delete(movies, chosen_movie_index)

    ratings_matrix = np.delete(ratings_matrix, chosen_movie_index, 1)

    return chosen_movie, movies, ratings_matrix
    
