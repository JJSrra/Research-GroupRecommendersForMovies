import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import movielens_utils

if __name__ == "__main__":

    # Set the seed
    np.random.seed(101)

    # Read MovieLens datasets
    ratings = pd.read_csv("dataset/ratings.csv", "|")
    movies = pd.read_csv("dataset/movies.csv", "|")

    # Train/test division
    movie_ids = np.random.permutation(movies.iloc[:, 0])
    train_size = int(np.ceil(len(movie_ids) * 0.8))
    train_movies = movie_ids[:train_size]
    test_movies = movie_ids[train_size:]
    train_ratings_by_user, test_ratings_by_user = movielens_utils.load_movie_ratings(train_movies, test_movies, ratings)

    # # RANDOM GROUPS
    # users_per_random_group = 5
    
    # remainder_users = len(train_users) % users_per_random_group
    # if remainder_users != 0:
    #     train_users = train_users[:-remainder_users]

    # num_random_groups = int(len(train_users) / users_per_random_group)
    # random_groups = np.random.permutation(train_users).reshape(num_random_groups, users_per_random_group)