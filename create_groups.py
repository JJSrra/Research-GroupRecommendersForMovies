import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import movielens_utils
import evaluation
import os

if __name__ == "__main__":

    # Set the seed
    np.random.seed(101)

    # Read MovieLens datasets
    ratings = pd.read_csv("dataset/ratings.csv", "|")
    movies = pd.read_csv("dataset/movies.csv", "|")
    num_users = 610 # Number of users in 100K MovieLens Dataset

    # Train/test division
    movie_ids = np.random.permutation(movies.iloc[:, 0])
    train_size = int(np.ceil(len(movie_ids) * 0.8))
    train_movies = movie_ids[:train_size]
    test_movies = movie_ids[train_size:]
    train_ratings_by_user, test_ratings_by_user = movielens_utils.load_movie_ratings(
        train_movies, test_movies, ratings, num_users)

    # Calculate Pearson Correlation Coefficient for all users (only using train movies)
    # Adding one extra initial row for user 0 which does not exist (users in range 1-610)
    pearson_matrix = np.matrix(np.zeros([num_users+1, num_users+1]))

    print("Processing Pearson Correlation Matrix...")

    for user1 in range(1, num_users+1):
        for user2 in range(user1+1, num_users+1):
            pearson_matrix[user1, user2] = movielens_utils.calculate_pearson_coefficient(
                train_ratings_by_user, user1, user2)

    # Duplicate upper triangle in lower triangle of the matrix
    low_indices = np.tril_indices(num_users+1, -1)
    pearson_matrix[low_indices] = pearson_matrix.T[low_indices]

    # Group creation
    users_per_group = 5

    # RANDOM GROUPS
    random_groups = []
    available_users = np.arange(1,611) # There are 610 users, with IDs starting at 1 and ending at 610

    while len(available_users) >= users_per_group:
        available_users = np.random.permutation(available_users)
        selected_users = available_users[:users_per_group]
        available_users = available_users[users_per_group:]

        random_groups.append(selected_users)

    # BUDDIES GROUPS
    buddies_groups = []
    available_users = np.arange(1,611)

    while len(available_users) >= users_per_group:
        available_users = np.random.permutation(available_users)
        current_user = available_users[0]

        # Sort the remaining users based on their correlation with the current user
        sorted_by_pearson = sorted(
            available_users[1:], key=lambda user: pearson_matrix[current_user, user], reverse=True)

        # We now consider buddies as users with the highest correlation indicated by the Pearson Correlation Matrix
        potential_buddies = sorted_by_pearson[:users_per_group-1]
        potential_buddies.append(current_user)

        # Form a group with the potential buddies
        buddies_groups.append(np.array(potential_buddies))

        # And remove these users from the pool
        available_users = np.setdiff1d(available_users, potential_buddies)

    # Select evaluable movies from each group
    random_evaluated_movies = {}

    for i in range(0, len(random_groups)):
        evaluated_movies = movielens_utils.movies_that_at_least_3_have_seen(random_groups[i], test_movies, test_ratings_by_user)
        if len(evaluated_movies) > 0:
            random_evaluated_movies[i] = evaluated_movies 

    os.makedirs("generated_data", exist_ok=True)

    f = open("generated_data/random_evaluated_movies.txt", "w")
    f.write(str(random_evaluated_movies))
    f.close()

    buddies_evaluated_movies = {}

    for i in range(0, len(buddies_groups)):
        evaluated_movies = movielens_utils.movies_that_at_least_3_have_seen(buddies_groups[i], test_movies, test_ratings_by_user)
        if len(evaluated_movies) > 0:
            buddies_evaluated_movies[i] = evaluated_movies

    f = open("generated_data/buddies_evaluated_movies.txt", "w")
    f.write(str(buddies_evaluated_movies))
    f.close()

    # Generate real ratings given by the groups for evaluation purposes
    os.makedirs("generated_data/rankings", exist_ok=True)
    evaluation.generate_real_ratings(random_groups, random_evaluated_movies, test_ratings_by_user, "generated_data/rankings/real_random_")
    evaluation.generate_real_ratings(buddies_groups, buddies_evaluated_movies, test_ratings_by_user, "generated_data/rankings/real_buddies_")

    # File saving
    pd.DataFrame(random_groups).to_csv(
        "generated_data/random_groups.csv", header=None, index=None)

    pd.DataFrame(buddies_groups).to_csv(
        "generated_data/buddies_groups.csv", header=None, index=None)

    # Save train and test movies
    pd.DataFrame(train_movies).to_csv(
        "generated_data/train_movies.csv", header=None, index=None)
    pd.DataFrame(test_movies).to_csv(
        "generated_data/test_movies.csv", header=None, index=None)

    # Save train and test ratings by user
    f = open("generated_data/train_ratings.txt", "w")
    f.write(str(train_ratings_by_user))
    f.close()

    f = open("generated_data/test_ratings.txt", "w")
    f.write(str(test_ratings_by_user))
    f.close()

    # Saving Pearson Correlation Matrix
    pd.DataFrame(pearson_matrix).to_csv(
        "generated_data/pearson_correlation_matrix.csv", header = None, index = None)
