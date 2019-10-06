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
    train_ratings_by_user, test_ratings_by_user = movielens_utils.load_movie_ratings(
        train_movies, test_movies, ratings)
    num_users = len(train_ratings_by_user)

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
    random_groups = {}

    for movie in test_movies:
        random_groups[movie] = []
        available_users = movielens_utils.users_who_have_seen(movie, test_ratings_by_user)

        while len(available_users) >= users_per_group:
            available_users = np.random.permutation(available_users)
            selected_users = available_users[:users_per_group]
            available_users = available_users[users_per_group:]

            random_groups[movie].append(selected_users)

    f = open("random_groups.txt", "w")
    f.write(str(random_groups))
    f.close()

    # BUDDIES GROUPS
    buddies_groups = {}

    for movie in test_movies:
        buddies_groups[movie] = []
        available_users = movielens_utils.users_who_have_seen(
            movie, test_ratings_by_user)

        while len(available_users) >= users_per_group:
            current_user = available_users[0]
            available_users = np.delete(available_users, 0)
            selected_users = []

            # Sort the remaining users based on their correlation with the current user
            sorted_by_pearson = sorted(
                available_users, key=lambda user: pearson_matrix[current_user, user], reverse=True)

            # If a user has seen at least 1 movie in common with the current user, it is considered a buddy
            # (considering the high correlation indicated by the Pearson Correlation Matrix)
            for user in sorted_by_pearson:
                if movielens_utils.have_seen_X_common_movies(1, current_user, user, train_ratings_by_user):
                    selected_users.append(user)

            # While there are enough users, form a group with the current user
            while len(selected_users) >= (users_per_group-1):
                new_group = [current_user]
                new_group.extend(selected_users[:users_per_group-1])
                buddies_groups[movie].append(np.array(new_group))
                selected_users = selected_users[users_per_group-1:]

    f = open("buddies_groups.txt", "w")
    f.write(str(buddies_groups))
    f.close()
