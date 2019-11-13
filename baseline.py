import numpy as np
import pandas as pd

def predict_group_individual_ratings_for_movie(group, movie, movie_ratings, pearson_matrix):
    
    # Specify the K in K-nearest neighbors for each member of the group, according to Pearson
    k_nearest = 10
    dim = len(pearson_matrix) # Rembember that User IDs start at 1 but we need a 0 row/column, so this is 1 more
    group_ratings = []

    # Obtain all available users (those who are not members of the group)
    available_users = [user for user in range(1,dim) if user not in group]

    # For each user in the group, calculate their K-nearest neighbors and predict their rating 
    for group_user in group:
        nearest_neighbors = sorted(
            available_users, key=lambda user: pearson_matrix[group_user, user], reverse=True)[:k_nearest]
        
        group_ratings.append(predict_rating(group_user, nearest_neighbors, movie, movie_ratings, pearson_matrix[group_user]))

    return group_ratings


def predict_rating(user, neighbors, movie, movie_ratings, user_correlation):
    neighbor_accumulated = 0
    normalizer = 0
    
    for neighbor in neighbors:
        neighbor_rating = movie_ratings[neighbor][movie] if movie in movie_ratings[neighbor] else 0
        neighbor_accumulated += neighbor_rating * user_correlation[neighbor]
        normalizer += abs(user_correlation[neighbor])

    return neighbor_accumulated / normalizer
