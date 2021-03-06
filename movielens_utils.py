import math
import numpy as np

def load_movie_ratings(train_movies, test_movies, ratings, num_users):
        
    train_movie_ratings = {}
    test_movie_ratings = {}

    for index, row in ratings.iterrows():
        movie_id = int(row['movieId'])
        user_id = int(row['userId'])
        rating = float(row['rating'])

        if movie_id in train_movies:
            train_movie_ratings.setdefault(user_id, {})
            train_movie_ratings[user_id][movie_id] = rating
        elif movie_id in test_movies:
            test_movie_ratings.setdefault(user_id, {})
            test_movie_ratings[user_id][movie_id] = rating

    for user in range(1,num_users+1): # Users in MovieLens Database start at 1 so upper bound must be set at max_index + 1
        if user not in train_movie_ratings:
            train_movie_ratings[user] = {}
        if user not in test_movie_ratings:
            test_movie_ratings[user] = {}

    return train_movie_ratings, test_movie_ratings


def calculate_pearson_coefficient(movie_ratings, user1, user2):

    common_movies = {}

    for movie in movie_ratings[user1]:
        if movie in movie_ratings[user2]:
            common_movies[movie] = 1
            
    num_common_movies = len(common_movies)

    if num_common_movies == 0:
        return 0

    sum1 = sum([movie_ratings[user1][movie] for movie in common_movies])
    sum2 = sum([movie_ratings[user2][movie] for movie in common_movies])

    sum1Sq = sum([pow(movie_ratings[user1][movie], 2) for movie in common_movies])
    sum2Sq = sum([pow(movie_ratings[user2][movie], 2) for movie in common_movies])

    pSum = sum([movie_ratings[user1][movie]*movie_ratings[user2][movie]
                for movie in common_movies])

    # Calculate Pearson score
    num = pSum-(sum1*sum2/num_common_movies)
    den = math.sqrt((sum1Sq-pow(sum1, 2)/num_common_movies)
                    * (sum2Sq-pow(sum2, 2)/num_common_movies))
    if den == 0:
        return 0

    r = num/den
    return r

def users_who_have_seen(movie, ratings_by_user):
    
    user_pool = []

    for user in ratings_by_user:
        if movie in ratings_by_user[user]:
            user_pool.append(user)

    return np.array(user_pool)

def have_seen_X_common_movies(num_movies, user1, user2, ratings_by_user):
    
    user1_movies = ratings_by_user[user1].keys()
    user2_movies = ratings_by_user[user2].keys()

    return len(user1_movies & user2_movies) >= num_movies

def movies_that_at_least_X_have_seen(num_movies, group, movies, ratings_by_user):

    result_movies = []

    for movie in movies:
        movie_seen_times = 0
        for user in group:
            if movie in ratings_by_user[user].keys():
                movie_seen_times += 1

        if movie_seen_times >= num_movies:
            result_movies.append(movie)

    return result_movies