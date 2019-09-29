def load_movie_ratings(train_movies, test_movies, ratings):
        
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

    return train_movie_ratings, test_movie_ratings