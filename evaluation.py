import pandas as pd
import numpy as np
import baseline
import combination_strategies

def sort_movies_by_ranking(ratings, movies):
    order = np.argsort(np.array(ratings))
    ranking = np.array(movies)[order]
    return ranking

def generate_real_ratings(groups, movies_by_group, ratings_by_user, output_file):
    
    avg_rankings = {}
    min_rankings = {}
    max_rankings = {}
    maj_rankings = {}

    for i in range(0, len(groups)):
        avg_values = np.array([])
        min_values = np.array([])
        max_values = np.array([])
        maj_values = np.array([])
        group_ratings = []

        if i in movies_by_group.keys(): # If there is at least 1 movie that the group have seen in common
            for movie in movies_by_group[i]:
                print("Group {}, movie {}".format(i, movie))
                for user in groups[i]:
                    group_ratings.append(0.0 if movie not in ratings_by_user[user] else ratings_by_user[user][movie])

                avg_values = np.append(avg_values, combination_strategies.avg(group_ratings))
                min_values = np.append(min_values, combination_strategies.min(group_ratings))
                max_values = np.append(max_values, combination_strategies.max(group_ratings))
                maj_values = np.append(maj_values, combination_strategies.maj(group_ratings))

            avg_rankings[i] = sort_movies_by_ranking(avg_values, movies_by_group[i])
            min_rankings[i] = sort_movies_by_ranking(min_values, movies_by_group[i])
            max_rankings[i] = sort_movies_by_ranking(max_values, movies_by_group[i])
            maj_rankings[i] = sort_movies_by_ranking(maj_values, movies_by_group[i])

    f = open(output_file + "avg.txt", "w")
    f.write(str(avg_rankings))
    f.close()

    f = open(output_file + "min.txt", "w")
    f.write(str(min_rankings))
    f.close()

    f = open(output_file + "max.txt", "w")
    f.write(str(max_rankings))
    f.close()

    f = open(output_file + "maj.txt", "w")
    f.write(str(maj_rankings))
    f.close()


def generate_baseline_predictions(movies, groups, ratings_by_user, pearson, output_file):
    group_ratings = []
    movie_ratings = {}

    for group in groups:
        for movie in movies:
            individual_group_ratings = baseline.predict_group_individual_ratings_for_movie(
                group, movie, ratings_by_user, pearson)

            movie_ratings[movie] = individual_group_ratings

        group_ratings.append(movie_ratings)

    return group_ratings
    

def evaluate_predictions(predicted_dataframe, real_dataframe, output_file):
    predicted_avg = predicted_dataframe[2].to_numpy()
    predicted_min = predicted_dataframe[3].to_numpy()
    predicted_max = predicted_dataframe[4].to_numpy()
    predicted_maj = predicted_dataframe[5].to_numpy()
    real_avg = real_dataframe[2].to_numpy()
    real_min = real_dataframe[3].to_numpy()
    real_max = real_dataframe[4].to_numpy()
    real_maj = real_dataframe[5].to_numpy()

    rmse_avg = rmse(predicted_avg, real_avg)
    rmse_min = rmse(predicted_min, real_min)
    rmse_max = rmse(predicted_max, real_max)
    rmse_maj = rmse(predicted_maj, real_maj)

    f = open(output_file, "w")
    f.write("AVG: {}\n".format(rmse_avg))
    f.write("MIN: {}\n".format(rmse_min))
    f.write("MAX: {}\n".format(rmse_max))
    f.write("MAJ: {}\n".format(rmse_maj))
    f.close()

def rmse(predicted, real):
    return np.sqrt(((predicted - real) ** 2).mean())
