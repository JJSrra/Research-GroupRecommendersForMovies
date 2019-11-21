import pandas as pd
import numpy as np
import baseline
import combination_strategies

def generate_real_ratings(groups, ratings_by_user, output_file):
    movies_column = []
    groups_column = []
    avg_column = []
    min_column = []
    max_column = []
    maj_column = []

    for movie in groups.keys():
        for group in groups[movie]:
            group_ratings = []
            for user in group:
                group_ratings.append(ratings_by_user[user][movie])

            movies_column.append(movie)
            groups_column.append(group)
            avg_column.append(combination_strategies.avg(group_ratings))
            min_column.append(combination_strategies.min(group_ratings))
            max_column.append(combination_strategies.max(group_ratings))
            maj_column.append(combination_strategies.maj(group_ratings))

    ratings_dataframe = pd.DataFrame()
    ratings_dataframe[0] = movies_column
    ratings_dataframe[1] = groups_column
    ratings_dataframe[2] = avg_column
    ratings_dataframe[3] = min_column
    ratings_dataframe[4] = max_column
    ratings_dataframe[5] = maj_column

    ratings_dataframe.to_csv(output_file, header=None, index=None)

def generate_baseline_predictions(groups, ratings_by_user, pearson, output_file):
    movies_column = []
    groups_column = []
    avg_column = []
    min_column = []
    max_column = []
    maj_column = []

    for movie in groups.keys():
        for group in groups[movie]:
            group_ratings = baseline.predict_group_individual_ratings_for_movie(
                group, movie, ratings_by_user, pearson)

            movies_column.append(movie)
            groups_column.append(group)
            avg_column.append(combination_strategies.avg(group_ratings))
            min_column.append(combination_strategies.min(group_ratings))
            max_column.append(combination_strategies.max(group_ratings))
            maj_column.append(combination_strategies.maj(group_ratings))

    ratings_dataframe = pd.DataFrame()
    ratings_dataframe[0] = movies_column
    ratings_dataframe[1] = groups_column
    ratings_dataframe[2] = avg_column
    ratings_dataframe[3] = min_column
    ratings_dataframe[4] = max_column
    ratings_dataframe[5] = maj_column

    ratings_dataframe.to_csv(output_file, header=None, index=None)
    return ratings_dataframe

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
