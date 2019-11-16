import pandas as pd
import baseline
import combination_strategies

def evaluate_baseline(groups, ratings_by_user, pearson, output_file):
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
    ratings_dataframe['movie'] = movies_column
    ratings_dataframe['group'] = groups_column
    ratings_dataframe['avg'] = avg_column
    ratings_dataframe['min'] = min_column
    ratings_dataframe['max'] = max_column
    ratings_dataframe['maj'] = maj_column

    ratings_dataframe.to_csv(output_file, header=None, index=None)
