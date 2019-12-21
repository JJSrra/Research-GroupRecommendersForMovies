import pandas as pd
import numpy as np
import combination_strategies
import baseline
import pogrs
import empathy
import cinephile
import optimist
import similarity
from sklearn import metrics

def sort_movies_by_ranking(ratings, movies):
    order = np.argsort(np.array((-ratings))) # Negated for descending order
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


def generate_baseline_predictions(groups, movies_by_group, ratings_by_user, pearson, output_file):
    
    avg_rankings = {}
    min_rankings = {}
    max_rankings = {}
    maj_rankings = {}

    for i in range(0, len(groups)):
        avg_values = np.array([])
        min_values = np.array([])
        max_values = np.array([])
        maj_values = np.array([])

        if i in movies_by_group.keys():  # If there is at least 1 movie that the group have seen in common
            for movie in movies_by_group[i]:
                individual_group_ratings = baseline.predict_group_individual_ratings_for_movie(
                    groups[i], movie, ratings_by_user, pearson)

                avg_values = np.append(avg_values, combination_strategies.avg(individual_group_ratings))
                min_values = np.append(min_values, combination_strategies.min(individual_group_ratings))
                max_values = np.append(max_values, combination_strategies.max(individual_group_ratings))
                maj_values = np.append(maj_values, combination_strategies.maj(individual_group_ratings))

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

    return {"avg": avg_rankings, "min": min_rankings, "max": max_rankings, "maj": maj_rankings}

def generate_pogrs_predictions(groups, movies_by_group, ratings_by_user, pearson, output_file):

    pogrs_rankings = {}
    for i in range(0, len(groups)):
        if i in movies_by_group.keys(): # If there is at least 1 movie that the group have seen in common
            pogrs_rankings[i] = pogrs.predict_ranking_from_group(groups[i], movies_by_group[i], ratings_by_user, pearson)

    f = open(output_file, "w")
    f.write(str(pogrs_rankings))
    f.close()

    return {"avg": pogrs_rankings, "min": pogrs_rankings, "max": pogrs_rankings, "maj": pogrs_rankings}

def generate_empathy_predictions(groups, movies_by_group, ratings_by_user, pearson, output_file):

    empathy_rankings = {}
    for i in range(0, len(groups)):
        if i in movies_by_group.keys(): # If there is at least 1 movie that the group have seen in common
            empathy_rankings[i] = empathy.predict_ranking_from_group(groups[i], movies_by_group[i], ratings_by_user)

    f = open(output_file, "w")
    f.write(str(empathy_rankings))
    f.close()

    return {"avg": empathy_rankings, "min": empathy_rankings, "max": empathy_rankings, "maj": empathy_rankings}

def generate_cinephile_predictions(groups, movies_by_group, ratings_by_user, pearson, output_file):

    cinephile_rankings = {}
    for i in range(0, len(groups)):
        if i in movies_by_group.keys(): # If there is at least 1 movie that the group have seen in common
            cinephile_rankings[i] = cinephile.predict_ranking_from_group(groups[i], movies_by_group[i], ratings_by_user)

    f = open(output_file, "w")
    f.write(str(cinephile_rankings))
    f.close()

    return {"avg": cinephile_rankings, "min": cinephile_rankings, "max": cinephile_rankings, "maj": cinephile_rankings}

def generate_optimist_predictions(groups, movies_by_group, ratings_by_user, pearson, output_file):

    optimist_rankings = {}
    for i in range(0, len(groups)):
        if i in movies_by_group.keys(): # If there is at least 1 movie that the group have seen in common
            optimist_rankings[i] = optimist.predict_ranking_from_group(groups[i], movies_by_group[i], ratings_by_user)

    f = open(output_file, "w")
    f.write(str(optimist_rankings))
    f.close()

    return {"avg": optimist_rankings, "min": optimist_rankings, "max": optimist_rankings, "maj": optimist_rankings}

def generate_similarity_predictions(groups, movies_by_group, ratings_by_user, pearson, output_file):

    similarity_rankings = {}
    for i in range(0, len(groups)):
        if i in movies_by_group.keys(): # If there is at least 1 movie that the group have seen in common
            similarity_rankings[i] = similarity.predict_ranking_from_group(groups[i], movies_by_group[i], ratings_by_user, pearson)

    f = open(output_file, "w")
    f.write(str(similarity_rankings))
    f.close()

    return {"avg": similarity_rankings, "min": similarity_rankings, "max": similarity_rankings, "maj": similarity_rankings}

def evaluate_predictions(predicted_rankings, real_rankings, output_file):
    ndcg_avg = calculate_mean_ndcg(real_rankings["avg"], predicted_rankings["avg"], 122)
    ndcg_min = calculate_mean_ndcg(real_rankings["min"], predicted_rankings["min"], 122)
    ndcg_max = calculate_mean_ndcg(real_rankings["max"], predicted_rankings["max"], 122)
    ndcg_maj = calculate_mean_ndcg(real_rankings["maj"], predicted_rankings["maj"], 122)

    f = open(output_file + "ndcg.csv", "w")
    f.write("nDCG Avg: {}\n".format(ndcg_avg))
    f.write("nDCG Min: {}\n".format(ndcg_min))
    f.write("nDCG Max: {}\n".format(ndcg_max))
    f.write("nDCG Maj: {}\n".format(ndcg_maj))
    f.close()

def calculate_mean_ndcg(real_rankings, predicted_rankings, max_groups):
    ndcg_results = []
    for i in range(0, max_groups):
        if i in real_rankings.keys():
            result = 1 if len(real_rankings[i]) == 1 else metrics.ndcg_score(
                np.array([real_rankings[i]]), np.array([predicted_rankings[i]]))
            ndcg_results.append(result)

    return np.mean(ndcg_results)