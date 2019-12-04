import numpy as np
import pandas as pd
import evaluation

if __name__ == "__main__":
    array = np.array
    
    random_groups = pd.read_csv("generated_data/random_groups.csv", header=None).to_numpy()
    read_dict = open('generated_data/random_evaluated_movies.txt', 'r').read()
    random_evaluated_movies = eval(read_dict)
    
    buddies_groups = pd.read_csv("generated_data/buddies_groups.csv", header=None).to_numpy()
    read_dict = open('generated_data/buddies_evaluated_movies.txt', 'r').read()
    buddies_evaluated_movies = eval(read_dict)

    read_dict = open('generated_data/test_ratings.txt', 'r').read()
    test_ratings_by_user = eval(read_dict)

    pearson = pd.read_csv("generated_data/pearson_correlation_matrix.csv", header=None).to_numpy()
    test_movies = pd.read_csv("generated_data/test_movies.csv", header=None).to_numpy().flatten()

    # ===================== BASELINE =====================

    # Random
    predicted_random_rankings = evaluation.generate_baseline_predictions(
        random_groups, random_evaluated_movies, test_ratings_by_user, pearson, "generated_data/rankings/predicted_random_")
    
    # evaluation.evaluate_predictions(
    #     predicted_random_rankings, "random", "generated_data/baseline_random_rmse.csv")

    # Buddies
    predicted_buddies_ratings = evaluation.generate_baseline_predictions(
        buddies_groups, buddies_evaluated_movies, test_ratings_by_user, pearson, "generated_data/rankings/predicted_buddies_")

    # evaluation.evaluate_predictions(
    #     predicted_buddies_ratings, "buddies", "generated_data/baseline_buddies_rmse.csv")
        

