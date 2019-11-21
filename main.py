import numpy as np
import pandas as pd
import evaluation

if __name__ == "__main__":
    array = np.array
    read_dict = open('generated_data/random_groups.txt', 'r').read()
    random_groups = eval(read_dict)

    real_random_ratings = pd.read_csv("generated_data/real_random_ratings.csv", header=None)
    
    read_dict = open('generated_data/buddies_groups.txt', 'r').read()
    buddies_groups = eval(read_dict)

    real_buddies_ratings = pd.read_csv("generated_data/real_buddies_ratings.csv", header=None)

    read_dict = open('generated_data/circumstantial_groups.txt', 'r').read()
    circumstantial_groups = eval(read_dict)

    real_circumstantial_ratings = pd.read_csv("generated_data/real_circumstantial_ratings.csv", header=None)
    
    read_dict = open('generated_data/test_ratings.txt', 'r').read()
    test_ratings_by_user = eval(read_dict)

    pearson = pd.read_csv("generated_data/pearson_correlation_matrix.csv", header=None).to_numpy()
    test_movies = pd.read_csv("generated_data/test_movies.csv", header=None).to_numpy().flatten()

    # ===================== BASELINE =====================

    # Random
    predicted_random_ratings = evaluation.generate_baseline_predictions(
        random_groups, test_ratings_by_user, pearson, "generated_data/baseline_random_ratings.csv")
    
    evaluation.evaluate_predictions(
        predicted_random_ratings, real_random_ratings, "generated_data/baseline_random_rmse.csv")

    # Buddies
    predicted_buddies_ratings = evaluation.generate_baseline_predictions(
        buddies_groups, test_ratings_by_user, pearson, "generated_data/baseline_buddies_ratings.csv")

    evaluation.evaluate_predictions(
        predicted_buddies_ratings, real_buddies_ratings, "generated_data/baseline_buddies_rmse.csv")

    # Circumstantial
    predicted_circumstantial_ratings = evaluation.generate_baseline_predictions(
        circumstantial_groups, test_ratings_by_user, pearson, "generated_data/baseline_circumstantial_ratings.csv")

    evaluation.evaluate_predictions(
        predicted_circumstantial_ratings, real_circumstantial_ratings, "generated_data/baseline_circumstantial_rmse.csv")
        

