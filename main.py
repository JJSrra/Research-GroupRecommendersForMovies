import numpy as np
import pandas as pd
import evaluation

if __name__ == "__main__":
    array = np.array
    read_dict = open('generated_data/random_groups.txt', 'r').read()
    random_groups = eval(read_dict)
    
    read_dict = open('generated_data/buddies_groups.txt', 'r').read()
    buddies_groups = eval(read_dict)

    read_dict = open('generated_data/circumstantial_groups.txt', 'r').read()
    circumstantial_groups = eval(read_dict)
    
    read_dict = open('generated_data/test_ratings.txt', 'r').read()
    test_ratings_by_user = eval(read_dict)

    pearson = pd.read_csv("generated_data/pearson_correlation_matrix.csv", header=None).to_numpy()
    test_movies = pd.read_csv("generated_data/test_movies.csv", header=None).to_numpy().flatten()

    # Random
    evaluation.evaluate_baseline(random_groups, test_ratings_by_user, pearson, "generated_data/baseline_random_ratings.csv")

    # Buddies
    evaluation.evaluate_baseline(buddies_groups, test_ratings_by_user, pearson, "generated_data/baseline_buddies_ratings.csv")

    # Circumstantial
    evaluation.evaluate_baseline(circumstantial_groups, test_ratings_by_user, pearson, "generated_data/baseline_circumstantial_ratings.csv")
        

