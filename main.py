import numpy as np
import pandas as pd
import evaluation
import similarity

if __name__ == "__main__":
    array = np.array
    
    random_groups = pd.read_csv("generated_data/random_groups.csv", header=None).to_numpy()
    read_dict = open('generated_data/random_evaluated_movies.txt', 'r').read()
    random_evaluated_movies = eval(read_dict)
    
    avg_real_rankings = eval(open('generated_data/rankings/real_random_avg.txt', 'r').read())
    min_real_rankings = eval(open('generated_data/rankings/real_random_min.txt', 'r').read())
    max_real_rankings = eval(open('generated_data/rankings/real_random_max.txt', 'r').read())
    maj_real_rankings = eval(open('generated_data/rankings/real_random_maj.txt', 'r').read())
    real_random_rankings = {"avg": avg_real_rankings, "min": min_real_rankings, "max": max_real_rankings, "maj": maj_real_rankings}

    buddies_groups = pd.read_csv("generated_data/buddies_groups.csv", header=None).to_numpy()
    read_dict = open('generated_data/buddies_evaluated_movies.txt', 'r').read()
    buddies_evaluated_movies = eval(read_dict)

    avg_real_rankings = eval(open('generated_data/rankings/real_buddies_avg.txt', 'r').read())
    min_real_rankings = eval(open('generated_data/rankings/real_buddies_min.txt', 'r').read())
    max_real_rankings = eval(open('generated_data/rankings/real_buddies_max.txt', 'r').read())
    maj_real_rankings = eval(open('generated_data/rankings/real_buddies_maj.txt', 'r').read())
    real_buddies_rankings = {"avg": avg_real_rankings, "min": min_real_rankings, "max": max_real_rankings, "maj": maj_real_rankings}

    read_dict = open('generated_data/test_ratings.txt', 'r').read()
    test_ratings_by_user = eval(read_dict)

    pearson = pd.read_csv("generated_data/pearson_correlation_matrix.csv", header=None).to_numpy()
    test_movies = pd.read_csv("generated_data/test_movies.csv", header=None).to_numpy().flatten()

    # ===================== BASELINE =====================

    # # Random
    # predicted_random_baseline_rankings = evaluation.generate_baseline_predictions(
    #     random_groups, random_evaluated_movies, test_ratings_by_user, pearson, "generated_data/rankings/predicted_random_")
    
    # evaluation.evaluate_predictions(
    #     predicted_random_baseline_rankings, real_random_rankings, "generated_data/baseline_random_ndcg.csv")

    # # Buddies
    # predicted_buddies_baseline_rankings = evaluation.generate_baseline_predictions(
    #     buddies_groups, buddies_evaluated_movies, test_ratings_by_user, pearson, "generated_data/rankings/predicted_buddies_")

    # evaluation.evaluate_predictions(
    #     predicted_buddies_baseline_rankings, real_buddies_rankings, "generated_data/baseline_buddies_ndcg.csv")

    # # ===================== POGRS =====================

    # # Random
    # predicted_random_pogrs_rankings = evaluation.generate_pogrs_predictions(
    #     random_groups, random_evaluated_movies, test_ratings_by_user, "generated_data/rankings/predicted_random_pogrs.txt")

    # evaluation.evaluate_predictions(
    #     predicted_random_pogrs_rankings, real_random_rankings, "generated_data/pogrs_random_ndcg.csv")

    # # Buddies
    # predicted_buddies_pogrs_rankings = evaluation.generate_pogrs_predictions(
    #     buddies_groups, buddies_evaluated_movies, test_ratings_by_user, "generated_data/rankings/predicted_buddies_pogrs.txt")

    # evaluation.evaluate_predictions(
    #     predicted_buddies_pogrs_rankings, real_buddies_rankings, "generated_data/pogrs_buddies_ndcg.csv")

    # ===================== EMPATHY =====================
    
    # Random
    # predicted_random_empathy_rankings = evaluation.generate_empathy_predictions(
    #     random_groups, random_evaluated_movies, test_ratings_by_user, "generated_data/rankings/predicted_random_empathy.txt")

    # evaluation.evaluate_predictions(
    #     predicted_random_empathy_rankings, real_random_rankings, "generated_data/empathy_random_ndcg.csv")

    # # Buddies
    # predicted_buddies_empathy_rankings = evaluation.generate_empathy_predictions(
    #     buddies_groups, buddies_evaluated_movies, test_ratings_by_user, "generated_data/rankings/predicted_buddies_empathy.txt")

    # evaluation.evaluate_predictions(
    #     predicted_buddies_empathy_rankings, real_buddies_rankings, "generated_data/empathy_buddies_ndcg.csv")

    # ===================== CINEPHILE =====================

    # Random
    # predicted_random_cinephile_rankings = evaluation.generate_cinephile_predictions(
    #     random_groups, random_evaluated_movies, test_ratings_by_user, "generated_data/rankings/predicted_random_cinephile.txt")

    # evaluation.evaluate_predictions(
    #     predicted_random_cinephile_rankings, real_random_rankings, "generated_data/cinephile_random_ndcg.csv")

    # # Buddies
    # predicted_buddies_cinephile_rankings = evaluation.generate_cinephile_predictions(
    #     buddies_groups, buddies_evaluated_movies, test_ratings_by_user, "generated_data/rankings/predicted_buddies_cinephile.txt")

    # evaluation.evaluate_predictions(
    #     predicted_buddies_cinephile_rankings, real_buddies_rankings, "generated_data/cinephile_buddies_ndcg.csv")

    # ===================== OPTIMIST =====================

    # Random
    # predicted_random_optimist_rankings = evaluation.generate_optimist_predictions(
    #     random_groups, random_evaluated_movies, test_ratings_by_user, "generated_data/rankings/predicted_random_optimist.txt")

    # evaluation.evaluate_predictions(
    #     predicted_random_optimist_rankings, real_random_rankings, "generated_data/optimist_random_ndcg.csv")

    # # Buddies
    # predicted_buddies_optimist_rankings = evaluation.generate_optimist_predictions(
    #     buddies_groups, buddies_evaluated_movies, test_ratings_by_user, "generated_data/rankings/predicted_buddies_optimist.txt")

    # evaluation.evaluate_predictions(
    #     predicted_buddies_optimist_rankings, real_buddies_rankings, "generated_data/optimist_buddies_ndcg.csv")

    # ===================== SIMILARITY =====================
    
    # Random
    predicted_random_similarity_rankings = evaluation.generate_similarity_predictions(
        random_groups, random_evaluated_movies, test_ratings_by_user, pearson, "generated_data/rankings/predicted_random_similarity.txt")

    evaluation.evaluate_predictions(
        predicted_random_similarity_rankings, real_random_rankings, "generated_data/similarity_random_ndcg.csv")

    # Buddies
    predicted_buddies_similarity_rankings = evaluation.generate_similarity_predictions(
        buddies_groups, buddies_evaluated_movies, test_ratings_by_user, pearson, "generated_data/rankings/predicted_buddies_similarity.txt")

    evaluation.evaluate_predictions(
        predicted_buddies_similarity_rankings, real_buddies_rankings, "generated_data/similarity_buddies_ndcg.csv")
