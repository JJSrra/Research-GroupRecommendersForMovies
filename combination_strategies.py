import numpy as np
from collections import Counter

def avg(ratings):
    # Remove 0.0 ratings
    ratings = np.array(ratings)
    ratings = ratings[ratings != 0.0]
    return 0.0 if len(ratings) == 0 else ratings.mean()

def max(ratings):
    # Remove 0.0 ratings
    ratings = np.array(ratings)
    ratings = ratings[ratings != 0.0]
    return 0.0 if len(ratings) == 0 else ratings.max()

def min(ratings):
    # Remove 0.0 ratings
    ratings = np.array(ratings)
    ratings = ratings[ratings != 0.0]
    return 0.0 if len(ratings) == 0 else ratings.min()

def maj(ratings):
    # Remove 0.0 ratings
    ratings = np.array(ratings)
    ratings = ratings[ratings != 0.0]
    return 0.0 if len(ratings) == 0 else Counter(ratings).most_common()[0][0]
