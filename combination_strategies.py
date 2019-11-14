import numpy as np
from collections import Counter

def avg(ratings):
    return np.array(ratings).mean()

def max(ratings):
    return np.array(ratings).max()

def min(ratings):
    return np.array(ratings).min()

def maj(ratings):
    return Counter(ratings).most_common()[0][0]