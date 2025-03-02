import numpy as np
import pandas as pd
np.set_printoptions(precision=2)

users = pd.read_csv('d2/user.csv')
items = pd.read_csv('d2/item.csv')
ratings = pd.read_csv('d2/rating.csv')

utility_matrix = ratings.pivot(index="item_id", columns="user_id", values="rating").values
print(utility_matrix)

means = np.nanmean(utility_matrix, axis=0)
print(means)

for i in range(utility_matrix.shape[0]):
    for j in range(utility_matrix.shape[1]):
        if np.isnan(utility_matrix[i][j]):
            utility_matrix[i][j] = 0
        else:
            utility_matrix[i][j] -= means[j]


print(utility_matrix)

from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(utility_matrix)

# In kết quả
print("Ma trận tương đồng Cosine giữa các user:")
print(similarity_matrix)