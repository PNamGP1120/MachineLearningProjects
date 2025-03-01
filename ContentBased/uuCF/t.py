import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class UserUserCF:
    """
    User-User Collaborative Filtering (CF) for rating prediction.
    
    Ý tưởng:
    - Dự đoán đánh giá của người dùng đối với một mặt hàng (item) dựa trên những người dùng có sở thích tương tự.
    - Sử dụng **cosine similarity** để đo độ tương đồng giữa các người dùng.
    - Dữ liệu được chuẩn hóa để giảm thiên vị giữa các người dùng có thang điểm đánh giá khác nhau.

    Dữ liệu đầu vào:
    - `Y_data`: Một numpy array có dạng (n_samples, 3) với mỗi dòng chứa `[user_id, item_id, rating]`.
    - `k`: Số lượng hàng xóm gần nhất (nearest neighbors) được sử dụng để dự đoán.

    Giải thuật:
    1. Tính **độ tương đồng** giữa các người dùng sử dụng cosine similarity.
    2. Chuẩn hóa dữ liệu bằng cách trừ đi trung bình đánh giá của từng người dùng.
    3. Dự đoán đánh giá bằng cách sử dụng công thức:

    """
    
    def __init__(self, Y_data, k=40, sim_func=cosine_similarity):
        self.Y_data = Y_data
        self.k = k
        self.sim_func = sim_func
        self.n_users = int(np.max(Y_data[:, 0])) + 1
        self.n_items = int(np.max(Y_data[:, 1])) + 1
        self.Ybar = None
    
    def fit(self):
        users = self.Y_data[:, 0]
        self.Ybar = self.Y_data.copy()
        self.mu = np.zeros(self.n_users)
        print('users:', users)
        print('Ybar:', self.Ybar)
        print('mu:', self.mu)

        for u in range(self.n_users):
            ids = np.where(users == u)[0]
            ratings = self.Y_data[ids, 2]
            self.mu[u] = np.mean(ratings) if ids.size > 0 else 0
            self.Ybar[ids, 2] = ratings - self.mu[u]
        
        self.Ybar = sparse.coo_matrix((self.Ybar[:, 2], (self.Ybar[:, 1], self.Ybar[:, 0])), 
                                      shape=(self.n_items, self.n_users)).tocsr()
        self.S = self.sim_func(self.Ybar.T, self.Ybar.T)
    
    def predict(self, u, i):
        ids = np.where(self.Y_data[:, 1] == i)[0]
        users_rated_i = self.Y_data[ids, 0].astype(np.int32)
        sim = self.S[u, users_rated_i]
        nns = np.argsort(sim)[-self.k:]
        nearest_s = sim[nns]
        r = self.Ybar[i, users_rated_i[nns]]
        eps = 1e-8
        return (r @ nearest_s) / (np.abs(nearest_s).sum() + eps) + self.mu[u]

# Load dữ liệu
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

def load_data(path):
    return pd.read_csv(path, sep='\t', names=r_cols)[['user_id', 'movie_id', 'rating']].values - [1, 1, 0]

rate_train = load_data('ml-100k/ua.base')
rate_test = load_data('ml-100k/ua.test')

print(rate_train)

# User-User Collaborative Filtering
rs = UserUserCF(rate_train, k=40)
rs.fit()
SE = sum((rs.predict(u, i) - r) ** 2 for u, i, r in rate_test)
print('User-user CF, RMSE =', np.sqrt(SE / len(rate_test)))

# Item-Item Collaborative Filtering
rate_train = rate_train[:, [1, 0, 2]]
rate_test = rate_test[:, [1, 0, 2]]
rs = UserUserCF(rate_train, k=40)
rs.fit()
SE = sum((rs.predict(u, i) - r) ** 2 for u, i, r in rate_test)
print('Item-item CF, RMSE =', np.sqrt(SE / len(rate_test)))
