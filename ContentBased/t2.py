import pandas as pd
import numpy as np

# 1. processing data
print('1. Processing data ...')
# User
user_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')
n_users = users.shape[0]
print('Number of users:', n_users)
print('Users \n', users)

# Rating
rating_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=rating_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=rating_cols, encoding='latin-1')

print('Number of ratings:', ratings_base.shape[0])
print('Ratings \n', ratings_base)
# print('Number of ratings:', ratings_test.shape[0])
# print('Ratings \n', ratings_test)

rate_train = ratings_base.values
rate_test = ratings_test.values

# Item
item_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
             'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
             'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=item_cols, encoding='latin-1')
X0 = items.iloc[:, -19:]
print('Number of items:', X0.shape[0])
print('Items X0 \n', X0)

X_train_counts = X0.values
# np.set_printoptions(threshold=np.inf)
# print('X_train_counts \n', X_train_counts)

# 2. Content-Based
print('2. Content-Based ...')
# a. Tfidf
print('a. Tfidf ...')
# tinh tfidf
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()
# print('tfidf \n', tfidf)

# b. lấy ra các item đã được user đánh giá
print('b. Get items rated by user ...')
def get_items_rated_by_user(rate_matrix, user_id):
    """_summary_

    Returns:
        (items_id, scores): _description_
    """
    y = rate_matrix[:, 0]
    ids = np.where(y == user_id + 1)[0]
    item_ids = rate_matrix[ids, 1]-1
    scores = rate_matrix[ids, 2]
    return item_ids, scores


from sklearn.linear_model import Ridge

d = tfidf.shape[1]
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))

for n in range(n_users):
    ids, scores = get_items_rated_by_user(rate_train, n)
    clf = Ridge(alpha=0.01, fit_intercept = True)
    Xhat = tfidf[ids, :]
    clf.fit(Xhat, scores)
    W[:, n] = clf.coef_
    b[0, n] = clf.intercept_
    
# predicted scores
Yhat = tfidf.dot(W) + b

n = 10
np.set_printoptions(precision=2)  # 2 digits after .
ids, scores = get_items_rated_by_user(rate_test, n)
Yhat[n, ids]
print('Rated movies ids:', ids)
print('True ratings:', scores)
print('Predicted ratings:', Yhat[n, ids])