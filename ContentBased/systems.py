import numpy as np
import pandas as pd

# Đọc file user
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

# Đọc file ratings
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

# Đọc file item (danh sách phim)
i_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + [f'genre_{i}' for i in range(19)]
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
items = items.values[:, -19:]

# Đọc file thể loại phim (u.genre)
genre_cols = ['genre', 'genre_id']
genres = pd.read_csv('ml-100k/u.genre', sep='|', names=genre_cols, encoding='latin-1')



