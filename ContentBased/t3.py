import numpy as np
import pandas as pd

# Đọc dữ liệu từ tập ml-100k
rating_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=rating_cols, encoding='latin-1')

# Tạo ma trận utility matrix
utility_matrix = ratings_base.pivot(index='movie_id', columns='user_id', values='rating')

# Kiểm tra giá trị NaN trong ma trận
has_nan = utility_matrix.isna().sum().sum() > 0

if has_nan:
    print("🔴 Ma trận có chứa giá trị NaN.")
else:
    print("✅ Ma trận không có giá trị NaN.")

# Nếu có NaN, in ra số lượng
print("Số lượng giá trị NaN:", utility_matrix.isna().sum().sum())
