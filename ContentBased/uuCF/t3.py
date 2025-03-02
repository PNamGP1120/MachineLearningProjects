import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Đọc dữ liệu
users = pd.read_csv('d2/user.csv')
items = pd.read_csv('d2/item.csv')
ratings = pd.read_csv('d2/rating.csv')

# Tạo ma trận utility (user x item)
utility_matrix = ratings.pivot(index="user_id", columns="item_id", values="rating").values

# Tính trung bình rating từng user để chuẩn hóa
means = np.nanmean(utility_matrix, axis=1, keepdims=True)

# Chuẩn hóa ma trận bằng cách trừ đi trung bình từng hàng
normalized_matrix = np.where(np.isnan(utility_matrix), 0, utility_matrix - means)

# Tính ma trận tương đồng giữa users
user_similarity = cosine_similarity(normalized_matrix)


# Hàm dự đoán rating cho ô bị thiếu
def predict_rating(user_id, item_id, k=2):
    # Lấy các user đã đánh giá item_id
    rated_users = np.where(~np.isnan(utility_matrix[:, item_id]))[0]

    # Tìm k user tương đồng nhất với user_id
    similarities = user_similarity[user_id, rated_users]
    sorted_indices = np.argsort(-np.abs(similarities))[:k]  # Chọn k giá trị lớn nhất

    # Chọn các user gần nhất
    nearest_users = rated_users[sorted_indices]
    nearest_similarities = similarities[sorted_indices]

    # Lấy ratings của các user này cho item_id
    nearest_ratings = normalized_matrix[nearest_users, item_id]

    # Tính rating dự đoán
    if np.sum(np.abs(nearest_similarities)) == 0:
        return means[user_id]  # Trả về trung bình nếu không có user nào gần
    predicted_rating = np.dot(nearest_similarities, nearest_ratings) / np.sum(np.abs(nearest_similarities))

    # Chuyển về thang đo ban đầu
    return predicted_rating + means[user_id]


# Dự đoán giá trị thiếu trong ma trận
for i in range(utility_matrix.shape[0]):
    for j in range(utility_matrix.shape[1]):
        if np.isnan(utility_matrix[i, j]):
            utility_matrix[i, j] = predict_rating(i, j)

print("Ma trận hoàn chỉnh sau khi dự đoán:")
print(np.round(utility_matrix, 2))
