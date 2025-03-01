import numpy as np
import pandas as pd

class ContentBasedRecommender:
    """
    Hệ thống gợi ý dựa trên nội dung sử dụng Ridge Regression để dự đoán đánh giá của người dùng.
    - Áp dụng cho các mục có đặc trưng (feature), ví dụ: điểm cho người lớn và trẻ em.
    - Sử dụng Ridge Regression để tính toán ma trận trọng số W.
    """
    def __init__(self):
        """
        Khởi tạo hệ thống gợi ý.
        - users: DataFrame chứa thông tin người dùng (user_id, user_name)
        - ratings: DataFrame chứa đánh giá của người dùng đối với các mục
        - items: DataFrame chứa thông tin các mục (item_id, adult_score, children_score)
        - W: Ma trận trọng số của từng user đối với các đặc trưng của item
        - n_users: Số lượng người dùng
        - X: Ma trận đặc trưng của các mục (kèm bias trick)
        """
        self.users = None
        self.ratings = None
        self.items = None
        self.W = None  # Ma trận trọng số
        self.n_users = 0
        self.X = None
        self._load_data()

    def _load_data(self):
        """
        Tải dữ liệu từ các file CSV:
        - users.csv: Chứa thông tin người dùng (user_id, user_name)
        - items.csv: Chứa thông tin mục (item_id, adult_score, children_score)
        - ratings.csv: Chứa đánh giá của người dùng đối với các mục
        
        Thiết lập ma trận đặc trưng X của item với Bias Trick.
        """
        self.users = pd.read_csv('data/users.csv', usecols=['user_id', 'user_name'])
        self.items = pd.read_csv('data/items.csv', usecols=['item_id', 'adult_score', 'children_score'])
        self.ratings = pd.read_csv('data/ratings.csv')

        self.n_users = self.users.shape[0]
        X = self.items[['adult_score', 'children_score']].values

        # Thêm Bias Trick: Cột 1s vào X
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))  # (n_items, 3)
        self.W = np.zeros((self.n_users, self.X.shape[1]))  # (n_users, 3)
        
        print("Users:\n", self.users)
        print("Items:\n", self.items)
        print("Ratings:\n", self.ratings)
        print("n_users: ", self.n_users)
        print("X:\n", self.X)
        print("W:\n", self.W)
        

    def get_items_rated_by_user(self, user_id):
        """
        Lấy danh sách item và rating mà user đã đánh giá.
        
        Args:
            user_id (int): ID của người dùng
        
        Returns:
            tuple: (item_ids, ratings)
            - item_ids (np.array): Danh sách ID của item mà user đã đánh giá
            - ratings (np.array): Danh sách rating tương ứng
        """
        rated_items = self.ratings[self.ratings['user_id'] == user_id]
        item_ids = rated_items['item_id'].values
        ratings = rated_items['rating'].values
        return item_ids, ratings

    def train(self, alpha=0.01):
        """
        Huấn luyện mô hình sử dụng Ridge Regression để tính toán ma trận trọng số W.
        
        Args:
            alpha (float, optional): Hệ số điều chỉnh (regularization parameter). Mặc định là 0.01.
        
        Công thức cập nhật W cho từng người dùng:
            W_user = (X.T * X + alpha * I)^(-1) * X.T * Y
        
        Quy trình thực hiện:
        1. Lặp qua từng user_id.
        2. Lấy danh sách item mà user đã đánh giá cùng rating tương ứng.
        3. Chuyển item_id từ 1-based sang 0-based.
        4. Loại bỏ các item_id không hợp lệ.
        5. Trích xuất đặc trưng của các item mà user đã đánh giá thành X_train.
        6. Xây dựng vector y_train chứa rating của user cho các item đó.
        7. Áp dụng Ridge Regression để tính W_user.
        8. Lưu kết quả vào ma trận W.
        """
        
        for user_id in range(1, self.n_users + 1):
            print("-----------------------------------------------------")
            item_ids, ratings = self.get_items_rated_by_user(user_id)
            print("User", user_id, "rated items:", item_ids, "with ratings:", ratings)
            if len(item_ids) == 0:
                continue  # Nếu user không đánh giá mục nào thì bỏ qua
            
            item_ids = item_ids - 1  # Chuyển từ item_id về index (0-based)
            print("Item IDs (0-based):", item_ids)
            # item_ids = item_ids[item_ids < self.X.shape[0]]  # Loại bỏ index không hợp lệ
            # print("Valid item IDs:", item_ids)
            
            X_train = self.X[item_ids]  # Ma trận đặc trưng của item đã đánh giá
            print("X_train:", X_train)
            y_train = ratings[: len(item_ids)]  # Đảm bảo kích thước phù hợp
            print("y_train:", y_train)

            # Ridge Regression với Bias Trick
            lambda_I = alpha * np.eye(X_train.shape[1])
            r = np.linalg.pinv(X_train.T @ X_train + lambda_I) @ X_train.T @ y_train

            self.W[user_id - 1] = r  # Cập nhật W cho user_id
            
            print("-----------------------------------------------------")
            

    def predict(self, user_id):
        """
        Dự đoán rating của user cho tất cả items.
        
        Args:
            user_id (int): ID của người dùng
        
        Returns:
            np.array: Danh sách rating dự đoán cho tất cả items
        
        Công thức dự đoán:
            rating_predicted = X * W_user
        """
        if self.W is None:
            raise ValueError("Model chưa được train. Vui lòng gọi train() trước.")
        
        return np.round(self.X @ self.W[user_id - 1], 2)
    
    def evaluate(self):
        """
        Đánh giá mô hình sử dụng RMSE (Root Mean Squared Error).
        
        Returns:
            float: Giá trị RMSE của mô hình
        """
        errors = []
        for user_id in range(1, self.n_users + 1):
            item_ids, actual_ratings = self.get_items_rated_by_user(user_id)
            if len(item_ids) == 0:
                continue  # Nếu user không đánh giá mục nào, bỏ qua

            item_ids = item_ids - 1  # Chuyển từ item_id về index (0-based)
            item_ids = item_ids[item_ids < self.X.shape[0]]  # Loại bỏ index không hợp lệ
            
            predicted_ratings = self.predict(user_id)[item_ids]  # Dự đoán rating cho các item
            
            # Tính lỗi bình phương
            squared_errors = (actual_ratings[:len(predicted_ratings)] - predicted_ratings) ** 2
            errors.extend(squared_errors)

        # Tính RMSE
        rmse = np.sqrt(np.mean(errors)) if errors else None
        return rmse



if __name__ == "__main__":
    recommender = ContentBasedRecommender()
    recommender.train()
    
    
    for i in range(1, recommender.n_users + 1):
        print(f"Dự đoán cho user {i}:", recommender.predict(i))
        
    rmse = recommender.evaluate()
    print("RMSE:", rmse)
