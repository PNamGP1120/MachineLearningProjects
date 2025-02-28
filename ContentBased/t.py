import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge


class ContentBasedRecommender:
    """
    Hệ thống đề xuất dựa trên nội dung (Content-Based Recommendation System)

    Ý tưởng của thuật toán:
    1. Trích xuất đặc trưng của các mục (items) từ dữ liệu nội dung
    2. Chuyển đổi đặc trưng thành biểu diễn vector sử dụng TF-IDF
    3. Xây dựng mô hình ưa thích cho từng người dùng bằng hồi quy Ridge
    4. Dự đoán xếp hạng dựa trên độ tương đồng giữa profile người dùng và đặc trưng mục

    Thuật toán này đặc biệt phù hợp cho các trường hợp có dữ liệu nội dung phong phú
    về các mục (như thể loại phim, từ khóa, mô tả), và có thể giải quyết vấn đề "cold-start"
    cho các mục mới.
    """

    def __init__(self, alpha=0.01):
        """
        Khởi tạo hệ thống đề xuất dựa trên nội dung

        Tham số:
        -----------
        alpha : float, mặc định=0.01
            Hệ số điều chuẩn (regularization strength) cho hồi quy Ridge,
            giúp tránh overfitting khi xây dựng mô hình ưa thích người dùng
        """
        self.alpha = alpha
        self.W = None  # Ma trận trọng số
        self.b = None  # Các hệ số bias
        self.X = None  # Đặc trưng của các mục
        self.n_users = None  # Số lượng người dùng
        self.n_items = None  # Số lượng mục
        self.item_ids = None  # ID của các mục
        self.user_ids = None  # ID của người dùng

    def fit(self, items_df, ratings_df):
        """
        Huấn luyện mô hình đề xuất dựa trên nội dung

        Ý tưởng:
        1. Xử lý đặc trưng của các mục bằng TF-IDF
        2. Với mỗi người dùng, huấn luyện một mô hình hồi quy Ridge riêng biệt
           để dự đoán xếp hạng dựa trên đặc trưng của các mục đã được đánh giá

        Tham số:
        -----------
        items_df : pandas DataFrame
            DataFrame chứa thông tin và đặc trưng của các mục (như phim)
        ratings_df : pandas DataFrame
            DataFrame chứa đánh giá của người dùng cho các mục

        Trả về:
        --------
        self : đối tượng
            Trả về đối tượng hiện tại để hỗ trợ method chaining
        """
        # Xử lý đặc trưng mục
        self.item_ids = items_df.iloc[:, 0].values
        self.n_items = len(self.item_ids)


        # Trích xuất đặc trưng thể loại (giả định 19 cột cuối là các thể loại)
        X_train_counts = items_df.iloc[:, -19:].values

        # Áp dụng biến đổi TF-IDF để tạo trọng số cho các đặc trưng
        transformer = TfidfTransformer(smooth_idf=True, norm='l2')
        self.X = transformer.fit_transform(X_train_counts).toarray()

        self.n_users = self.X.shape[0]
        # Lấy ID người dùng duy nhất
        self.user_ids = ratings_df['user_id'].unique()
        self.n_users = len(self.user_ids)

        # Chuyển đổi đánh giá thành mảng numpy để xử lý nhanh hơn
        rate_matrix = ratings_df[['user_id', 'movie_id', 'rating']].values

        # Khởi tạo tham số mô hình
        d = self.X.shape[1]  # số chiều đặc trưng
        self.W = np.zeros((d, self.n_users))
        self.b = np.zeros(self.n_users)

        # Huấn luyện mô hình hồi quy Ridge cho từng người dùng
        for user_idx, user_id in enumerate(self.user_ids):
            item_ids, scores = self._get_items_rated_by_user(rate_matrix, user_id)

            if len(item_ids) > 0:  # Bỏ qua người dùng không có đánh giá
                # Lấy đặc trưng cho các mục được đánh giá bởi người dùng này
                Xhat = self.X[item_ids, :]

                # Huấn luyện mô hình
                model = Ridge(alpha=self.alpha, fit_intercept=True)
                model.fit(Xhat, scores)

                # Lưu trữ tham số mô hình
                self.W[:, user_idx] = model.coef_
                self.b[user_idx] = model.intercept_

        return self

    def _get_items_rated_by_user(self, rate_matrix, user_id):
        """
        Lấy các mục đã được đánh giá bởi một người dùng cụ thể

        Ý tưởng:
        Tìm kiếm trong ma trận đánh giá tất cả các mục mà người dùng đã đánh giá,
        chuyển đổi ID mục thành chỉ số tương ứng trong ma trận đặc trưng

        Tham số:
        -----------
        rate_matrix : mảng numpy
            Ma trận đánh giá
        user_id : int
            ID của người dùng

        Trả về:
        --------
        item_ids : mảng numpy
            Chỉ số của các mục đã được đánh giá bởi người dùng
        scores : mảng numpy
            Điểm đánh giá của người dùng
        """
        # Tìm các bản ghi cho người dùng này
        user_entries = rate_matrix[:, 0] == user_id

        # Lấy ID mục và điểm số
        item_ids_raw = rate_matrix[user_entries, 1]

        # Chuyển đổi thành chỉ số bắt đầu từ 0 cho ma trận đặc trưng
        item_lookup = {id_val: idx for idx, id_val in enumerate(self.item_ids)}
        item_ids = np.array([item_lookup.get(id_val, -1) for id_val in item_ids_raw])

        # Loại bỏ các mục không hợp lệ (trường hợp có đánh giá cho mục không có trong danh sách)
        valid_items = item_ids >= 0
        item_ids = item_ids[valid_items]

        scores = rate_matrix[user_entries, 2][valid_items]

        return item_ids, scores

    def predict(self, user_indices=None):
        """
        Tạo dự đoán đánh giá cho tất cả người dùng hoặc những người dùng cụ thể

        Ý tưởng:
        Sử dụng công thức dự đoán: Yhat = X * W + b, trong đó:
        - X: ma trận đặc trưng mục
        - W: ma trận trọng số người dùng
        - b: hệ số bias

        Tham số:
        -----------
        user_indices : array-like, tùy chọn
            Chỉ số của người dùng cần dự đoán. Nếu None, dự đoán cho tất cả người dùng.

        Trả về:
        --------
        Yhat : mảng numpy
            Đánh giá dự đoán cho tất cả các mục bởi những người dùng đã chỉ định
        """
        if user_indices is None:
            user_indices = np.arange(self.n_users)

        # Tính toán dự đoán: X * W + b
        Yhat = self.X @ self.W[:, user_indices] + self.b[user_indices]

        return Yhat

    def recommend(self, user_id, n_recommendations=10, exclude_rated=True, ratings_df=None):
        """
        Đề xuất N mục hàng đầu cho một người dùng cụ thể

        Ý tưởng:
        1. Dự đoán đánh giá của người dùng cho tất cả các mục
        2. Sắp xếp các mục theo điểm dự đoán giảm dần
        3. Loại bỏ các mục người dùng đã đánh giá (tùy chọn)
        4. Trả về N mục có điểm cao nhất

        Tham số:
        -----------
        user_id : int
            ID của người dùng
        n_recommendations : int, mặc định=10
            Số lượng đề xuất cần trả về
        exclude_rated : bool, mặc định=True
            Có loại bỏ các mục đã được đánh giá bởi người dùng hay không
        ratings_df : pandas DataFrame, tùy chọn
            DataFrame chứa đánh giá người dùng-mục, cần thiết nếu exclude_rated=True

        Trả về:
        --------
        recommendations : mảng numpy
            ID của các mục được đề xuất
        scores : mảng numpy
            Điểm dự đoán cho các mục được đề xuất
        """
        # Tìm chỉ số người dùng
        user_idx = np.where(self.user_ids == user_id)[0]

        if len(user_idx) == 0:
            raise ValueError(f"User ID {user_id} không tìm thấy trong dữ liệu huấn luyện")

        user_idx = user_idx[0]

        # Lấy dự đoán cho người dùng này
        predictions = self.predict(user_indices=[user_idx]).flatten()

        # Tùy chọn loại bỏ các mục đã được đánh giá bởi người dùng
        if exclude_rated and ratings_df is not None:
            rated_items = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].values
            rated_indices = np.array([np.where(self.item_ids == item_id)[0][0] for item_id in rated_items
                                      if item_id in self.item_ids])

            # Đặt dự đoán cho các mục đã đánh giá thành giá trị rất thấp để loại trừ
            predictions[rated_indices] = float('-inf')

        # Lấy N đề xuất hàng đầu
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]

        recommended_items = self.item_ids[top_indices]
        scores = predictions[top_indices]

        return recommended_items, scores

    def evaluate(self, test_ratings_df):
        """
        Đánh giá mô hình trên dữ liệu kiểm tra

        Ý tưởng:
        1. Tính toán RMSE (Root Mean Square Error) giữa đánh giá thực tế và dự đoán
        2. Chỉ đánh giá trên các mục mà người dùng đã đánh giá trong tập kiểm tra

        Tham số:
        -----------
        test_ratings_df : pandas DataFrame
            DataFrame chứa đánh giá kiểm tra

        Trả về:
        --------
        rmse : float
            Root Mean Squared Error của các dự đoán
        """
        test_matrix = test_ratings_df[['user_id', 'movie_id', 'rating']].values

        # Tính toán dự đoán cho tất cả người dùng
        all_predictions = self.predict()

        squared_errors = 0
        count = 0

        # Lặp qua từng người dùng trong tập kiểm tra
        for user_id in np.unique(test_matrix[:, 0]):
            user_idx = np.where(self.user_ids == user_id)[0]

            if len(user_idx) == 0:  # Bỏ qua người dùng không có trong tập huấn luyện
                continue

            user_idx = user_idx[0]

            # Lấy các mục được đánh giá bởi người dùng này trong tập kiểm tra
            item_ids, true_scores = self._get_items_rated_by_user(test_matrix, user_id)

            if len(item_ids) == 0:  # Bỏ qua nếu không có mục hợp lệ
                continue

            # Lấy dự đoán cho các mục này
            pred_scores = all_predictions[item_ids, user_idx]

            # Tính toán lỗi
            errors = true_scores - pred_scores
            squared_errors += np.sum(errors ** 2)
            count += len(errors)

        if count == 0:
            return float('inf')

        rmse = np.sqrt(squared_errors / count)
        return rmse


# Ví dụ sử dụng
if __name__ == "__main__":
    # Đọc file người dùng
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

    # Đọc file đánh giá
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols)
    ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols)

    # Đọc file mục
    i_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
              'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
    # print(users, ratings_train, ratings_test, items)
    # Tạo và huấn luyện mô hình
    cb_recommender = ContentBasedRecommender(alpha=0.01)
    cb_recommender.fit(items, ratings_train)

    # Đánh giá trên dữ liệu kiểm tra
    rmse = cb_recommender.evaluate(ratings_test)
    print(f"RMSE trên dữ liệu kiểm tra: {rmse:.4f}")

    # Lấy đề xuất cho một người dùng
    user_id = 2
    recommended_items, scores = cb_recommender.recommend(user_id, n_recommendations=5,
                                                         exclude_rated=True, ratings_df=ratings_train)

    print(f"\nTop 5 đề xuất cho người dùng {user_id}:")
    for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
        movie_title = items[items['movie_id'] == item_id]['movie_title'].values[0]
        print(f"{i + 1}. {movie_title} (đánh giá dự đoán: {score:.2f})")