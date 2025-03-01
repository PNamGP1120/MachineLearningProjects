import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge

class ContentBasedRecommender:
    """
    Hệ thống gợi ý dựa trên nội dung sử dụng TF-IDF và hồi quy Ridge.
    """
    
    def __init__(self, data_path='ml-100k'):
        """
        Khởi tạo hệ thống và tải dữ liệu.
        
        Args:
            data_path (str): Đường dẫn đến thư mục chứa dữ liệu MovieLens 100k.
        """
        self.data_path = data_path
        self.users = None
        self.ratings_base = None
        self.ratings_test = None
        self.items = None
        self.tfidf = None
        self.W = None
        self.b = None
        self.n_users = 0
        self._load_data()
        self._compute_tfidf()
    
    def _load_data(self):
        """
        Tải dữ liệu người dùng, đánh giá và phim.
        """
        print('1. Processing data ...')
        
        # Load user data
        user_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.users = pd.read_csv(f'{self.data_path}/u.user', sep='|', names=user_cols, encoding='latin-1')
        self.n_users = self.users.shape[0]
        print(f'Number of users: {self.n_users}')
        
        # Load ratings data
        rating_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        self.ratings_base = pd.read_csv(f'{self.data_path}/ua.base', sep='\t', names=rating_cols, encoding='latin-1')
        self.ratings_test = pd.read_csv(f'{self.data_path}/ua.test', sep='\t', names=rating_cols, encoding='latin-1')
        print(f'Number of ratings: {self.ratings_base.shape[0]}')
        
        # Load movie data
        item_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown',
                     'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 
                     'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 
                     'War', 'Western']
        self.items = pd.read_csv(f'{self.data_path}/u.item', sep='|', names=item_cols, encoding='latin-1')
        print(f'Number of items: {self.items.shape[0]}')
    
    def _compute_tfidf(self):
        """
        Tính toán ma trận TF-IDF từ thể loại phim.
        """
        print('2. Computing TF-IDF ...')
        X_train_counts = self.items.iloc[:, -19:].values  # Lấy thông tin thể loại phim
        transformer = TfidfTransformer(smooth_idf=True, norm=None)
        self.tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()
        print('TF-IDF matrix: \n', self.tfidf)
    
    def get_items_rated_by_user(self, user_id):
        """
        Lấy danh sách các phim mà người dùng đã đánh giá.
        
        Args:
            user_id (int): ID của người dùng.
        
        Returns:
            tuple: (mảng movie_ids, mảng scores tương ứng)
        """
        rate_matrix = self.ratings_base.values
        y = rate_matrix[:, 0]
        ids = np.where(y == user_id + 1)[0]
        item_ids = rate_matrix[ids, 1] - 1  # Giảm 1 để phù hợp với index
        scores = rate_matrix[ids, 2]
        return item_ids, scores
    
    def train(self, alpha=0.01):
        """
        Huấn luyện mô hình Ridge Regression cho từng người dùng.
        
        Args:
            alpha (float): Hệ số điều chuẩn của Ridge Regression.
        """
        print('3. Training model ...')
        d = self.tfidf.shape[1]
        self.W = np.zeros((d, self.n_users))
        self.b = np.zeros((1, self.n_users))
        
        for n in range(self.n_users):
            ids, scores = self.get_items_rated_by_user(n)
            clf = Ridge(alpha=alpha, fit_intercept=True)
            Xhat = self.tfidf[ids, :]
            clf.fit(Xhat, scores)
            self.W[:, n] = clf.coef_
            self.b[0, n] = clf.intercept_
        print('W:', self.W.shape)
        print('b:', self.b.shape)
        print('Training completed!')
    
    def predict(self, user_id):
        """
        Dự đoán điểm số cho một người dùng.
        
        Args:
            user_id (int): ID của người dùng.
        
        Returns:
            numpy.ndarray: Điểm dự đoán cho tất cả phim.
        """
        return self.tfidf.dot(self.W) + self.b[:, user_id]
    
    def evaluate(self, user_id):
        """
        Đánh giá mô hình bằng cách so sánh dự đoán với thực tế.
        
        Args:
            user_id (int): ID của người dùng.
        """
        print(f'Evaluating user {user_id} ...')
        ids, scores = self.get_items_rated_by_user(user_id)
        predicted_scores = self.predict(user_id)[ids]
        print(f'Rated movies ids: {ids}')
        print(f'True ratings: {scores}')
        print(f'Predicted ratings: {predicted_scores}')

# Sử dụng class
if __name__ == "__main__":
    recommender = ContentBasedRecommender()
    recommender.train()
    recommender.evaluate(user_id=1)
