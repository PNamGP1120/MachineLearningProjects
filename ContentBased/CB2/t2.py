import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class ContentBasedRecommender:
    """
    Hệ thống gợi ý dựa trên nội dung sử dụng Ridge Regression để dự đoán đánh giá của người dùng.
    - Áp dụng cho các mục có đặc trưng (feature), ví dụ: điểm cho người lớn và trẻ em.
    - Sử dụng Ridge Regression để tính toán ma trận trọng số W.
    """
    def __init__(self, data_dir='data', alpha=0.01, log_level='INFO'):
        """
        Khởi tạo hệ thống gợi ý.
        
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu
            alpha (float): Hệ số điều chỉnh cho Ridge Regression
            log_level (str): Mức độ chi tiết của log ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        # Thiết lập logging
        self._setup_logging(log_level)
        
        # Thông số
        self.data_dir = Path(data_dir)
        self.alpha = alpha
        
        # Dữ liệu
        self.users = None
        self.ratings = None
        self.items = None
        
        # Thông tin mô hình
        self.W = None  # Ma trận trọng số
        self.n_users = 0
        self.n_items = 0
        self.X = None  # Ma trận đặc trưng
        
        # Mapping ID
        self.user_id_to_index = {}  # Ánh xạ user_id -> index
        self.item_id_to_index = {}  # Ánh xạ item_id -> index
        self.index_to_user_id = {}  # Ánh xạ index -> user_id
        self.index_to_item_id = {}  # Ánh xạ index -> item_id
        
        # Tải dữ liệu
        self._load_data()

    def _setup_logging(self, log_level):
        """
        Thiết lập logging với mức độ chi tiết tùy chỉnh.
        
        Args:
            log_level (str): Mức độ chi tiết của log
        """
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('ContentBasedRecommender')

    def _load_data(self):
        """
        Tải dữ liệu từ các file CSV và chuẩn bị ma trận đặc trưng.
        """
        try:
            # Tải dữ liệu
            self.logger.info(f"Đang tải dữ liệu từ {self.data_dir}")
            self.users = pd.read_csv(self.data_dir / 'users.csv')
            self.items = pd.read_csv(self.data_dir / 'items.csv')
            self.ratings = pd.read_csv(self.data_dir / 'ratings.csv')
            
            # Tạo ánh xạ ID
            self._create_id_mappings()
            
            # Tạo ma trận đặc trưng X với Bias Trick
            X_features = self.items[['adult_score', 'children_score']].values
            self.X = np.hstack((np.ones((X_features.shape[0], 1)), X_features))  # Thêm cột 1 (bias)
            
            # Khởi tạo ma trận trọng số W
            self.W = np.zeros((self.n_users, self.X.shape[1]))
            
            self.logger.info(f"Đã tải thành công: {self.n_users} người dùng, {self.n_items} mục")
            self.logger.debug(f"Ma trận đặc trưng X có kích thước: {self.X.shape}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải dữ liệu: {str(e)}")
            raise

    def _create_id_mappings(self):
        """
        Tạo ánh xạ giữa ID thực tế và chỉ số trong ma trận.
        """
        # Tạo ánh xạ cho users
        unique_user_ids = sorted(self.users['user_id'].unique())
        self.n_users = len(unique_user_ids)
        
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        self.index_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()}
        
        # Tạo ánh xạ cho items
        unique_item_ids = sorted(self.items['item_id'].unique())
        self.n_items = len(unique_item_ids)
        
        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(unique_item_ids)}
        self.index_to_item_id = {idx: item_id for item_id, idx in self.item_id_to_index.items()}
        
        self.logger.debug(f"Đã tạo ánh xạ ID cho {self.n_users} users và {self.n_items} items")

    def get_items_rated_by_user(self, user_id):
        """
        Lấy danh sách item và rating mà user đã đánh giá.
        
        Args:
            user_id (int): ID của người dùng
        
        Returns:
            tuple: (item_indices, ratings)
                - item_indices (np.array): Danh sách chỉ số của item đã đánh giá
                - ratings (np.array): Danh sách rating tương ứng
        """
        if user_id not in self.user_id_to_index:
            self.logger.warning(f"User ID {user_id} không tồn tại trong dữ liệu")
            return np.array([]), np.array([])
        
        # Lọc ratings của user cụ thể
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        
        if user_ratings.empty:
            return np.array([]), np.array([])
        
        # Lọc và chuyển đổi item_id sang index
        valid_items = user_ratings['item_id'].isin(self.item_id_to_index.keys())
        valid_ratings = user_ratings[valid_items]
        
        if valid_ratings.empty:
            return np.array([]), np.array([])
        
        item_indices = np.array([self.item_id_to_index[item_id] for item_id in valid_ratings['item_id']])
        ratings = valid_ratings['rating'].values
        
        return item_indices, ratings

    def train(self, alpha=None):
        """
        Huấn luyện mô hình sử dụng Ridge Regression.
        
        Args:
            alpha (float, optional): Hệ số điều chỉnh. Nếu None, sử dụng giá trị khởi tạo.
        
        Returns:
            self: Đối tượng recommender đã train
        """
        if alpha is not None:
            self.alpha = alpha
            
        self.logger.info(f"Bắt đầu huấn luyện mô hình với alpha={self.alpha}")
        
        # Reset ma trận trọng số
        self.W = np.zeros((self.n_users, self.X.shape[1]))
        
        # Lặp qua từng user
        for user_idx, user_id in self.index_to_user_id.items():
            self.logger.debug(f"Đang huấn luyện cho user {user_id} (index: {user_idx})")
            
            # Lấy dữ liệu đánh giá của user
            item_indices, ratings = self.get_items_rated_by_user(user_id)
            
            if len(item_indices) == 0:
                self.logger.debug(f"User {user_id} không có đánh giá hợp lệ, bỏ qua")
                continue
                
            # Trích xuất đặc trưng và xây dựng mô hình
            X_train = self.X[item_indices]
            y_train = ratings
            
            try:
                # Ridge Regression
                lambda_I = self.alpha * np.eye(X_train.shape[1])
                ridge_solution = np.linalg.pinv(X_train.T @ X_train + lambda_I) @ X_train.T @ y_train
                
                # Cập nhật trọng số
                self.W[user_idx] = ridge_solution
                
                self.logger.debug(f"Đã huấn luyện cho user {user_id}: trọng số = {ridge_solution}")
                
            except np.linalg.LinAlgError as e:
                self.logger.error(f"Lỗi giải ma trận cho user {user_id}: {str(e)}")
                
        self.logger.info("Hoàn thành huấn luyện mô hình")
        return self

    def predict(self, user_id, item_ids=None):
        """
        Dự đoán rating của user cho một hoặc nhiều items.
        
        Args:
            user_id (int): ID của người dùng
            item_ids (int/list, optional): ID của item hoặc danh sách ID các item.
                                         Nếu None, dự đoán cho tất cả items.
        
        Returns:
            float/np.array: Rating dự đoán
        """
        if self.W is None:
            raise ValueError("Mô hình chưa được huấn luyện, vui lòng gọi train() trước")
            
        if user_id not in self.user_id_to_index:
            raise ValueError(f"User ID {user_id} không tồn tại trong dữ liệu")
            
        user_idx = self.user_id_to_index[user_id]
        
        # Trường hợp dự đoán cho tất cả items
        if item_ids is None:
            predictions = self.X @ self.W[user_idx]
            return np.clip(np.round(predictions, 2), 1, 5)
            
        # Trường hợp dự đoán cho một item
        if isinstance(item_ids, (int, np.integer)):
            if item_ids not in self.item_id_to_index:
                raise ValueError(f"Item ID {item_ids} không tồn tại trong dữ liệu")
                
            item_idx = self.item_id_to_index[item_ids]
            prediction = self.X[item_idx] @ self.W[user_idx]
            return float(np.clip(round(prediction, 2), 1, 5))
            
        # Trường hợp dự đoán cho nhiều items
        valid_items = [item_id for item_id in item_ids if item_id in self.item_id_to_index]
        
        if not valid_items:
            raise ValueError("Không có item nào hợp lệ trong danh sách")
            
        item_indices = [self.item_id_to_index[item_id] for item_id in valid_items]
        predictions = self.X[item_indices] @ self.W[user_idx]
        
        return np.clip(np.round(predictions, 2), 1, 5)

    def recommend(self, user_id, top_n=5, exclude_rated=True):
        """
        Gợi ý top-n items cho một người dùng.
        
        Args:
            user_id (int): ID của người dùng
            top_n (int): Số lượng items gợi ý
            exclude_rated (bool): Loại bỏ các items đã được đánh giá
            
        Returns:
            pd.DataFrame: DataFrame chứa thông tin về items được gợi ý
        """
        if user_id not in self.user_id_to_index:
            raise ValueError(f"User ID {user_id} không tồn tại trong dữ liệu")
            
        # Dự đoán rating cho tất cả items
        all_predictions = self.predict(user_id)
        
        # Lấy danh sách items đã được đánh giá
        rated_indices, _ = self.get_items_rated_by_user(user_id)
        
        # Tạo mask cho các items chưa được đánh giá (nếu cần)
        if exclude_rated and len(rated_indices) > 0:
            mask = np.ones(self.n_items, dtype=bool)
            mask[rated_indices] = False
            candidate_indices = np.arange(self.n_items)[mask]
            candidate_scores = all_predictions[mask]
        else:
            candidate_indices = np.arange(self.n_items)
            candidate_scores = all_predictions
            
        # Sắp xếp theo thứ tự giảm dần của rating dự đoán
        if len(candidate_indices) <= top_n:
            top_indices = candidate_indices[np.argsort(-candidate_scores)]
        else:
            top_indices = candidate_indices[np.argsort(-candidate_scores)][:top_n]
            
        # Chuyển đổi từ index sang item_id
        top_item_ids = [self.index_to_item_id[idx] for idx in top_indices]
        top_scores = all_predictions[top_indices]
        
        # Tạo DataFrame kết quả
        recommendations = pd.DataFrame({
            'item_id': top_item_ids,
            'predicted_rating': top_scores
        })
        
        # Bổ sung thông tin về item (nếu có)
        if set(['item_name', 'adult_score', 'children_score']).issubset(self.items.columns):
            item_info = self.items.set_index('item_id')[['item_name', 'adult_score', 'children_score']]
            recommendations = recommendations.join(item_info, on='item_id')
            
        return recommendations

    def evaluate(self, test_size=0.2, random_state=42, strategy='user'):
        """
        Đánh giá mô hình sử dụng RMSE trên tập kiểm tra.
        
        Args:
            test_size (float): Tỷ lệ dữ liệu dùng làm tập kiểm tra (0-1)
            random_state (int): Seed cho việc chia dữ liệu
            strategy (str): Chiến lược chia dữ liệu: 'user' hoặc 'rating'
                - 'user': Chia theo người dùng (đánh giá tổng quát hóa người dùng mới)
                - 'rating': Chia ngẫu nhiên các rating (đánh giá tổng quát hóa rating mới)
            
        Returns:
            dict: Kết quả đánh giá (RMSE, số lượng ratings trong tập test)
        """
        self.logger.info(f"Đánh giá mô hình với chiến lược '{strategy}', test_size={test_size}")
        
        if strategy == 'rating':
            # Chiến lược 1: Chia ngẫu nhiên các ratings
            return self._evaluate_rating_split(test_size, random_state)
        elif strategy == 'user':
            # Chiến lược 2: Chia theo người dùng
            return self._evaluate_user_split(test_size, random_state)
        else:
            raise ValueError(f"Chiến lược '{strategy}' không hợp lệ. Chọn 'user' hoặc 'rating'")

    def _evaluate_rating_split(self, test_size, random_state):
        """
        Đánh giá mô hình bằng cách chia ngẫu nhiên các ratings.
        """
        # Chia dữ liệu thành tập train và test
        train_ratings, test_ratings = train_test_split(
            self.ratings, test_size=test_size, random_state=random_state
        )
        
        # Lưu trữ ratings gốc
        original_ratings = self.ratings
        
        try:
            # Huấn luyện mô hình với tập train
            self.ratings = train_ratings
            self.train()
            
            # Dự đoán và tính RMSE trên tập test
            y_true = []
            y_pred = []
            
            for _, row in test_ratings.iterrows():
                user_id = row['user_id']
                item_id = row['item_id']
                
                if (user_id in self.user_id_to_index) and (item_id in self.item_id_to_index):
                    try:
                        prediction = self.predict(user_id, item_id)
                        y_true.append(row['rating'])
                        y_pred.append(prediction)
                    except Exception as e:
                        self.logger.warning(f"Lỗi khi dự đoán: {str(e)}")
            
            # Khôi phục dữ liệu gốc
            self.ratings = original_ratings
            self.train()  # Huấn luyện lại với toàn bộ dữ liệu
            
            if len(y_true) == 0:
                self.logger.warning("Không có dự đoán nào hợp lệ để đánh giá")
                return {'rmse': None, 'test_count': 0}
                
            # Tính RMSE
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            self.logger.info(f"Kết quả đánh giá: RMSE = {rmse:.4f} (trên {len(y_true)} ratings)")
            
            return {
                'rmse': rmse,
                'test_count': len(y_true)
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi trong quá trình đánh giá: {str(e)}")
            # Khôi phục dữ liệu gốc
            self.ratings = original_ratings
            return {'rmse': None, 'test_count': 0}

    def _evaluate_user_split(self, test_size, random_state):
        """
        Đánh giá mô hình bằng cách chia theo người dùng.
        """
        # Chia người dùng thành tập train và test
        user_ids = np.array(list(self.user_id_to_index.keys()))
        np.random.seed(random_state)
        np.random.shuffle(user_ids)
        
        split_idx = int(len(user_ids) * (1 - test_size))
        train_users = user_ids[:split_idx]
        test_users = user_ids[split_idx:]
        
        # Chia ratings
        train_ratings = self.ratings[self.ratings['user_id'].isin(train_users)]
        test_ratings = self.ratings[self.ratings['user_id'].isin(test_users)]
        
        # Lưu trữ ratings gốc
        original_ratings = self.ratings
        
        try:
            # Huấn luyện mô hình với tập train
            self.ratings = train_ratings
            self._create_id_mappings()  # Cập nhật lại ánh xạ ID
            self.train()
            
            # Dự đoán và tính RMSE trên tập test
            y_true = []
            y_pred = []
            
            # Chỉ áp dụng cold-start cho users đã có trong tập train
            common_users = set(train_users).intersection(set(test_users))
            test_subset = test_ratings[test_ratings['user_id'].isin(common_users)]
            
            for _, row in test_subset.iterrows():
                user_id = row['user_id']
                item_id = row['item_id']
                
                if (user_id in self.user_id_to_index) and (item_id in self.item_id_to_index):
                    try:
                        prediction = self.predict(user_id, item_id)
                        y_true.append(row['rating'])
                        y_pred.append(prediction)
                    except Exception as e:
                        self.logger.warning(f"Lỗi khi dự đoán: {str(e)}")
            
            # Khôi phục dữ liệu gốc
            self.ratings = original_ratings
            self._create_id_mappings()  # Cập nhật lại ánh xạ ID
            self.train()  # Huấn luyện lại với toàn bộ dữ liệu
            
            if len(y_true) == 0:
                self.logger.warning("Không có dự đoán nào hợp lệ để đánh giá")
                return {'rmse': None, 'test_count': 0}
                
            # Tính RMSE
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            self.logger.info(f"Kết quả đánh giá: RMSE = {rmse:.4f} (trên {len(y_true)} ratings)")
            
            return {
                'rmse': rmse,
                'test_count': len(y_true)
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi trong quá trình đánh giá: {str(e)}")
            # Khôi phục dữ liệu gốc
            self.ratings = original_ratings
            self._create_id_mappings()
            return {'rmse': None, 'test_count': 0}

    def save_model(self, filepath):
        """
        Lưu mô hình vào file.
        
        Args:
            filepath (str): Đường dẫn đến file lưu trữ
        """
        model_data = {
            'W': self.W,
            'X': self.X,
            'alpha': self.alpha,
            'user_id_to_index': self.user_id_to_index,
            'item_id_to_index': self.item_id_to_index,
            'index_to_user_id': self.index_to_user_id,
            'index_to_item_id': self.index_to_item_id
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Đã lưu mô hình thành công vào {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu mô hình: {str(e)}")
            return False

    def load_model(self, filepath):
        """
        Tải mô hình từ file.
        
        Args:
            filepath (str): Đường dẫn đến file mô hình
            
        Returns:
            bool: True nếu tải thành công, False nếu có lỗi
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.W = model_data['W']
            self.X = model_data['X']
            self.alpha = model_data['alpha']
            self.user_id_to_index = model_data['user_id_to_index']
            self.item_id_to_index = model_data['item_id_to_index']
            self.index_to_user_id = model_data['index_to_user_id']
            self.index_to_item_id = model_data['index_to_item_id']
            
            self.n_users = len(self.user_id_to_index)
            self.n_items = len(self.item_id_to_index)
            
            self.logger.info(f"Đã tải mô hình thành công từ {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải mô hình: {str(e)}")
            return False

    def get_similar_items(self, item_id, top_n=5):
        """
        Tìm các items tương tự dựa trên đặc trưng.
        
        Args:
            item_id (int): ID của item cần tìm tương tự
            top_n (int): Số lượng items tương tự cần trả về
            
        Returns:
            pd.DataFrame: DataFrame chứa thông tin về items tương tự
        """
        if item_id not in self.item_id_to_index:
            raise ValueError(f"Item ID {item_id} không tồn tại trong dữ liệu")
            
        # Lấy đặc trưng của item (bỏ qua bias)
        item_idx = self.item_id_to_index[item_id]
        item_features = self.X[item_idx, 1:]  # Bỏ qua cột bias
        
        # Tính khoảng cách cosine giữa item này và tất cả items khác
        similarities = []
        
        for idx in range(self.n_items):
            if idx != item_idx:  # Bỏ qua item hiện tại
                other_features = self.X[idx, 1:]  # Bỏ qua bias
                
                # Tính độ tương tự cosine
                dot_product = np.dot(item_features, other_features)
                norm_product = np.linalg.norm(item_features) * np.linalg.norm(other_features)
                
                if norm_product == 0:
                    similarity = 0
                else:
                    similarity = dot_product / norm_product
                    
                similarities.append((idx, similarity))
                
        # Sắp xếp theo độ tương tự giảm dần
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy top-n items
        top_indices = [idx for idx, _ in similarities[:top_n]]
        top_similarities = [sim for _, sim in similarities[:top_n]]
        top_item_ids = [self.index_to_item_id[idx] for idx in top_indices]
        
        # Tạo DataFrame kết quả
        result = pd.DataFrame({
            'item_id': top_item_ids,
            'similarity': top_similarities
        })
        
        # Bổ sung thông tin về item (nếu có)
        if set(['item_name', 'adult_score', 'children_score']).issubset(self.items.columns):
            item_info = self.items.set_index('item_id')[['item_name', 'adult_score', 'children_score']]
            result = result.join(item_info, on='item_id')
            
        return result


if __name__ == "__main__":
    # Khởi tạo recommender
    recommender = ContentBasedRecommender(data_dir='data', alpha=0.01, log_level='INFO')
    
    # Huấn luyện mô hình
    recommender.train()
    
    # Đánh giá mô hình
    eval_result = recommender.evaluate(test_size=0.2, strategy='rating')
    print(f"RMSE trên tập test: {eval_result['rmse']:.4f}")
    
    # Gợi ý cho một người dùng
    if recommender.n_users > 0:
        user_id = list(recommender.user_id_to_index.keys())[0]
        recommendations = recommender.recommend(user_id, top_n=5)
        print(f"\nGợi ý cho user {user_id}:")
        print(recommendations)
        
        # Lưu mô hình
        recommender.save_model('recommender_model.pkl')