import mysql.connector
import pandas as pd

# Kết nối MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="pnam0212",
    database="movielens"
)
cursor = conn.cursor()

# Tạo bảng trong MySQL
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INT PRIMARY KEY,
        age INT,
        gender CHAR(1),
        occupation VARCHAR(50),
        zip_code VARCHAR(10)
    );
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS movies (
        movie_id INT PRIMARY KEY,
        title VARCHAR(255),
        release_date DATE,
        video_release_date VARCHAR(50),
        imdb_url VARCHAR(255),
        unknown INT,
        Action INT,
        Adventure INT,
        Animation INT,
        Childrens INT,
        Comedy INT,
        Crime INT,
        Documentary INT,
        Drama INT,
        Fantasy INT,
        Film_Noir INT,
        Horror INT,
        Musical INT,
        Mystery INT,
        Romance INT,
        Sci_Fi INT,
        Thriller INT,
        War INT,
        Western INT
    );
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS ratings (
        user_id INT,
        movie_id INT,
        rating INT,
        timestamp INT,
        PRIMARY KEY (user_id, movie_id),
        FOREIGN KEY (user_id) REFERENCES users(user_id),
        FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
    );
""")

# Nhập dữ liệu vào bảng users
users = pd.read_csv("ml-100k/u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"], encoding="latin-1")
for _, row in users.iterrows():
    cursor.execute("INSERT IGNORE INTO users VALUES (%s, %s, %s, %s, %s)", tuple(row))
conn.commit()

# Nhập dữ liệu vào bảng movies
item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'Action', 'Adventure',
             'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
             'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv("ml-100k/u.item", sep="|", names=item_cols, encoding="latin-1")
movies = movies.fillna(0)  # Thay thế NaN bằng 0
for _, row in movies.iterrows():
    cursor.execute("INSERT IGNORE INTO movies VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                   tuple(row))
conn.commit()

# Nhập dữ liệu vào bảng ratings
ratings = pd.read_csv("ml-100k/ua.base", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"], encoding="latin-1")
for _, row in ratings.iterrows():
    cursor.execute("INSERT IGNORE INTO ratings VALUES (%s, %s, %s, %s)", tuple(row))
conn.commit()

# Đóng kết nối
cursor.close()
conn.close()
print("Dữ liệu đã nhập thành công!")