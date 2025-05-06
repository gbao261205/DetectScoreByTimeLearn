import pandas as pd
import tensorflow as tf

# 1. Đọc dữ liệu từ Excel, chỉ rõ engine
df = pd.read_excel('data.xlsx', engine='openpyxl')
# Hoặc nếu bạn cài xlrd và file là .xls:
# df = pd.read_excel('data.xlsx', engine='xlrd')

# 2. Chuẩn bị dữ liệu
X = df['hours'].values.reshape(-1,1).astype(float)
y = df['score'].values.reshape(-1,1).astype(float)

# 3. Xây mô hình
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
model.compile(optimizer='sgd', loss='mean_squared_error')

# 4. Huấn luyện
model.fit(X, y, epochs=500, verbose=0)

# 5. Lưu mô hình
model.save('trained_model.h5')
print("Huấn luyện xong, mô hình đã lưu vào trained_model.h5")
