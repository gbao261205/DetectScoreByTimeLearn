# evaluate_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 1. Đọc dữ liệu
df = pd.read_excel('data.xlsx', engine='openpyxl')
X = df['hours'].values.reshape(-1,1).astype(float)
y_true = df['score'].values.reshape(-1,1).astype(float)

# 2. Load mô hình
model = tf.keras.models.load_model('trained_model.h5')

# 3. Dự đoán và tính metrics
y_pred = model.predict(X)
mse = mean_squared_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²:  {r2:.4f}")

# 4. Vẽ đồ thị
plt.scatter(X, y_true, label='Dữ liệu thực tế')
plt.plot(X, y_pred, label='Đường dự đoán', linewidth=2)
plt.xlabel('Số giờ học')
plt.ylabel('Điểm số')
plt.title('Kiểm tra mô hình hồi quy tuyến tính')
plt.legend()
plt.show()
