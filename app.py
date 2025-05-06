import streamlit as st
import numpy as np
import tensorflow as tf

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model('trained_model.h5')

st.title("Dự đoán điểm số sinh viên")

# Thanh trượt nhập số giờ học: 0-24 tiếng
hours = st.slider(
    "Chọn số giờ học:",
    min_value=0,
    max_value=24,
    value=5
)

# Dự đoán điểm số
predicted_score = model.predict([[hours]])[0][0]

# Giới hạn điểm trong khoảng 0-10
predicted_score = max(0.0, min(10.0, predicted_score))

st.write(f"Điểm số dự đoán khi học {hours} giờ là: **{predicted_score:.2f}** điểm")