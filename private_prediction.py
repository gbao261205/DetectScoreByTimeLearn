# private_prediction.py
import sys
import tensorflow as tf
import tf_encrypted as tfe

# 1) Load model và lấy weight, bias
pretrained = tf.keras.models.load_model('trained_model.h5')
w_val, b_val = pretrained.get_weights()

# 2) Đọc input giờ học từ argv
hour = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

# 3) Thiết lập SecureNN
protocol = tfe.protocol.SecureNN()
tfe.set_protocol(protocol)

with protocol:
    W = tfe.define_public_variable(tf.constant(w_val, dtype=tf.float32))
    B = tfe.define_public_variable(tf.constant(b_val, dtype=tf.float32))
    x_private = tfe.define_private_input('client',
        lambda: tf.constant([[hour]], dtype=tf.float32))
    y_enc = tfe.add(tfe.matmul(x_private, W), B)
    y_plain = y_enc.reveal()

with tfe.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(y_plain)
    print(f"Giờ học = {hour} → Điểm dự đoán (bảo mật): {result[0][0]:.4f}")