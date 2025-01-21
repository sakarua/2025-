import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
import matplotlib
matplotlib.use('TkAgg')

# 忽略警告信息
warnings.filterwarnings('ignore')

# 确保使用的是 TensorFlow 2.x
assert tf.__version__.startswith('2'), "请确保使用 TensorFlow 2.x 版本。"

# 1. 生成符合正态分布的随机数据
np.random.seed(42)  # 设置随机种子，确保结果可复现
n_samples_train = 300  # 训练数据样本数量
n_samples_test = 100   # 测试数据样本数量
mean = 400  # 平均分
std = 100   # 标准差
low, high = 100, 700  # 成绩范围

# 生成训练数据
train_scores = np.random.normal(mean, std, n_samples_train)
train_scores = np.clip(train_scores, low, high)  # 截断数据，确保在 100-700 之间

# 生成测试数据
test_scores = np.random.normal(mean, std, n_samples_test)
test_scores = np.clip(test_scores, low, high)  # 截断数据，确保在 100-700 之间

# 打印生成的成绩数据
print("训练数据（前10个）：", train_scores[:10])
print("测试数据（前10个）：", test_scores[:10])

# 2. 数据预处理
# 假设输入特征是一个随机变量（例如学生的平时成绩）
X_train = np.random.rand(n_samples_train, 1) * 100  # 平时成绩（0-100分）
X_test = np.random.rand(n_samples_test, 1) * 100    # 平时成绩（0-100分）

# 对输入特征进行标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 将数据扩展为 3D 形状，以符合LSTM网络的输入要求
X_train_scaled = np.expand_dims(X_train_scaled, axis=2)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)

# 3. 定义双向LSTM网络
def build_bidirectional_lstm(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(128, activation='tanh', return_sequences=True))(inputs)
    x = layers.Bidirectional(layers.LSTM(64, activation='tanh', return_sequences=False))(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 4. 构建模型
bilstm_model = build_bidirectional_lstm(input_shape=(X_train_scaled.shape[1], 1))
bilstm_model.summary()

# 5. 编译模型
bilstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 6. 训练双向LSTM网络
history_bilstm = bilstm_model.fit(
    X_train_scaled, train_scores,
    validation_data=(X_test_scaled, test_scores),
    epochs=100,
    batch_size=32,
    verbose=1
)

# 7. 模型评估
# 进行预测
train_pred = bilstm_model.predict(X_train_scaled).flatten()
test_pred = bilstm_model.predict(X_test_scaled).flatten()

# 定义模型评价函数
def evaluate_model(Y_true, Y_pred, dataset_type="数据集"):
    mse = mean_squared_error(Y_true, Y_pred)
    mae = mean_absolute_error(Y_true, Y_pred)
    r2 = r2_score(Y_true, Y_pred)
    print(f"{dataset_type} - MSE: {mse:.4f}")
    print(f"{dataset_type} - MAE: {mae:.4f}")
    print(f"{dataset_type} - R²: {r2:.4f}\n")

# 评估训练集和测试集的性能
print("双向LSTM模型评价指标：")
evaluate_model(train_scores, train_pred, "训练集")
evaluate_model(test_scores, test_pred, "测试集")

# 8. 结果可视化
plt.figure(figsize=(14, 6))

# 训练集对比图
plt.subplot(1, 2, 1)
plt.scatter(train_scores, train_pred, color='blue', alpha=0.5, label='预测值')
plt.plot([train_scores.min(), train_scores.max()], [train_scores.min(), train_scores.max()], 'k--', lw=2, label='理想情况')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('训练集: 预测值 vs. 真实值')
plt.legend()

# 测试集对比图
plt.subplot(1, 2, 2)
plt.scatter(test_scores, test_pred, color='green', alpha=0.5, label='预测值')
plt.plot([test_scores.min(), test_scores.max()], [test_scores.min(), test_scores.max()], 'k--', lw=2, label='理想情况')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('测试集: 预测值 vs. 真实值')
plt.legend()

plt.tight_layout()
plt.show()