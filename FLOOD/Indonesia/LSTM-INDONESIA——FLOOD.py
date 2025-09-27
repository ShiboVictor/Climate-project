# 调用相关库
import os  # 导入os模块，用于操作系统功能，比如环境变量
import math  # 导入math模块，提供基本的数学功能
import pandas as pd  # 导入pandas模块，用于数据处理和分析
import openpyxl
from math import sqrt  # 从math模块导入sqrt函数，用于计算平方根
from numpy import concatenate  # 从numpy模块导入concatenate函数，用于数组拼接
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
import numpy as np  # 导入numpy模块，用于数值计算
# import tensorflow as tf  # 导入tensorflow模块，用于深度学习

from sklearn.preprocessing import MinMaxScaler  # 导入sklearn中的MinMaxScaler，用于特征缩放
from sklearn.preprocessing import StandardScaler  # 导入sklearn中的StandardScaler，用于特征标准化
from sklearn.preprocessing import LabelEncoder  # 导入sklearn中的LabelEncoder，用于标签编码
from sklearn.metrics import mean_squared_error  # 导入sklearn中的mean_squared_error，用于计算均方误差
from tensorflow.keras.layers import *  # 从tensorflow.keras.layers导入所有层，用于构建神经网络
from tensorflow.keras.models import *  # 从tensorflow.keras.models导入所有模型，用于构建和管理模型
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 导入额外的评估指标
from pandas import DataFrame  # 从pandas导入DataFrame，用于创建和操作数据表
from pandas import concat  # 从pandas导入concat函数，用于DataFrame的拼接
import keras.backend as K  # 导入keras的后端接口
from scipy.io import savemat, loadmat  # 从scipy.io导入savemat和loadmat，用于MATLAB文件的读写
from sklearn.neural_network import MLPRegressor  # 从sklearn.neural_network导入MLPRegressor，用于创建多层感知器回归模型
from keras.callbacks import LearningRateScheduler  # 从keras.callbacks导入LearningRateScheduler，用于调整学习率
from tensorflow.keras import Input, Model, Sequential  # 从tensorflow.keras导入Input, Model和Sequential，用于模型构建
import mplcyberpunk
from qbstyles import mpl_style
import warnings
from prettytable import PrettyTable  # 可以优美的打印表格结果
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional, LayerNormalization, Input
# 从keras.layers模块导入多种层类型。
# Dense是用于创建全连接层的类。
# Activation是用于添加激活函数的层。
# Dropout是用于减少过拟合的丢弃层。
# LSTM是长短时记忆网络层，用于处理序列数据。
# Bidirectional是用于创建双向LSTM层的包装器。
# LayerNormalization是用于层级归一化的类。
# Input是用于模型输入层的函数。
from tensorflow.keras.models import Model

# 从tensorflow.keras.models模块导入Model类。
# Model是用于创建Keras函数式API模型的类。
warnings.filterwarnings("ignore")  # 取消警告
warnings.filterwarnings("ignore", category=UserWarning, module=r"google\.protobuf\.runtime_version")

# %%
dataset = pd.read_csv("INDONESIA-FloodPrediction.csv", encoding='UTF-8')
# 使用pandas模块的read_csv函数读取名为"INDONESIA-DroughtPrediction.csv"的文件。
# 参数'encoding'设置为'UTF-8'。
# 读取的数据被存储在名为'dataset'的DataFrame变量中。
print(dataset)  # 显示dataset数据

# 保存月份信息用于后续绘图（需保证CSV中有"Month"列，形如"2010-01"）
months_data = dataset['Month'].values  # 保存月份信息

# 取出特征与目标（假设最后一列为 rx5day，前面列为气候因子）
values = dataset.values[:, 1:]  # 去除第1列（Month），保留其余列（含特征与目标）
values = values.astype('float32')

# 对齐长度（仅使用本CSV，不依赖CEEMDAN文件）
min_length = values.shape[0]
values_aligned = values[:min_length]
months_aligned = months_data[:min_length]

# 分离特征和目标变量（最后一列视为 rx5day）
features_aligned = values_aligned[:, :-1]   # 原始特征（不包含rx5day）
target_aligned   = values_aligned[:, -1]    # 目标变量（rx5day）

print(f"对齐后长度: {min_length}")
print(f"特征维度: {features_aligned.shape[1]}")

# %%
def data_collation(data, n_in, n_out, or_dim, scroll_window, num_samples):
    res = np.zeros((num_samples, n_in * or_dim + n_out))
    for i in range(0, num_samples):
        h1 = data[scroll_window * i: n_in + scroll_window * i, 0:or_dim]
        h2 = h1.reshape(1, n_in * or_dim)
        h3 = data[n_in + scroll_window * (i): n_in + scroll_window * (i) + n_out, -1].T
        h4 = h3[np.newaxis, :]
        h5 = np.hstack((h2, h4))
        res[i, :] = h5
    return res

# %%
def lstm_model(input_shape, output_dim):
    # 定义一个函数来创建LSTM模型（保持原始简单架构）
    inputs = Input(shape=input_shape)
    lstm = LSTM(128, return_sequences=False)(inputs)
    outputs = Dense(output_dim)(lstm)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='Adam')
    model.summary()
    return model

# %%
# 与现有脚本保持一致的参数
n_in = 3   # 输入前5行的数据
n_out = 1  # 预测未来1步的数据
scroll_window = 1

# 计算可用样本数量（尽可能使用靠近全体的数据量，测试集是最后的15%：即靠近 2023-12 的时间段）
T = features_aligned.shape[0]
max_samples = (T - n_in - n_out) // scroll_window + 1
if max_samples <= 0:
    raise ValueError(f"序列太短：T={T}, 需要至少 n_in+n_out={n_in + n_out}")

original_num_samples = max_samples  # 使用可用的最大样本量
or_dim = features_aligned.shape[1]  # 特征维度（不包含目标）

print(f"计划使用 {original_num_samples} 个样本")
n_train_number = int(original_num_samples * 0.85)
n_test_number  = original_num_samples - n_train_number
print(f"训练集: {n_train_number} 个样本")
print(f"测试集: {n_test_number} 个样本（约{n_test_number}个月，即{n_test_number / 12:.1f}年）")

# 组装“特征 + 目标(rx5day)”并做样本拼接
combined_data = np.hstack((features_aligned, target_aligned.reshape(-1, 1)))
res = data_collation(combined_data, n_in, n_out, or_dim, scroll_window, original_num_samples)
combined_data_processed = np.array(res)

# 按 85% / 15% 划分训练/测试（测试集为时间上最近的一段）
Xtrain = combined_data_processed[:n_train_number, :n_in * or_dim]
Ytrain = combined_data_processed[:n_train_number, n_in * or_dim:]

Xtest  = combined_data_processed[n_train_number:, :n_in * or_dim]
Ytest  = combined_data_processed[n_train_number:, n_in * or_dim:]

# 归一化（按训练集 fit），保持与原脚本一致
m_in = MinMaxScaler()
vp_train = m_in.fit_transform(Xtrain)
vp_test  = m_in.transform(Xtest)

m_out = MinMaxScaler()
vt_train = m_out.fit_transform(Ytrain)
vt_test  = m_out.transform(Ytest)

# 重塑为 LSTM 所需三维张量
vp_train = vp_train.reshape((vp_train.shape[0], n_in, or_dim))
vp_test  = vp_test.reshape((vp_test.shape[0], n_in, or_dim))

# 训练普通 LSTM（不使用 CEEMDAN）
model = lstm_model((n_in, or_dim), vt_train.shape[1])
model.fit(
    vp_train, vt_train,
    batch_size=32, epochs=50,
    validation_split=0.0,   # 不用随机切分，保持时间连续性
    shuffle=False,          # 关键：不要打乱时间顺序
    verbose=2
)

# 预测并反归一化
yhat = model.predict(vp_test)
yhat = yhat.reshape(n_test_number, n_out)
pre_test = m_out.inverse_transform(yhat)  # 反归一化得到预测的 rx5day

# 真实值（测试段）
actual_test = Ytest.reshape(n_test_number, n_out)

# 计算测试集对应的月份（用于横轴）
test_start_index = n_train_number * scroll_window + n_in
test_end_index   = test_start_index + n_test_number
test_months = months_aligned[test_start_index: test_end_index]

if len(test_months) > 0:
    print(f"测试集时间范围: {test_months[0]} 到 {test_months[-1]}")
    print(f"测试集包含 {len(test_months)} 个月份")

# %%
def mape(y_true, y_pred):
    # 定义一个计算平均绝对百分比误差（MAPE）的函数。
    record = []
    for index in range(len(y_true)):
        if abs(y_true[index]) > 1e-5:  # 避免除零错误
            temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
            record.append(temp_mape)
    return np.mean(record) * 100 if record else 0

# %%
def evaluate_forecasts(Ytest, predicted_data, n_out):
    # 定义一个函数来评估预测的性能。
    mse_dic = []
    rmse_dic = []
    mae_dic = []
    mape_dic = []
    r2_dic = []
    table = PrettyTable(['测试集指标', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2'])
    for i in range(n_out):
        actual = [float(row[i]) for row in Ytest]
        predicted = [float(row[i]) for row in predicted_data]
        mse = mean_squared_error(actual, predicted)
        mse_dic.append(mse)
        rmse = sqrt(mean_squared_error(actual, predicted))
        rmse_dic.append(rmse)
        mae = mean_absolute_error(actual, predicted)
        mae_dic.append(mae)
        MApe = mape(actual, predicted)
        mape_dic.append(MApe)
        r2 = r2_score(actual, predicted)
        r2_dic.append(r2)
        if n_out == 1:
            strr = '预测结果指标：'
        else:
            strr = '第' + str(i + 1) + '步预测结果指标：'
        table.add_row([strr, mse, rmse, mae, str(MApe) + '%', str(r2 * 100) + '%'])
    return mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table

mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_forecasts(actual_test, pre_test, n_out)
print(table)  # 显示预测指标数值

# %%
from matplotlib import rcParams
import matplotlib.dates as mdates
from datetime import datetime

config = {
    "font.family": 'serif',
    "font.size": 10,  # 相当于小四大小
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ['Times New Roman'],  # Times New Roman
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)

# %%
plt.ion()
for ii in range(n_out):
    plt.rcParams['axes.unicode_minus'] = False
    try:
        plt.style.use('cyberpunk')
    except:
        plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 2), dpi=300)

    # 横轴尽量使用日期
    try:
        dates = pd.to_datetime(test_months)
        x = dates
        use_dates = True
    except:
        x = range(1, len(actual_test) + 1)
        use_dates = False

    plt.plot(x, pre_test[:, ii], linestyle="--", linewidth=0.5, label='predict')
    plt.plot(x, actual_test[:, ii], linestyle="-", linewidth=0.5, label='Real')

    plt.rcParams.update({'font.size': 5})
    plt.legend(loc='upper right', frameon=False)

    if use_dates:
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
        plt.xticks(rotation=0, ha='center')
        plt.xlabel("Time (Month)", fontsize=5)
    else:
        plt.xticks(x[::int((len(actual_test) + 1))])
        plt.xlabel("Sample points", fontsize=5)

    plt.tick_params(labelsize=5)
    plt.ylabel("rx5day (mm)", fontsize=5)

    if n_out == 1:
        plt.title(f"The prediction result of LSTM :\nMAPE: {mape(actual_test[:, ii], pre_test[:, ii]):.2f} %")
    else:
        plt.title(f"{ii + 1} step of LSTM prediction\nMAPE: {mape(actual_test[:, ii], pre_test[:, ii]):.2f} %")

    plt.tight_layout()

plt.ioff()
plt.show()

# ================== rx5day 3-class classification (Non-flood / Slight flood / Flood) ==================
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

assert n_out == 1, "This post-processing expects n_out=1."

# 1) Flatten vectors
y_pred_rx5day = pre_test.reshape(-1)      # predicted rx5day
y_true_rx5day = actual_test.reshape(-1)   # true rx5day
months = pd.to_datetime(pd.Series(test_months))

# 2) Thresholds: <50 = Non-flood, 50–100 = Slight flood, ≥100 = Flood
def rx5day_to_label(x: float) -> int:
    if x < 50:
        return 0  # Non-flood
    elif x < 100:
        return 1  # Slight flood
    else:
        return 2  # Flood

label_names = {0: "Non-flood", 1: "Slight flood", 2: "Flood"}

y_true_lbl = np.array([rx5day_to_label(v) for v in y_true_rx5day], dtype=int)
y_pred_lbl = np.array([rx5day_to_label(v) for v in y_pred_rx5day], dtype=int)

# 3) Metrics
acc  = accuracy_score(y_true_lbl, y_pred_lbl)
err  = 1.0 - acc
f1m  = f1_score(y_true_lbl, y_pred_lbl, average='macro')
prec = precision_score(y_true_lbl, y_pred_lbl, average='macro', zero_division=0)
rec  = recall_score(y_true_lbl, y_pred_lbl, average='macro', zero_division=0)

labels_order = [0, 1, 2]
target_names = [label_names[i] for i in labels_order]
cm   = confusion_matrix(y_true_lbl, y_pred_lbl, labels=labels_order)

print("\n=== rx5day classification (thresholds: <50=Non-flood, 50–100=Slight flood, ≥100=Flood) ===")
print(f"Accuracy: {acc:.4f}")
print(f"Error rate: {err:.4f}")
print(f"Macro-averaged F1: {f1m:.4f}")
print(f"Macro-averaged Precision: {prec:.4f}")
print(f"Macro-averaged Recall: {rec:.4f}")

print("\n[Confusion matrix] Row = True, Col = Pred (0=Non-flood, 1=Slight flood, 2=Flood):")
print(cm)

print("\n[Classification report]")
print(classification_report(
    y_true_lbl, y_pred_lbl,
    labels=labels_order,
    target_names=target_names,
    digits=4,
    zero_division=0
))

# 4) Export per-month classification
out_df = pd.DataFrame({
    "Month": months.values,
    "rx5day_true": y_true_rx5day,
    "rx5day_pred": y_pred_rx5day,
    "label_true": y_true_lbl,
    "label_pred": y_pred_lbl,
})
out_df["label_true_txt"] = out_df["label_true"].map(label_names)
out_df["label_pred_txt"] = out_df["label_pred"].map(label_names)

csv_path = "LSTM_Flood_classification_results.csv"
out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\nExported monthly 3-class results: {csv_path}")
# ====================================================================
