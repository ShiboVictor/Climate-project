# 调用相关库
import os
import math
import pandas as pd
import openpyxl
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pandas import DataFrame
from pandas import concat
import keras.backend as K
from scipy.io import savemat, loadmat
from sklearn.neural_network import MLPRegressor
from keras.callbacks import LearningRateScheduler
from tensorflow.keras import Input, Model, Sequential
import mplcyberpunk
from qbstyles import mpl_style
import warnings
from prettytable import PrettyTable
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional, LayerNormalization, Input
from tensorflow.keras.models import Model

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module=r"google\.protobuf\.runtime_version")

# %%
dataset = pd.read_csv("China-DroughtPrediction.csv", encoding='UTF-8')
print(dataset)

# 保存月份信息用于后续绘图
months_data = dataset['Month'].values

# %%
dataset_vmd = pd.read_excel("CEEMDAN-CHINA-CDD.xlsx")
values_vmd = dataset_vmd.values.astype('float32')

# 原始数据（去掉 Month 列），最后一列为 cdd
values = dataset.values[:, 1:].astype('float32')

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
def cnn_lstm_model(input_shape, output_dim):
    """
    CNN-LSTM: Conv1D -> MaxPool1D -> Reshape -> LSTM(128, selu) -> Dense(output_dim)
    与原脚本风格一致，唯一变化是模型主体。
    """
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=2, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Reshape((-1, 32))(x)           # 适配 LSTM 的 [time, feat] 形状
    x = LSTM(128, activation='selu', return_sequences=False)(x)
    outputs = Dense(output_dim)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='Adam')
    model.summary()
    return model

# %%
# 与现有脚本保持一致的参数
n_in = 5        # 输入窗口
n_out = 1       # 单步预测
original_num_samples = 880   # 与现有脚本一致；后续会与可用样本数取 min
scroll_window = 1
n_train_number = int(original_num_samples * 0.85)
n_test_number = original_num_samples - n_train_number
predicted_data = []
actual_data = []

print(f"计划使用 {original_num_samples} 个样本")
print(f"训练集: {n_train_number} 个样本")
print(f"测试集: {n_test_number} 个样本（约{n_test_number}个月，即{n_test_number / 12:.1f}年）")

# %%
# 对齐长度（原始特征/目标 与 CEEMDAN 分量）
min_length = min(values.shape[0], values_vmd.shape[0])
values_aligned = values[:min_length]
values_vmd_aligned = values_vmd[:min_length]
months_aligned = months_data[:min_length]

# 特征与原始目标
features_aligned = values_aligned[:, :-1]  # 不含 cdd 的气象因子
target_aligned   = values_aligned[:, -1]   # cdd

print(f"对齐后长度: {min_length}")
print(f"特征维度: {features_aligned.shape[1]}")
print(f"IMF分量数: {values_vmd_aligned.shape[1]}")

# %%
# 逐 IMF 建模（用原始多变量特征 -> 预测该 IMF），预测完再求和重构 cdd
for vmd_num in range(values_vmd_aligned.shape[1]):
    print(f"处理第 {vmd_num + 1}/{values_vmd_aligned.shape[1]} 个CEEMDAN分量...")

    imf = values_vmd_aligned[:, vmd_num].reshape(-1, 1)       # 当前 IMF 作为目标
    combined_data = np.hstack((features_aligned, imf))         # 特征 + IMF(目标)

    or_dim = features_aligned.shape[1]
    T = combined_data.shape[0]
    max_samples = (T - n_in - n_out) // scroll_window + 1
    if max_samples <= 0:
        raise ValueError(f"序列太短：T={T}, 需要至少 n_in+n_out={n_in + n_out}")
    current_num_samples = min(original_num_samples, max_samples)

    res = data_collation(combined_data, n_in, n_out, or_dim, scroll_window, current_num_samples)
    combined_data_processed = np.array(res)

    current_n_train_number = int(current_num_samples * 0.85)
    Xtrain = combined_data_processed[:current_n_train_number, :n_in * or_dim]
    Ytrain = combined_data_processed[:current_n_train_number, n_in * or_dim:]
    Xtest  = combined_data_processed[current_n_train_number:, :n_in * or_dim]
    Ytest  = combined_data_processed[current_n_train_number:, n_in * or_dim:]

    # 归一化
    m_in = MinMaxScaler()
    vp_train = m_in.fit_transform(Xtrain)
    vp_test  = m_in.transform(Xtest)

    m_out = MinMaxScaler()
    vt_train = m_out.fit_transform(Ytrain)
    vt_test  = m_out.transform(Ytest)

    # 重塑为三维 [样本, 时间步, 特征维]
    vp_train = vp_train.reshape((vp_train.shape[0], n_in, or_dim))
    vp_test  = vp_test.reshape((vp_test.shape[0],  n_in, or_dim))

    # 训练 CNN-LSTM（保持与原脚本相同的训练设置：不打乱、无随机验证切分）
    model = cnn_lstm_model((n_in, or_dim), vt_train.shape[1])
    model.fit(
        vp_train, vt_train,
        batch_size=32, epochs=100,
        validation_split=0.0,
        shuffle=False,
        verbose=2
    )

    # 预测并反归一化
    yhat = model.predict(vp_test)
    yhat = yhat.reshape(current_num_samples - current_n_train_number, n_out)
    yy = m_out.inverse_transform(yhat)

    predicted_data.append(yy)
    actual_data.append(Ytest)  # 该 IMF 的真实值（未反归一化，因为 Ytest 此时是原比例）

print(f"完成所有 {len(predicted_data)} 个CEEMDAN分量的处理")

# %%
# 汇总各 IMF 的预测并重构最终预测 cdd（与原脚本相同做法）
min_len = min(len(pred) for pred in predicted_data)
print(f"统一预测结果长度为: {min_len}")
pre_test = []
for i in range(min_len):
    s = 0
    for j in range(len(predicted_data)):
        if i < len(predicted_data[j]):
            s += predicted_data[j][i]
    pre_test.append(s)
pre_test = np.array(pre_test)
pre_test = np.maximum(pre_test, 0)

# %%
# 取出真实 cdd 的测试段（与训练/预测对齐）
original_combined = np.hstack((features_aligned, target_aligned.reshape(-1, 1)))
original_or_dim = features_aligned.shape[1]
T_original = original_combined.shape[0]
max_samples_original = (T_original - n_in - n_out) // scroll_window + 1
actual_num_samples = min(original_num_samples, max_samples_original)
print(f"原始数据有效样本数: {actual_num_samples}")

res_original = data_collation(original_combined, n_in, n_out, original_or_dim, scroll_window, actual_num_samples)
values_processed = np.array(res_original)
actual_n_train_number = int(actual_num_samples * 0.85)

actual_length = min(len(values_processed) - actual_n_train_number, len(pre_test))
actual_test = values_processed[actual_n_train_number:actual_n_train_number + actual_length, n_in * original_or_dim:]
actual_test = actual_test.reshape(actual_length, n_out)

pre_test = pre_test[:actual_length]
print(f"最终测试集长度: {actual_length}")
print(f"预测值范围: [{pre_test.min():.2f}, {pre_test.max():.2f}]")
print(f"实际值范围: [{actual_test.min():.2f}, {actual_test.max():.2f}]")

# 测试集对应的月份
test_start_index = actual_n_train_number * scroll_window + n_in
test_end_index   = test_start_index + actual_length
test_months = months_aligned[test_start_index: test_end_index]
if len(test_months) > 0:
    print(f"测试集时间范围: {test_months[0]} 到 {test_months[-1]}")
    print(f"测试集包含 {len(test_months)} 个月份")

# %%
def mape(y_true, y_pred):
    record = []
    for index in range(len(y_true)):
        if abs(y_true[index]) > 1e-5:
            temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
            record.append(temp_mape)
    return np.mean(record) * 100 if record else 0

# %%
def evaluate_forecasts(Ytest, predicted_data, n_out):
    mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic = [], [], [], [], []
    table = PrettyTable(['测试集指标', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2'])
    for i in range(n_out):
        actual = [float(row[i]) for row in Ytest]
        predicted = [float(row[i]) for row in predicted_data]
        mse = mean_squared_error(actual, predicted); mse_dic.append(mse)
        rmse = sqrt(mean_squared_error(actual, predicted)); rmse_dic.append(rmse)
        mae = mean_absolute_error(actual, predicted); mae_dic.append(mae)
        MApe = mape(actual, predicted); mape_dic.append(MApe)
        r2 = r2_score(actual, predicted); r2_dic.append(r2)
        table.add_row(['预测结果指标：' if n_out == 1 else f'第{i+1}步预测结果指标：',
                       mse, rmse, mae, f"{MApe}%", f"{r2*100}%"])
    return mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table

mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_forecasts(actual_test, pre_test, n_out)
print(table)

# %%
from matplotlib import rcParams
import matplotlib.dates as mdates
from datetime import datetime

config = {
    "font.family": 'serif',
    "font.size": 10,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
    'axes.unicode_minus': False
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

    try:
        dates = pd.to_datetime(test_months)
        x = dates; use_dates = True
    except:
        x = range(1, len(actual_test) + 1); use_dates = False

    plt.plot(x, pre_test[:, ii], linestyle="--", linewidth=0.5, label='predict')
    plt.plot(x, actual_test[:, ii], linestyle="-",  linewidth=0.5, label='Real')

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
    plt.ylabel("CDD (days)", fontsize=5)

    if n_out == 1:
        plt.title(f"The prediction result of CEEMDAN-CNN-LSTM :\nMAPE: {mape(actual_test[:, ii], pre_test[:, ii]):.2f} %")
    else:
        plt.title(f"{ii + 1} step of CEEMDAN-CNN-LSTM prediction\nMAPE: {mape(actual_test[:, ii], pre_test[:, ii]):.2f} %")

    plt.tight_layout()

plt.ioff()
plt.show()

# ================== 追加：CDD 三分类（干旱/轻微干旱/非干旱）评估 ==================
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

assert n_out == 1

# 1) 准备向量（扁平化）
y_pred_cdd = pre_test.reshape(-1)     # 预测 cdd
y_true_cdd = actual_test.reshape(-1)  # 真实 cdd
months = pd.to_datetime(pd.Series(test_months))

# 2) 阈值规则（你指定的三段式）
def cdd_to_label(x: float) -> int:
    if x < 11:
        return 0    # Non-drought
    elif x < 16:
        return 1    # Slight drought
    else:
        return 2    # Drought

label_names = {0: "Non-drought", 1: "Slight drought", 2: "Drought"}

y_true_lbl = np.array([cdd_to_label(v) for v in y_true_cdd], dtype=int)
y_pred_lbl = np.array([cdd_to_label(v) for v in y_pred_cdd], dtype=int)

# 3) 核心指标
acc  = accuracy_score(y_true_lbl, y_pred_lbl)
err  = 1.0 - acc
f1m  = f1_score(y_true_lbl, y_pred_lbl, average='macro')
prec = precision_score(y_true_lbl, y_pred_lbl, average='macro', zero_division=0)
rec  = recall_score(y_true_lbl, y_pred_lbl, average='macro', zero_division=0)
cm   = confusion_matrix(y_true_lbl, y_pred_lbl, labels=[0,1,2])

print("\n=== CDD classification（threshold value：<11=Non-drought, 11–15=Slight drought, ≥16=Drought） ===")
print(f"Accuracy: {acc:.4f}")
print(f"Error rate: {err:.4f}")
print(f"Macro-averaged F1: {f1m:.4f}")
print(f"Macro-averaged Precision: {prec:.4f}")
print(f"Macro-averaged Recall: {rec:.4f}")

print("\n[confusion matrix] Row = real value, column = prediction (0=Non-drought, 1=Slight drought, 2=Drought):")
print(cm)

print("\n[classification_result]")
print(classification_report(
    y_true_lbl, y_pred_lbl,
    target_names=[label_names[0], label_names[1], label_names[2]],
    digits=4,
    zero_division=0
))

# 4) 导出逐月结果
out_df = pd.DataFrame({
    "Month": months.values,
    "cdd_true": y_true_cdd,
    "cdd_pred": y_pred_cdd,
    "label_true": y_true_lbl,
    "label_pred": y_pred_lbl,
})
out_df["label_true_txt"] = out_df["label_true"].map(label_names)
out_df["label_pred_txt"] = out_df["label_pred"].map(label_names)

csv_path = "CEEMDAN_CNN_LSTM_classification_results.csv"
out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n The monthly three-category results have been exported：{csv_path}")