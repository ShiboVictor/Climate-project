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
dataset = pd.read_csv("THAILAND-FloodPrediction.csv", encoding='UTF-8')
# 使用pandas模块的read_csv函数读取名为"THAILAND-FloodPrediction.csv"的文件。
# 参数'encoding'设置为'UTF-8'。
# 读取的数据被存储在名为'dataset'的DataFrame变量中。
print(dataset)  # 显示dataset数据

# 保存月份信息用于后续绘图
months_data = dataset['Month'].values  # 保存月份信息

# %%
dataset_vmd = pd.read_excel("CEEMDAN-THAILAND-RX5DAY.xlsx")
# 使用pandas模块的read_excel函数读取名为"CEEMDAN-THAILAND-RX5DAY.xlsx"的文件。
values_vmd = dataset_vmd.values
values_vmd = values_vmd.astype('float32')
values = dataset.values[:, 1:]  # 只取第2列数据，要写成1:2；只取第3列数据，要写成2:3，取第2列之后(包含第二列)的所有数据，写成 1：
# 从dataset DataFrame中提取数据。
# dataset.values将DataFrame转换为numpy数组。
# [:,1:]表示选择所有行（:）和从第二列到最后一列（1:）的数据。
# 这样做通常是为了去除第一列，这在第一列是索引或不需要的数据时很常见。

# %%
# 确保所有数据是浮动的
values = values.astype('float32')


# 将values数组中的数据类型转换为float32。
# 这通常用于确保数据类型的一致性，特别是在准备输入到神经网络模型中时。

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
    # 创建模型的输入层，输入的形状为时间步和特征数
    lstm = LSTM(128, return_sequences=False)(inputs)
    # 添加一个LSTM层，其中有128个神经元
    # return_sequences=False表示LSTM层只返回最后一个时间步的输出。
    outputs = Dense(output_dim)(lstm)
    # 创建一个全连接层，神经元的数量等于输出维度
    model = Model(inputs=inputs, outputs=outputs)
    # 创建模型，指定输入和输出
    model.compile(loss='mse', optimizer='Adam')
    # 编译模型，设置损失函数为均方误差（mse），优化器为Adam
    model.summary()
    # 展示模型的结构
    return model
    # 返回构建的模型


# %%
n_in = 3  # 输入前4行的数据
n_out = 1  # 预测未来1步的数据

# 删除第8个cell中的循环代码，初始化基本参数
# 使用更多数据以确保测试集是最近的时间段（2012-2023）
# 数据集总共888行，减去n_in（5）后最多可用883个样本
original_num_samples = 880  # 使用接近全部数据，确保测试集是最新的时间段
scroll_window = 1  # 如果等于1，下一个数据从第二行开始取。如果等于2，下一个数据从第三行开始取
n_train_number = int(original_num_samples * 0.85)  # 取出85%作为训练集（748个），剩余的为测试集（132个）
n_test_number = original_num_samples - n_train_number  # 测试集数量
predicted_data = []
actual_data = []

print(f"计划使用 {original_num_samples} 个样本")
print(f"训练集: {n_train_number} 个样本")
print(f"测试集: {n_test_number} 个样本（约{n_test_number}个月，即{n_test_number / 12:.1f}年）")

# %%
# 添加数据长度检查和对齐
print(f"原始数据形状: {values.shape}")
print(f"CEEMDAN数据形状: {values_vmd.shape}")

# 确保两个数据集长度一致，取较短的长度
min_length = min(values.shape[0], values_vmd.shape[0])
values_aligned = values[:min_length]
values_vmd_aligned = values_vmd[:min_length]
months_aligned = months_data[:min_length]  # 对齐月份数据

# 分离特征和目标变量
features_aligned = values_aligned[:, :-1]  # 原始特征（不包含rx5day）
target_aligned = values_aligned[:, -1]  # 原始目标变量（rx5day）

print(f"对齐后长度: {min_length}")
print(f"特征维度: {features_aligned.shape[1]}")
print(f"IMF分量数: {values_vmd_aligned.shape[1]}")

# %%
# 修正的CEEMDAN-LSTM逻辑：用原始特征预测每个IMF分量
for vmd_num in range(0, len(values_vmd_aligned[0])):
    print(f"处理第 {vmd_num + 1}/{len(values_vmd_aligned[0])} 个CEEMDAN分量...")

    # 获取当前IMF分量作为目标变量（这是关键修改）
    imf = values_vmd_aligned[:, vmd_num]
    imf = imf.reshape(-1, 1)

    # 使用原始特征和IMF分量作为目标构建数据（不是将IMF作为特征）
    combined_data = np.hstack((features_aligned, imf))  # 特征 + IMF目标

    or_dim = features_aligned.shape[1]  # 特征维度（不包含目标）

    # 防越界：收紧 num_samples
    T = combined_data.shape[0]
    max_samples = (T - n_in - n_out) // scroll_window + 1
    if max_samples <= 0:
        raise ValueError(f"序列太短：T={T}, 需要至少 n_in+n_out={n_in + n_out}")
    current_num_samples = min(original_num_samples, max_samples)

    # 数据整理
    res = data_collation(combined_data, n_in, n_out, or_dim, scroll_window, current_num_samples)

    # 把数据集分为训练集和测试集
    combined_data_processed = np.array(res)
    # 将前面处理好的数据转换成numpy数组，方便后续的数据操作。

    current_n_train_number = int(current_num_samples * 0.85)
    # 计算训练集的大小。
    # 设置85%作为训练集
    # int(...) 确保得到的训练集大小是一个整数。
    # 先划分数据集，在进行归一化，这才是正确的做法！
    Xtrain = combined_data_processed[:current_n_train_number, :n_in * or_dim]
    Ytrain = combined_data_processed[:current_n_train_number, n_in * or_dim:]

    Xtest = combined_data_processed[current_n_train_number:, :n_in * or_dim]
    Ytest = combined_data_processed[current_n_train_number:, n_in * or_dim:]

    # 对训练集和测试集进行归一化
    m_in = MinMaxScaler()
    vp_train = m_in.fit_transform(Xtrain)  # 注意fit_transform() 和 transform()的区别
    vp_test = m_in.transform(Xtest)  # 注意fit_transform() 和 transform()的区别

    m_out = MinMaxScaler()
    vt_train = m_out.fit_transform(Ytrain)  # 注意fit_transform() 和 transform()的区别
    vt_test = m_out.transform(Ytest)  # 注意fit_transform() 和 transform()的区别

    vp_train = vp_train.reshape((vp_train.shape[0], n_in, or_dim))
    # 将训练集的输入数据vp_train重塑成三维格式。
    # 结果是一个三维数组，其形状为[样本数量, 时间步长, 特征数量]。

    vp_test = vp_test.reshape((vp_test.shape[0], n_in, or_dim))
    # 将训练集的输入数据vp_test重塑成三维格式。
    # 结果是一个三维数组，其形状为[样本数量, 时间步长, 特征数量]。

    model = lstm_model((n_in, or_dim), vt_train.shape[1])
    # 调用lstm_model函数来建立LSTM模型，传入正确的输入形状和输出维度
    model.fit(
        vp_train, vt_train,
        batch_size=32, epochs=100,
        validation_split=0.0,  # 不用随机切分
        shuffle=False,  # 关键：不要打乱
        verbose=2
    )
    # 训练模型。指定批处理大小为32，训练轮数为100，将0%的数据用作验证集。
    # verbose=2表示在训练过程中会输出详细信息。

    # 作出预测
    yhat = model.predict(vp_test)
    # 使用模型对测试集的输入特征(vp_test)进行预测。
    # yhat是模型预测的输出值。
    yhat = yhat.reshape(current_num_samples - current_n_train_number, n_out)
    # 将预测值yhat重塑为二维数组，以便进行后续操作。
    yy = m_out.inverse_transform(yhat)  # 反归一化

    predicted_data.append(yy)
    actual_data.append(Ytest)  # 这里存储的是未归一化的IMF值

print(f"完成所有 {len(predicted_data)} 个CEEMDAN分量的处理")

# %%
# 修改：确保所有预测结果具有相同的长度
min_length = min(len(pred) for pred in predicted_data)
print(f"统一预测结果长度为: {min_length}")

pre_test = []
# 初始化一个空列表，用于存储每个时间点的预测值总和。
for i in range(0, min_length):
    # 遍历预测结果的每个时间点。
    sum_val = 0
    # 初始化变量 sum_val，用于累加当前时间点的所有预测值。
    for j in range(0, len(predicted_data)):
        # 内层循环，遍历所有预测结果。
        if i < len(predicted_data[j]):
            sum_val = sum_val + predicted_data[j][i]
        # 将当前时间点的预测值累加到 sum_val。
    pre_test.append(sum_val)

pre_test = np.array(pre_test)
pre_test = np.maximum(pre_test, 0)

# %%
# 从原始数据中，取出来真实输出值
# 使用原始数据（包含原始rx5day值）来获取真实测试集
original_combined = np.hstack((features_aligned, target_aligned.reshape(-1, 1)))
original_or_dim = features_aligned.shape[1]  # 原始特征维度（不包含目标变量）

# 计算原始数据的有效样本数
T_original = original_combined.shape[0]
max_samples_original = (T_original - n_in - n_out) // scroll_window + 1
actual_num_samples = min(original_num_samples, max_samples_original)

print(f"原始数据有效样本数: {actual_num_samples}")

res_original = data_collation(original_combined, n_in, n_out, original_or_dim, scroll_window, actual_num_samples)
values_processed = np.array(res_original)

# 重新计算训练集数量
actual_n_train_number = int(actual_num_samples * 0.85)

# 确保与预测结果长度一致
actual_length = min(len(values_processed) - actual_n_train_number, len(pre_test))
actual_test = values_processed[actual_n_train_number:actual_n_train_number + actual_length, n_in * original_or_dim:]
actual_test = actual_test.reshape(actual_length, n_out)

# 同步调整预测结果长度
pre_test = pre_test[:actual_length]
print(f"最终测试集长度: {actual_length}")
print(f"预测值范围: [{pre_test.min():.2f}, {pre_test.max():.2f}]")
print(f"实际值范围: [{actual_test.min():.2f}, {actual_test.max():.2f}]")

# 获取测试集对应的月份
# 数据集总共500个样本，85%训练（425个），15%测试（75个）
# 每个样本用n_in个历史数据预测下一个时间点
# 测试集的第一个预测对应原始数据的第 actual_n_train_number*scroll_window + n_in 个位置
test_start_index = actual_n_train_number * scroll_window + n_in
test_end_index = test_start_index + actual_length
test_months = months_aligned[test_start_index: test_end_index]

# 输出测试集时间范围以验证
if len(test_months) > 0:
    print(f"测试集时间范围: {test_months[0]} 到 {test_months[-1]}")
    print(f"测试集包含 {len(test_months)} 个月份")


# actual_test就是真实的输出值

# %%
def mape(y_true, y_pred):
    # 定义一个计算平均绝对百分比误差（MAPE）的函数。
    record = []
    for index in range(len(y_true)):
        # 遍历实际值和预测值。
        if abs(y_true[index]) > 1e-5:  # 避免除零错误
            temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
            # 计算单个预测的MAPE。
            record.append(temp_mape)
        # 将MAPE添加到记录列表中。
    return np.mean(record) * 100 if record else 0
    # 返回所有记录的平均值，乘以100得到百分比。


# %%
def evaluate_forecasts(Ytest, predicted_data, n_out):
    # 定义一个函数来评估预测的性能。
    mse_dic = []
    rmse_dic = []
    mae_dic = []
    mape_dic = []
    r2_dic = []
    # 初始化存储各个评估指标的字典。
    table = PrettyTable(['测试集指标', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2'])
    for i in range(n_out):
        # 遍历每一个预测步长。每一列代表一步预测，现在是在求每步预测的指标
        actual = [float(row[i]) for row in Ytest]  # 一列列提取
        # 从测试集中提取实际值。
        predicted = [float(row[i]) for row in predicted_data]
        # 从预测结果中提取预测值。
        mse = mean_squared_error(actual, predicted)
        # 计算均方误差（MSE）。
        mse_dic.append(mse)
        rmse = sqrt(mean_squared_error(actual, predicted))
        # 计算均方根误差（RMSE）。
        rmse_dic.append(rmse)
        mae = mean_absolute_error(actual, predicted)
        # 计算平均绝对误差（MAE）。
        mae_dic.append(mae)
        MApe = mape(actual, predicted)
        # 计算平均绝对百分比误差（MAPE）。
        mape_dic.append(MApe)
        r2 = r2_score(actual, predicted)
        # 计算R平方值（R2）。
        r2_dic.append(r2)
        if n_out == 1:
            strr = '预测结果指标：'
        else:
            strr = '第' + str(i + 1) + '步预测结果指标：'
        table.add_row([strr, mse, rmse, mae, str(MApe) + '%', str(r2 * 100) + '%'])

    return mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table
    # 返回包含所有评估指标的字典。


# %%
mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_forecasts(actual_test, pre_test, n_out)
# 调用evaluate_forecasts函数。
# 传递实际值(actual_test)、预测值(pre_test)以及预测的步数(n_out)作为参数。
# 此函数将计算每个预测步长的RMSE、MAE、MAPE和R2值。

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
    # 设置matplotlib的配置，用来正常显示负号。
    # 使用赛博朋克风样式
    try:
        plt.style.use('cyberpunk')
    except:
        plt.style.use('seaborn-darkgrid')
    # 创建一个图形对象，并设置大小为10x2英寸，分辨率为300dpi。
    plt.figure(figsize=(10, 2), dpi=300)

    # 转换月份数据为datetime对象
    try:
        # 尝试将月份字符串转换为datetime对象
        dates = pd.to_datetime(test_months)
        x = dates
        use_dates = True
    except:
        # 如果转换失败，使用原始索引
        x = range(1, len(actual_test) + 1)
        use_dates = False

    # 绘制预测值和真实值
    plt.plot(x, pre_test[:, ii], linestyle="--", linewidth=0.5, label='predict')
    # 绘制预测值的折线图，线型为虚线，线宽为0.5，标签为'predict'。

    plt.plot(x, actual_test[:, ii], linestyle="-", linewidth=0.5, label='Real')
    # 绘制实际值的折线图，线型为直线，线宽为0.5，标签为'Real'。

    plt.rcParams.update({'font.size': 5})  # 改变图例里面的字体大小
    # 更新图例的字体大小。

    plt.legend(loc='upper right', frameon=False)
    # 显示图例，位置在图形的右上角，没有边框。

    if use_dates:
        # 如果使用日期，设置x轴格式
        ax = plt.gca()
        # 设置x轴日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        # 设置x轴刻度间隔
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
        # 旋转日期标签
        plt.xticks(rotation=0, ha='center')
        plt.xlabel("Time (Month)", fontsize=5)
        # 设置x轴标签为"Time (Month)"，字体大小为5。
    else:
        plt.xticks(x[::int((len(actual_test) + 1))])
        # 设置x轴的刻度，每几个点显示一个刻度。
        plt.xlabel("Sample points", fontsize=5)
        # 设置x轴标签为"样本点"，字体大小为5。

    plt.tick_params(labelsize=5)  # 改变刻度字体大小
    # 设置刻度标签的字体大小。

    plt.ylabel("Rx5day (mm)", fontsize=5)
    # 设置y轴标签为"rx5day (mm)"，字体大小为5。

    if n_out == 1:  # 如果是单步预测
        plt.title(f"The prediction result of CEEMDAN-LSTM :\nMAPE: {mape(actual_test[:, ii], pre_test[:, ii]):.2f} %")
    else:
        plt.title(f"{ii + 1} step of CEEMDAN-LSTM prediction\nMAPE: {mape(actual_test[:, ii], pre_test[:, ii]):.2f} %")

    # 调整布局以防止标签被截断
    plt.tight_layout()

    # plt.xlim(xmin=600, xmax=700)  # 显示600-1000的值   局部放大有利于观察
    # 如果需要，可以取消注释这行代码，以局部放大显示600到700之间的值。

    # plt.savefig('figure/预测结果图.png')
    # 如果需要，可以取消注释这行代码，以将图形保存为PNG文件。

plt.ioff()  # 关闭交互模式
plt.show()
# 显示图形。