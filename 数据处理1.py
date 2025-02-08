import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

# 加载数据
print("开始加载数据...")
all_data = pd.read_excel(r'E:\A_workbench\A-lab\25-2-4\Human-Activity-Recognition-master\my_data\On-site data.xlsx',
                         sheet_name=None)

# 查看特定sheet的形状
specific_sheet_name = 'Carpenter1'
if specific_sheet_name in all_data:
    print(f"Shape of {specific_sheet_name}: {all_data[specific_sheet_name].shape}")
else:
    print(f"{specific_sheet_name} 不在数据中。")


def transform_value(x):
    """重新映射标签"""
    if x in [1, 2]:
        return x
    elif x in [3, 4, 6, 7]:
        return 3
    elif x == 5:
        return 4
    elif x == 8:
        return 5
    return x


def extract_statistical_features(data):
    """提取统计特征"""
    features = {}
    for i in range(19):
        col = data.iloc[:, i]
        features[f'mean_{i + 1}'] = col.mean()
        features[f'std_{i + 1}'] = col.std()
        features[f'skew_{i + 1}'] = stats.skew(col)
        features[f'kurtosis_{i + 1}'] = stats.kurtosis(col)
        features[f'max_{i + 1}'] = col.max()
        features[f'min_{i + 1}'] = col.min()
        features[f'q25_{i + 1}'] = col.quantile(0.25)
        features[f'q75_{i + 1}'] = col.quantile(0.75)
        features[f'iqr_{i + 1}'] = features[f'q75_{i + 1}'] - features[f'q25_{i + 1}']
    return features


def extract_frequency_features(data):
    """提取频域特征"""
    features = {}
    for i in range(19):
        col = data.iloc[:, i].to_numpy()
        fft_vals = np.abs(fft(col))
        features[f'fft_mean_{i + 1}'] = np.mean(fft_vals)
        features[f'fft_std_{i + 1}'] = np.std(fft_vals)
        features[f'fft_max_{i + 1}'] = np.max(fft_vals)
        features[f'fft_power_{i + 1}'] = np.sum(fft_vals * fft_vals) / len(fft_vals)
    return features


def extract_correlation_features(data):
    """提取相关性特征"""
    features = {}
    for i in range(4):
        for j in range(i + 1, 4):
            sensor1 = data.iloc[:, i * 3:(i + 1) * 3]
            sensor2 = data.iloc[:, j * 3:(j + 1) * 3]
            corr = np.corrcoef(sensor1.T, sensor2.T)
            features[f'corr_{i + 1}_{j + 1}'] = np.mean(np.abs(corr))
    return features


def feature_engineering(df):
    """优化后的特征工程主函数"""
    new_df = pd.concat([df.iloc[:, 1:20], df.iloc[:, -1]], axis=1)
    new_df['Label_2'] = pd.to_numeric(new_df['Label_2'], errors='coerce').ffill()
    new_df = new_df[new_df[new_df.columns[-1]].between(1, 8)]
    new_df['Label_2'] = new_df['Label_2'].apply(transform_value)

    all_features = []
    labels = []

    window_size = 50
    stride = 25

    for i in range(0, len(new_df) - window_size + 1, stride):
        window = new_df.iloc[i:i + window_size, :19]
        label = new_df.iloc[i + window_size - 1, -1]

        features = {}
        features.update(extract_statistical_features(window))
        features.update(extract_frequency_features(window))
        features.update(extract_correlation_features(window))

        all_features.append(features)
        labels.append(label)

    X = pd.DataFrame(all_features)
    y = pd.Series(labels)

    return X, y


# 处理每个人的特征和标签
person_features = {}
person_labels = {}

for person_id, df in all_data.items():
    print(f"正在处理 {person_id} 的数据...")
    X, y = feature_engineering(df)
    person_features[person_id] = X
    person_labels[person_id] = y
    print(X.shape)
    print(y.shape)

# 分割训练集和测试集
all_persons = list(person_features.keys())
np.random.seed(42)
train_persons = np.random.choice(all_persons, size=4, replace=False)
test_persons = [p for p in all_persons if p not in train_persons]

print("\n训练集使用的人员：", train_persons)
print("测试集使用的人员：", test_persons)

X_train = pd.concat([person_features[p] for p in train_persons])
y_train = pd.concat([person_labels[p] for p in train_persons])
X_test = pd.concat([person_features[p] for p in test_persons])
y_test = pd.concat([person_labels[p] for p in test_persons])

# 处理缺失值
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# 特征选择
selector = SelectFromModel(Lasso(alpha=0.1))
selector.fit(X_train, y_train)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

selected_columns = X.columns[selector.get_support()]
X_train_selected = pd.DataFrame(X_train_selected, columns=selected_columns)
X_test_selected = pd.DataFrame(X_test_selected, columns=selected_columns)

# 保存数据
X_train_selected.to_csv('train_features.csv', index=False)
y_train.to_csv('train_labels.csv', index=False)

X_test_selected.to_csv('test_features.csv', index=False)
y_test.to_csv('test_labels.csv', index=False)

# 验证数据完整性
train_features = pd.read_csv('train_features.csv')
train_labels = pd.read_csv('train_labels.csv')

test_features = pd.read_csv('test_features.csv')
test_labels = pd.read_csv('test_labels.csv')

print("训练集特征是否有缺失:", train_features.isnull().values.any())
print("测试集特征是否有缺失:", test_features.isnull().values.any())

