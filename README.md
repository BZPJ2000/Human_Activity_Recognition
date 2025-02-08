# Human Activity Recognition

## 一、项目概述
本项目旨在通过对人体活动数据的分析和处理，实现对不同人体活动的准确识别。项目涵盖了数据加载、特征工程、数据预处理、模型训练与评估等多个环节，运用了多种机器学习和深度学习模型进行实验对比。

## 二、数据来源与处理
### 数据加载
从多个Excel文件中加载数据，如`On-site data.xlsx` 和`混合position.csv` ，并使用`pandas`库的`read_excel`和`read_csv`函数进行读取。
```python
all_data = pd.read_excel(r'E:\A_workbench\A-lab\25-2-4\Human-Activity-Recognition-master\my_data\On-site  data.xlsx',  sheet_name=None)
df = pd.read_csv(file_path) 
```
数据预处理
标签映射：对数据中的标签进行重新映射，以便于模型处理。例如，将某些标签值合并或转换为特定的类别。
```python

def transform_value(x):
    if x in [1, 2]:
        return x
    elif x in [3, 4, 6, 7]:
        return 3
    elif x == 5:
        return 4
    elif x == 8:
        return 5
    return x
```
特征工程：提取多种特征，包括统计特征、频域特征和相关性特征。
统计特征：如均值、标准差、偏度、峰度等。
```python
def extract_statistical_features(data):
    features = {}
    for i in range(19):
        col = data.iloc[:,  i]
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
```
- **频域特征**：通过快速傅里叶变换（FFT）提取频域特征。
```python
def extract_frequency_features(data):
    features = {}
    for i in range(19):
        col = data.iloc[:,  i].to_numpy()
        fft_vals = np.abs(fft(col)) 
        features[f'fft_mean_{i + 1}'] = np.mean(fft_vals) 
        features[f'fft_std_{i + 1}'] = np.std(fft_vals) 
        features[f'fft_max_{i + 1}'] = np.max(fft_vals) 
        features[f'fft_power_{i + 1}'] = np.sum(fft_vals  * fft_vals) / len(fft_vals)
    return features
```    
- **相关性特征**：计算不同传感器之间的相关性。
```python
def extract_correlation_features(data):
    features = {}
    for i in range(4):
        for j in range(i + 1, 4):
            sensor1 = data.iloc[:,  i * 3:(i + 1) * 3]
            sensor2 = data.iloc[:,  j * 3:(j + 1) * 3]
            corr = np.corrcoef(sensor1.T,  sensor2.T)
            features[f'corr_{i + 1}_{j + 1}'] = np.mean(np.abs(corr)) 
    return features
```
数据分割：采用两种方式分割数据，按用户分割和按比例随机分割。
按用户分割：指定训练集和测试集用户。
```python
train_df = df[df['subject'].isin([1, 2, 3, 4, 5])]
test_df = df[df['subject'].isin([6, 7])]
- **按比例随机分割**：使用`train_test_split`函数按指定比例分割数据。
```
```python
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df['Activity']
)
```
## 三、模型训练与评估
机器学习模型
逻辑回归（Logistic Regression）：使用GridSearchCV进行超参数调优。
```python

parameters = {'C':[0.01, 0.1, 1, 10, 20, 30], 'penalty':['l2','l1']}
log_reg = linear_model.LogisticRegression()
log_reg_grid = GridSearchCV(log_reg, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
K近邻（KNN）：同样使用GridSearchCV调优超参数。
```

```python

parameters = {'n_neighbors': [1, 10, 11, 20, 30]}
log_knn = KNeighborsClassifier(n_neighbors=6)
log_knn_grid = GridSearchCV(log_knn, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
线性支持向量机（Linear SVC）：进行超参数搜索。
```

```python

parameters = {'C':[0.125, 0.5]}
lr_svc = LinearSVC(tol=0.00005)
lr_svc_grid = GridSearchCV(lr_svc, param_grid=parameters, n_jobs=-1, verbose=1)
核支持向量机（Kernel SVM）：调优C和gamma等超参数。
```

```python

parameters = {'C':[2,8], 'gamma': [0.125, 2]}
rbf_svm = SVC(kernel='rbf')
rbf_svm_grid = GridSearchCV(rbf_svm,param_grid=parameters, n_jobs=-1)
决策树（Decision Trees）：调整max_depth超参数。
```

```python

parameters = {'max_depth':np.arange(3,6,2)} 
dt = DecisionTreeClassifier()
dt_grid = GridSearchCV(dt,param_grid=parameters, n_jobs=-1)
随机森林（Random Forest Classifier）：对n_estimators和max_depth进行调优。
```

```python

params = {'n_estimators': np.arange(10,201,20),  'max_depth':np.arange(3,15,2)} 
rfc = RandomForestClassifier()
rfc_grid = GridSearchCV(rfc, param_grid=params, n_jobs=-1)
梯度提升决策树（Gradient Boosted Decision Trees）：调优max_depth和n_estimators。
```

```python

param_grid = {'max_depth': np.arange(5,6,1),  'n_estimators':np.arange(130,140,10)} 
gbdt = GradientBoostingClassifier()
gbdt_grid = GridSearchCV(gbdt, param_grid=param_grid, n_jobs=-1)
XGBoost：调整max_depth和n_estimators等参数。
```

```python

param_grid_xgb = {'max_depth': np.arange(3,  4, 1), 'n_estimators': np.arange(100,  121, 20)}
xgb_model = XGBClassifier()
xgb_grid = GridSearchCV(xgb_model, param_grid=param_grid_xgb, n_jobs=-1)
LightGBM：调优num_leaves和n_estimators。
```
```python

param_grid_lgb = {'num_leaves': np.arange(31,   41, 5), 'n_estimators': np.arange(100,   121, 20)}
lgb_model = LGBMClassifier() 
lgb_grid = GridSearchCV(lgb_model, param_grid=param_grid_lgb, n_jobs=-1)
CatBoost：调整depth和iterations。
```
```python

param_grid_cat = {'depth': np.arange(4,  5, 1), 'iterations': np.arange(100,   121, 20)}
cat_model = CatBoostClassifier(verbose=0) 
cat_grid = GridSearchCV(cat_model, param_grid=param_grid_cat, n_jobs=-1) 
深度学习模型
LSTM模型：使用PyTorch搭建LSTM模型，用于序列数据处理。
```
```python

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers

        self.lstm  = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.batch_norm  = nn.BatchNorm1d(hidden_dim)
        self.dropout  = nn.Dropout(dropout)
        self.fc  = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  

        h0 = torch.zeros(self.num_layers,  x.size(0),  self.hidden_dim).to(x.device) 
        c0 = torch.zeros(self.num_layers,  x.size(0),  self.hidden_dim).to(x.device) 

        out, (h_n, c_n) = self.lstm(x,  (h0, c0))

        out = out[:, -1, :]

        out = self.batch_norm(out) 
        out = self.dropout(out) 
        out = self.fc(out) 

        return out
模型评估
使用混淆矩阵和准确率等指标评估模型性能。
```
```python

cm = confusion_matrix(y_test,  y_pred)
accuracy = metrics.accuracy_score(y_true=y_test,   y_pred=y_pred)
```
## 四、项目总结
本项目通过对人体活动数据的多方面处理和多种模型的实验，实现了对人体活动的识别。不同模型在准确率和误差率上表现各异，可根据具体需求选择合适的模型。未来可进一步探索更复杂的模型和特征工程方法，以提高识别准确率。

## 五、依赖库
pandas
numpy
scikit-learn
PyTorch
matplotlib
seaborn
xgboost
lightgbm
catboost
七、运行说明
确保安装了所有依赖库。
准备好数据文件，确保路径正确。
运行代码，根据提示查看训练和评估结果。
