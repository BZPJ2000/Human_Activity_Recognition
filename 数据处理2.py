import pandas as pd
from sklearn.model_selection import train_test_split

# 数据加载
file_path = r'E:\A_workbench\A-lab\25-2-4\Human-Activity-Recognition-master\my_data\混合position.csv'
df = pd.read_csv(file_path)

# ====== 数据预处理 ======
# 1. 处理Subject列
subject_mapping = {
    'Masonry1': 1, 'Masonry2': 2, 'Carpenter1': 3,
    'Carpenter2': 4, 'Rebar1': 5, 'Rebar2': 6, 'Rebar3': 7
}
df['Subject'] = df['Subject'].map(subject_mapping)

# 2. 筛选Label为1-5的数据
# 过滤 Label_2 中大于8的值
df = df[df['Label'] <= 8]

# 定义标签映射函数
def transform_label(x):
    if x in [1, 2]:
        return x
    elif x in [3, 4, 6, 7]:
        return 3
    elif x == 5:
        return 4
    elif x == 8:
        return 5
    else:
        return x

# 应用映射函数到 Label_2 列
df['Label'] = df['Label'].apply(transform_label)

# 3. 创建Action列
label_to_action = {
    1: 'Standing', 2: 'Walking', 3: 'Transitional movements',
    4: 'Bending', 5: 'Squatting'
}
df['Action'] = df['Label'].map(label_to_action)

# 4. 调整列顺序
base_columns = [col for col in df.columns if col not in ['Subject', 'Label', 'Action']]
df = df[base_columns + ['Subject', 'Label', 'Action']]

# 5. 重命名列
df = df.rename(columns={
    'Subject': 'subject',
    'Label': 'Activity',
    'Action': 'ActivityName'
})

# ====== 数据分割方式选择 ======
# （请根据需要注释/取消注释其中一种分割方式）

# ====== 方式1：按用户分割 ======
# 训练集用户：1-5（Masonry1, Masonry2, Carpenter1, Carpenter2, Rebar1）
# 测试集用户：6-7（Rebar2, Rebar3）
train_df = df[df['subject'].isin([1, 2, 3, 4, 5])]
test_df = df[df['subject'].isin([6, 7])]

# ====== 方式2：按比例随机分割（原方法） ======
# train_df, test_df = train_test_split(
#     df,
#     test_size=0.3,
#     random_state=42,
#     stratify=df['Activity']  # 保持类别比例
# )

# ====== 数据验证 ======
# 打印分割结果（可选）
print("==== 数据分割验证 ====")
print(f"训练集用户分布:\n{train_df['subject'].value_counts()}")
print(f"\n测试集用户分布:\n{test_df['subject'].value_counts()}")
print(f"\n训练集样本数: {len(train_df)}")
print(f"测试集样本数: {len(test_df)}")

# ====== 新增标签分布分析代码 ======
print("\n==== 用户活动分布分析 ====")

# 使用交叉表统计用户-活动分布（绝对数量）
distribution_abs = pd.crosstab(
    index=df['subject'],
    columns=df['ActivityName'],
    margins=True,  # 添加总计
    margins_name="Total"
)

# 使用交叉表统计用户-活动分布（百分比）
distribution_pct = pd.crosstab(
    index=df['subject'],
    columns=df['ActivityName'],
    normalize='index'  # 按行标准化
).round(4) * 100  # 转换为百分比

# 打印格式化输出
print("\n【绝对数量分布】")
print(distribution_abs)

print("\n\n【百分比分布（按用户）】")
print(distribution_pct.applymap(lambda x: f"{x:.2f}%"))

# 可选：训练/测试集分布对比（如果使用方式1分割）
if 'train_df' in locals():
    print("\n\n==== 训练集活动分布 ====")
    print(pd.crosstab(train_df['subject'], train_df['ActivityName'], margins=True))

    print("\n==== 测试集活动分布 ====")
    print(pd.crosstab(test_df['subject'], test_df['ActivityName'], margins=True))

import matplotlib.pyplot  as plt

# 按用户分布的绝对数量可视化
distribution_abs.drop('Total',  axis=1).plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    title='Activity Distribution by Subject (Absolute Counts)'
)
plt.show()

# 百分比分布可视化
distribution_pct.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    title='Activity Distribution by Subject (Percentage)'
)
plt.ylabel('Percentage  (%)')
plt.show()









# ====== 数据保存 ======
# train_output = 'train_dataset.xlsx'
# test_output = 'test_dataset.xlsx'
#
# train_df.to_excel(train_output, index=False)
# test_df.to_excel(test_output, index=False)
#
# print(f"训练集已保存至 {train_output}（{len(train_df)} 条）")
# print(f"测试集已保存至 {test_output}（{len(test_df)} 条）")