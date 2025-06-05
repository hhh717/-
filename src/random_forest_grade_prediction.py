# 导入所有必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  

# ======================
# 1. 数据预处理
# ======================

# 数据获取与基本情况
# 注意：实际使用时需要Kaggle账号和API密钥
od.download("https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset")
file =('student-habits-and-academic-performance-dataset/enhanced_student_habits_performance_dataset/enhanced_student_habits_performance_dataset.csv')
df= pd.read_csv(file)
df.head()


print("数据集形状:", df.shape)
print("\n数据集前5行:")
display(df.head())

print("\n数据集描述性统计:")
display(df.describe())

# 缺失值处理
print("\n原始缺失值统计:")
print(df.isnull().sum())

# 分类变量使用众数填充
categorical_cols = ['diet_quality', 'parental_education_level', 'study_environment', 'learning_style']
for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)

# 数值变量使用中位数填充
numerical_cols = ['time_management_score', 'sleep_hours', 'attendance_percentage', 
                 'social_media_hours', 'netflix_hours', 'exercise_frequency']
for col in numerical_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

print("\n处理后的缺失值统计：")
print(df.isnull().sum())

# 异常值处理
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 将异常值替换为边界值
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# 处理数值特征的异常值
numerical_features = ['study_hours_per_day', 'social_media_hours', 'netflix_hours', 
                     'sleep_hours', 'exercise_frequency', 'social_activity',
                     'screen_time', 'attendance_percentage', 'previous_gpa',
                     'stress_level', 'motivation_level', 'exam_anxiety_score',
                     'time_management_score', 'exam_score']

for feature in numerical_features:
    df = handle_outliers(df, feature)

# 类型转换与编码
# 二元变量转换为0/1
binary_cols = ['part_time_job', 'extracurricular_participation', 
              'access_to_tutoring', 'dropout_risk']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# 分类变量编码
categorical_features = ['major', 'gender', 'study_environment', 'learning_style']
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le

# 有序分类变量映射
diet_mapping = {'Poor': 1, 'Fair': 2, 'Good': 3}
internet_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
family_income_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
parental_edu_mapping = {
    'High School': 1, 
    'Some College': 2, 
    'Bachelor': 3,
    'Master': 4,
    'PhD': 5
}

df['diet_quality'] = df['diet_quality'].map(diet_mapping)
df['internet_quality'] = df['internet_quality'].map(internet_mapping)
df['family_income_range'] = df['family_income_range'].map(family_income_mapping)
df['parental_education_level'] = df['parental_education_level'].map(parental_edu_mapping)

print("\n预处理后的数据集前5行:")
display(df.head())

# ======================
# 2. 特征工程（基于教育心理学理论）
# ======================

# 1. 学习专注指数-基于认知负荷理论
df['learning_focus_index'] = (
    df['study_hours_per_day'] * 
    (df['time_management_score'] / 10) *
    (df['internet_quality'] / 3)
)

# 2. 数字分心指数-基于注意力分散理论
df['digital_distraction_index'] = (
    (df['social_media_hours']**1.5 + df['netflix_hours']**1.5) / 
    (df['study_hours_per_day'] + 1)  # 相对于学习时间的比例
)

# 3. 健康平衡指数 - 基于自我决定理论
df['health_balance_index'] = (
    0.3 * np.log1p(df['sleep_hours']) +          # 睡眠收益递减
    0.2 * (df['diet_quality']**0.5) +           # 饮食质量平方根转换
    0.3 * np.tanh(df['exercise_frequency']/2) + # 运动饱和效应
    0.2 * df['mental_health_rating']/10
)

# 4. 学业压力指数-基于压力-成绩倒U型关系理论
def quadratic_stress_effect(stress, optimal=6, scale=0.5):
    return 10 * np.exp(-scale * (stress - optimal)**2)

df['academic_stress_index'] = quadratic_stress_effect(df['stress_level'])

# 5. 学习动机综合指数-基于期望价值理论
df['motivation_composite'] = (
    0.5 * (df['motivation_level'] / 10) +
    0.3 * (df['parental_support_level'] / 10) -
    0.2 * (df['exam_anxiety_score'] / 10)
)

# 6. 进步潜力-基于增长思维理论
df['gpa_growth_potential'] = (
    0.35 * (df['previous_gpa'] / 4.0) +
    0.35 * (df['attendance_percentage'] / 100) +
    0.3 * (df['time_management_score'] / 10)
)

# 7. 学习效率指数-基于认知心理学
df['learning_efficiency'] = (
    df['previous_gpa'] * 
    (df['time_management_score']/10) * 
    (df['learning_focus_index']/df['learning_focus_index'].max())
) / (df['study_hours_per_day'] + 1)

# 8. 学习韧性指数-应对挫折能力
df['academic_resilience'] = 0.6*df['motivation_level'] + 0.4*df['mental_health_rating'] - 0.3*df['exam_anxiety_score']

# 9. 动机与专注交互特征
df['focus_motivation_interaction'] = df['learning_focus_index'] * df['motivation_composite']

# 10. 时间分配健康度 - 基于时间管理理论
# 理想比例：学习:娱乐:睡眠=4:3:3
ideal_ratio = np.array([0.4, 0.3, 0.3])
actual_ratio = np.column_stack([
    df['study_hours_per_day'] / 24,
    (df['social_media_hours'] + df['netflix_hours']) / 24,
    df['sleep_hours'] / 24
])
euclidean_dist = np.sqrt(np.sum((actual_ratio - ideal_ratio)**2, axis=1))
df['time_balance'] = 1 - euclidean_dist

# 展示新特征
print("\n特征工程后的新特征示例:")
new_features = [
    'learning_focus_index','digital_distraction_index',
    'time_balance', 'academic_stress_index','health_balance_index',
    'motivation_composite', 'gpa_growth_potential', 'learning_efficiency',
    'academic_resilience', 'focus_motivation_interaction'
]
display(df[new_features].head())

# 特征相关性分析
plt.figure(figsize=(20, 15))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('特征相关性热力图')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

# 新特征与目标变量的相关性
corr_with_target = df[new_features + ['exam_score']].corr()['exam_score'][:-1].sort_values()

plt.figure(figsize=(10, 6))
corr_with_target.plot(kind='barh', color='skyblue')
plt.title('新特征与考试成绩的相关性')
plt.xlabel('相关系数')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=300)
plt.show()

# 确保没有NaN值
print("\n最终数据缺失值检查:")
print(df.isnull().sum())



# ======================
# 3. 模型构建与评估
# ======================

# 数据集准备
X = df.drop(['student_id','exam_score'], axis=1)
y = df['exam_score']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 初始化模型
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'SVR': SVR(),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# 训练和评估模型
results = []
for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 交叉验证
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_r2_mean = np.mean(cv_scores)
    
    results.append({
        'Model': name,
        'MSE': mse,
        'R²': r2,
        'CV R² Mean': cv_r2_mean
    })
    
    print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}, CV R²: {cv_r2_mean:.4f}")

# 转换为DataFrame
results_df = pd.DataFrame(results)

# ======================
# 4. 模型比较与优化
# ======================

# 可视化模型性能
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
sns.barplot(x='Model', y='R²', data=results_df, palette='viridis')
plt.title('模型R²分数比较')
plt.ylim(0.7, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(2, 1, 2)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('模型MSE比较')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()

# 随机森林参数优化
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=1, scoring='r2')
grid_search.fit(X_train, y_train)

print("\n最佳参数:", grid_search.best_params_)
print("最佳R²分数:", grid_search.best_score_)

# 使用优化后的模型
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
final_r2 = r2_score(y_test, y_pred_rf)
final_mse = mean_squared_error(y_test, y_pred_rf)
print(f"优化后的随机森林 - MSE: {final_mse:.4f}, R²: {final_r2:.4f}")

# 特征重要性分析
feature_importances = best_rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
plt.title('Top 15 特征重要性')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# ======================
# 5. 结果可视化与解释
# ======================

# 预测值与实际值对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('实际成绩')
plt.ylabel('预测成绩')
plt.title('实际成绩 vs 预测成绩')
plt.grid(True)
plt.savefig('actual_vs_predicted.png', dpi=300)
plt.show()

# 误差分布
errors = y_test - y_pred_rf
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=30)
plt.xlabel('预测误差')
plt.ylabel('频率')
plt.title('预测误差分布')
plt.axvline(x=0, color='r', linestyle='--')
plt.grid(True)
plt.savefig('error_distribution.png', dpi=300)
plt.show()

# 关键特征与成绩关系
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.scatterplot(x='learning_focus_index', y='exam_score', data=df, ax=axes[0, 0])
sns.scatterplot(x='digital_distraction_index', y='exam_score', data=df, ax=axes[0, 1])
sns.scatterplot(x='health_balance_index', y='exam_score', data=df, ax=axes[1, 0])
sns.scatterplot(x='time_balance', y='exam_score', data=df, ax=axes[1, 1])
plt.tight_layout()
plt.savefig('feature_relationships.png', dpi=300)
plt.show()

# 最终报告
print("\n================ 最终模型报告 ================")
print(f"最佳模型: 优化后的随机森林 (R²={final_r2:.4f}, MSE={final_mse:.4f})")
print("关键影响因素:")
for i, row in importance_df.head(5).iterrows():
    print(f"{row['Feature']}: {row['Importance']*100:.2f}%")
