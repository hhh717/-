
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')


# ======================
# 1. 数据预处理
# ======================

# 数据获取与基本情况
df= pd.read_csv("src/enhanced_student_habits_performance_dataset.csv")
df.head()

# 分类变量使用众数填充
categorical_cols = ['diet_quality', 'parental_education_level', 'study_environment', 'learning_style'] \
                   if 'diet_quality' in df.columns else []
for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)

# 数值变量使用中位数填充
numerical_cols = ['time_management_score', 'sleep_hours', 'attendance_percentage', 
                 'social_media_hours', 'netflix_hours', 'exercise_frequency'] \
                 if 'time_management_score' in df.columns else []
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
                     'time_management_score', 'exam_score'] \
                     if 'study_hours_per_day' in df.columns else []
for feature in numerical_features:
    df = handle_outliers(df, feature)

# 类型转换与编码
# 二元变量转换为0/1
binary_cols = ['part_time_job', 'extracurricular_participation', 
              'access_to_tutoring', 'dropout_risk'] \
              if 'part_time_job' in df.columns else []
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0}) if df[col].dtype == 'object' else df[col]

# 分类变量编码
categorical_features = ['major', 'gender', 'study_environment', 'learning_style'] \
                       if 'major' in df.columns else []
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le

# 有序分类变量映射
if 'diet_quality' in df.columns:
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

# ======================
# 2. 特征工程（基于教育心理学理论）
# ======================

# 1. 学习专注指数-基于认知负荷理论
if 'study_hours_per_day' in df.columns and 'time_management_score' in df.columns and 'internet_quality' in df.columns:
    df['learning_focus_index'] = (
        df['study_hours_per_day'] * 
        (df['time_management_score'] / 10) *
        (df['internet_quality'] / 3)
    )

# 2. 数字分心指数-基于注意力分散理论
if 'social_media_hours' in df.columns and 'netflix_hours' in df.columns and 'study_hours_per_day' in df.columns:
    df['digital_distraction_index'] = (
        (df['social_media_hours']**1.5 + df['netflix_hours']**1.5) / 
        (df['study_hours_per_day'] + 1)  # 相对于学习时间的比例
    )

# 3. 健康平衡指数 - 基于自我决定理论
if 'sleep_hours' in df.columns and 'diet_quality' in df.columns and 'exercise_frequency' in df.columns and 'mental_health_rating' in df.columns:
    df['health_balance_index'] = (
        0.3 * np.log1p(df['sleep_hours']) +          # 睡眠收益递减
        0.2 * (df['diet_quality']**0.5) +           # 饮食质量平方根转换
        0.3 * np.tanh(df['exercise_frequency']/2) + # 运动饱和效应
        0.2 * df['mental_health_rating']/10
    )

# 4. 学业压力指数-基于压力-成绩倒U型关系理论
if 'stress_level' in df.columns:
    def quadratic_stress_effect(stress, optimal=6, scale=0.5):
        return 10 * np.exp(-scale * (stress - optimal)**2)
    
    df['academic_stress_index'] = quadratic_stress_effect(df['stress_level'])

# 5. 学习动机综合指数-基于期望价值理论
if 'motivation_level' in df.columns and 'parental_support_level' in df.columns and 'exam_anxiety_score' in df.columns:
    df['motivation_composite'] = (
        0.5 * (df['motivation_level'] / 10) +
        0.3 * (df['parental_support_level'] / 10) -
        0.2 * (df['exam_anxiety_score'] / 10)
    )

# 6. 进步潜力-基于增长思维理论
if 'previous_gpa' in df.columns and 'attendance_percentage' in df.columns and 'time_management_score' in df.columns:
    df['gpa_growth_potential'] = (
        0.35 * (df['previous_gpa'] / 4.0) +
        0.35 * (df['attendance_percentage'] / 100) +
        0.3 * (df['time_management_score'] / 10)
    )

# 7. 学习效率指数-基于认知心理学
if 'previous_gpa' in df.columns and 'time_management_score' in df.columns and 'learning_focus_index' in df.columns and 'study_hours_per_day' in df.columns:
    df['learning_efficiency'] = (
        df['previous_gpa'] * 
        (df['time_management_score']/10) * 
        (df['learning_focus_index']/df['learning_focus_index'].max())
    ) / (df['study_hours_per_day'] + 1)

# 8. 学习韧性指数-应对挫折能力
if 'motivation_level' in df.columns and 'mental_health_rating' in df.columns and 'exam_anxiety_score' in df.columns:
    df['academic_resilience'] = 0.6*df['motivation_level'] + 0.4*df['mental_health_rating'] - 0.3*df['exam_anxiety_score']

# 9. 动机与专注交互特征
if 'learning_focus_index' in df.columns and 'motivation_composite' in df.columns:
    df['focus_motivation_interaction'] = df['learning_focus_index'] * df['motivation_composite']

# 10. 时间分配健康度 - 基于时间管理理论
# 理想比例：学习:娱乐:睡眠=4:3:3
if 'study_hours_per_day' in df.columns and 'social_media_hours' in df.columns and 'netflix_hours' in df.columns and 'sleep_hours' in df.columns:
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
# 仅显示存在的特征
existing_features = [f for f in new_features if f in df.columns]
if existing_features:
    print(df[existing_features].head())

# 特征相关性分析
plt.figure(figsize=(20, 15))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('特征相关性热力图')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

# 新特征与目标变量的相关性
if 'exam_score' in df.columns and existing_features:
    corr_with_target = df[existing_features + ['exam_score']].corr()['exam_score'][:-1].sort_values()

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
if 'student_id' in df.columns and 'exam_score' in df.columns:
    X = df.drop(['student_id','exam_score'], axis=1)
    y = df['exam_score']
else:
    X = df.drop('exam_score', axis=1)
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
    'XGBoost': XGBRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(n_jobs=-1)
}

# 训练和评估模型
results = []
for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 交叉验证
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    cv_r2_mean = np.mean(cv_scores)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'CV R² Mean': cv_r2_mean
    })
    
    print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, CV R²: {cv_r2_mean:.4f}")

# 转换为DataFrame
results_df = pd.DataFrame(results)

# ======================
# 4. 模型比较与优化（使用贝叶斯优化）
# ======================

# 可视化模型性能
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
sns.barplot(x='Model', y='R²', data=results_df, palette='viridis')
plt.title('模型R²分数比较')
plt.ylim(0.0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(2, 1, 2)
sns.barplot(x='Model', y='RMSE', data=results_df, palette='viridis')
plt.title('模型RMSE比较')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()

# 使用贝叶斯优化替代网格搜索
def rf_cv(n_estimators, max_depth, min_samples_split, max_features):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    
    # 使用5折交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')  # 减少cv次数以加快速度
    return np.mean(cv_scores)  # 贝叶斯优化需要最大化这个值

# 定义参数空间
pbounds = {
    'n_estimators': (50, 300),
    'max_depth': (3, 30),
    'min_samples_split': (2, 20),
    'max_features': (0.1, 0.999)  # 特征比例
}

# 创建贝叶斯优化器
optimizer = BayesianOptimization(
    f=rf_cv,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# 执行优化（减少迭代次数以提高速度）
optimizer.maximize(
    init_points=3,  # 初始随机采样点
    n_iter=10,      # 贝叶斯优化迭代次数
)

# 获取最佳参数
best_params = optimizer.max['params']
# 将整数参数转换为整数
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])

print("\n贝叶斯优化最佳参数:", best_params)
print("最佳交叉验证R²分数:", optimizer.max['target'])

# 使用优化后的模型
best_rf = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)
final_r2 = r2_score(y_test, y_pred_rf)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
final_mae = mean_absolute_error(y_test, y_pred_rf)
print(f"优化后的随机森林 - RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}, R²: {final_r2:.4f}")


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
if existing_features:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    top_features = importance_df['Feature'].head(4).values
    
    for i, feature in enumerate(top_features):
        ax = axes[i//2, i%2]
        sns.scatterplot(x=feature, y='exam_score', data=df, ax=ax)
        ax.set_title(f'{feature} vs 考试成绩')
    
    plt.tight_layout()
    plt.savefig('feature_relationships.png', dpi=300)
    plt.show()

# 最终报告
print("\n================ 最终模型报告 ================")
print(f"最佳模型: 贝叶斯优化的随机森林 (R²={final_r2:.4f}, RMSE={final_rmse:.4f})")
print("关键影响因素:")
for i, row in importance_df.head(5).iterrows():
    print(f"{row['Feature']}: {row['Importance']*100:.2f}%")

# 模型性能对比
print("\n所有模型性能对比:")
print(results_df.sort_values('R²', ascending=False))

print("代码执行完毕！")
