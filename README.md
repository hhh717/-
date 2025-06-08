# 基于随机森林学生成绩多因素预测模型

## 项目概述

本项目基于Kaggle的学生习惯与学业表现数据集，通过机器学习方法预测学生考试成绩，识别影响学业表现的关键因素，并提供个性化的学习干预建议。系统实现了完整的数据分析流程：

1. **数据获取与预处理**：下载数据集并处理缺失值、异常值
2. **特征工程**：构建10个创新性的综合指标（如学习专注指数、健康平衡指数等）
3. **模型训练与评估**：比较4种回归模型性能（随机森林、XGBoost等）
4. **结果解释**：使用SHAP值分析关键特征影响
5. **干预建议**：识别低分学生并提供针对性改进建议

## 环境配置

### 系统要求
- Python 3.7+
- pip包管理工具

### 安装依赖
```bash
pip install opendatasets numpy pandas matplotlib seaborn scikit-learn xgboost shap
```

### 额外配置
- 需要Kaggle账户下载数据集（运行时会提示输入Kaggle凭据）
- 确保系统支持中文字体显示（如使用中文标签）

## 使用说明

### 运行完整分析
```python
python student_performance_analysis.py
```

### 核心功能模块

#### 1. 数据获取与预处理
```python
# 下载数据集
od.download("https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset")

# 处理缺失值
df.fillna(mode_val, inplace=True)  # 分类变量
df.fillna(median_val, inplace=True)  # 数值变量

# 处理异常值
df = handle_outliers(df, feature)  # 对每个特征应用IQR方法
```

#### 2. 特征工程（10个创新指标）
```python
# 学习专注指数
df['learning_focus_index'] = (study_hours * time_management * internet_quality)

# 健康平衡指数
df['health_balance_index'] = (0.3*log(sleep) + 0.2*sqrt(diet) + ...)

# 学业压力指数（基于倒U型理论）
df['academic_stress_index'] = 10 * exp(-0.5*(stress - 6)**2)

# 时间分配健康度（基于理想比例4:3:3）
df['time_balance'] = 1 - euclidean_dist(实际比例, 理想比例)
```

#### 3. 模型训练与评估
```python
# 初始化模型集合
models = {
    "梯度提升": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "随机森林": RandomForestRegressor(),
    '决策树': DecisionTreeRegressor()
}

# 训练评估模型
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    # 计算RMSE, MAE, R²
```

#### 4. 结果解释与可视化
```python
# 特征重要性分析
plt.bar(range(len(importances)), importances[indices])

# SHAP值解释
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_sampled)
shap.summary_plot(shap_values, X_test_sampled)
```

#### 5. 干预建议生成
```python
def identify_intervention_points(df):
    low_performers = df[df['exam_score'] < 70]
    for student in low_performers:
        issues = []
        if student['digital_distraction_index'] > 0.7:
            issues.append("数字干扰过高")
        if student['sleep_hours'] < 6:
            issues.append("睡眠不足")
        # ...其他条件判断
    return intervention_candidates
```

## 关键结果

### 模型性能比较
| 模型 | RMSE | MAE | R² |
|------|------|-----|----|
| 随机森林 | 4.42 | 3.51 | 0.85 |
| XGBoost | 4.47 | 3.27 | 0.84 |
| 梯度提升 | 5.54 | 4.51 | 0.76 |
| 决策树 | 9.38 | 7.46 | 0.34 |

### TOP 5关键特征
1. 学业进步潜力指数
2. 时间管理能力
3. 出勤率
4. 学习动机水平
5. 每日学习时长
   
## 预测误差可视化
![预测vs实际](https://github.com/user-attachments/assets/b2495bb3-e4ce-499c-b7e0-555fb895789b)

![预测误差分布](https://github.com/user-attachments/assets/18a2924e-2ecf-465f-adb5-91ed75730afd)

![学生问题分布图](https://github.com/user-attachments/assets/447344c3-bc91-415f-adcf-f3a250bc3ce5)


## 文件结构
```
student_performance_analysis/
├── data/  # 数据集存储位置
│   └── enhanced_student_habits_performance_dataset.csv
├── outputs/  # 生成的可视化结果
│   ├── feature_importance.png
│   ├── prediction_vs_actual.png
│   ├── error_distribution.png
│   └── intervention_distribution.png
├── student_performance_analysis.py  # 主程序
└── README.md  # 本文档
```

## 定制化扩展
1. **调整干预阈值**：修改`identify_intervention_points`函数中的成绩阈值（默认70分）
2. **增加新特征**：在特征工程部分添加新的综合指标
3. **尝试新模型**：在模型字典中添加新的回归模型
4. **优化参数**：使用GridSearchCV进行超参数调优

## 注意事项
1. 首次运行需要Kaggle账户凭据以下载数据集
2. 确保系统已安装中文字体（如SimHei）以正确显示中文标签
3. SHAP分析可能需要较长时间


shap >=0.41.0


3.数据说明



https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset
