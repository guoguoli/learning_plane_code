# Day 1 - 线性代数核心概念与Python实战（调整版）
# 配套代码文件：增强版，包含详细注释、成都特色实现和完整挑战框架

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("Day 1 - 线性代数核心概念与Python实战（调整版）")
print("配套代码文件：增强理论深度，优化挑战框架，强化成都特色")
print("=" * 80)

# ============================================================================
# 第一部分：NumPy高级线性代数实战（增强版）
# ============================================================================

def numpy_advanced_linear_algebra():
    """
    NumPy高级线性代数运算演示（增强版）
    
    功能亮点：
    1. 包含特殊矩阵创建与性质验证
    2. 矩阵分解的完整实现与数学验证
    3. 线性方程组求解的数值稳定性分析
    4. 丰富的数学解释和几何直观
    
    返回：
    dict: 包含矩阵、特征值、特征向量等关键数据
    """
    print("\n🔢 第一部分：NumPy高级线性代数实战（增强版）")
    print("-" * 70)
    
    # 1. 特殊矩阵创建与性质验证
    print("1. 特殊矩阵的创建与数学性质验证:")
    
    # 1.1 单位矩阵：对角线为1，其他为0
    I_3 = np.eye(3)  # 3阶单位矩阵
    print(f"a) 3阶单位矩阵 I₃:\n{I_3}")
    
    # 验证单位矩阵性质：I × I = I，I × A = A × I = A
    A_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    left_mult = I_3 @ A_test  # 左乘
    right_mult = A_test @ I_3  # 右乘
    print(f"   性质验证 - 左乘一致性: {np.allclose(left_mult, A_test)}")
    print(f"   性质验证 - 右乘一致性: {np.allclose(right_mult, A_test)}")
    
    # 1.2 对角矩阵：只有对角线有非零值
    diag_values = [1, 2, 3]
    D = np.diag(diag_values)
    print(f"\nb) 对角矩阵 D=diag({diag_values}):\n{D}")
    
    # 对角矩阵的性质：与向量相乘实现各维度独立缩放
    v = np.array([1, 1, 1])
    scaled_v = D @ v
    print(f"   几何意义: D@[1,1,1]ᵀ = {scaled_v}，实现各维度独立缩放")
    
    # 1.3 正交矩阵：QᵀQ = QQᵀ = I
    np.random.seed(42)
    A_rand = np.random.randn(4, 4)
    Q, R = np.linalg.qr(A_rand)  # QR分解得到正交矩阵
    
    print(f"\nc) 随机正交矩阵 Q (通过QR分解生成):")
    print(f"   形状: {Q.shape}")
    print(f"   前2行:\n{Q[:2]}")
    
    # 验证正交性
    Q_T_Q = Q.T @ Q
    Q_Q_T = Q @ Q.T
    orth_error = np.max(np.abs(Q_T_Q - np.eye(4)))
    print(f"   正交性验证: ||QᵀQ - I||_∞ = {orth_error:.6f}")
    print(f"   几何意义: 正交矩阵保持向量长度和夹角不变")
    
    # 2. 矩阵分解的数学意义与实现
    print("\n2. 矩阵分解的核心应用与数学验证:")
    
    # 2.1 特征值分解：A = VΛV⁻¹，要求A为方阵
    # 使用对称正定矩阵（协方差矩阵的典型形式）
    cov_matrix = np.array([[4, 2], [2, 3]])
    print(f"a) 对称正定矩阵（模拟协方差矩阵）:\n{cov_matrix}")
    
    # 特征值分解
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    print(f"\n   特征值 λ: {eigvals}")
    print(f"   特征向量矩阵 V:\n{eigvecs}")
    
    # 数学验证：AV = VΛ
    AV = cov_matrix @ eigvecs
    V_Lambda = eigvecs @ np.diag(eigvals)
    eig_error = np.linalg.norm(AV - V_Lambda, 'fro')
    print(f"   验证: ||AV - VΛ||_F = {eig_error:.6f}")
    
    # 重构验证：A = VΛV⁻¹
    reconstructed = eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs)
    recon_error = np.linalg.norm(cov_matrix - reconstructed, 'fro')
    print(f"   重构: ||A - VΛV⁻¹||_F = {recon_error:.6f}")
    
    # 几何解释：特征向量指示数据主要变化方向
    print(f"   几何解释: 特征向量v₁={eigvecs[:,0].round(3)}方向方差={eigvals[0]:.2f}")
    print(f"             特征向量v₂={eigvecs[:,1].round(3)}方向方差={eigvals[1]:.2f}")
    
    # 2.2 奇异值分解（SVD）：A = UΣVᵀ，适用于任意矩阵
    print(f"\nb) 奇异值分解（SVD）应用:")
    U, S, Vt = np.linalg.svd(cov_matrix)
    
    print(f"   奇异值 σ: {S}")
    print(f"   U矩阵（左奇异向量）:\n{U}")
    print(f"   Vᵀ矩阵（右奇异向量转置）:\n{Vt}")
    
    # SVD重构验证
    Sigma = np.zeros((2, 2))
    np.fill_diagonal(Sigma, S)
    svd_reconstructed = U @ Sigma @ Vt
    svd_error = np.linalg.norm(cov_matrix - svd_reconstructed, 'fro')
    print(f"   重构误差: ||A - UΣVᵀ||_F = {svd_error:.6f}")
    
    # 3. 线性方程组求解的数值稳定性分析
    print("\n3. 线性方程组的数值稳定性与工程应用:")
    
    # 3.1 良态方程组：条件数较小，求解稳定
    print("a) 良态方程组求解:")
    A_well = np.array([[3, 2], [1, -1]])
    b_well = np.array([5, 1])
    
    cond_well = np.linalg.cond(A_well)
    print(f"   系数矩阵 A:\n{A_well}")
    print(f"   条件数 κ(A) = {cond_well:.4f}（κ<10³为良态）")
    
    # 求解
    x_well = np.linalg.solve(A_well, b_well)
    residual_well = np.linalg.norm(A_well @ x_well - b_well)
    print(f"   解: x = {x_well[0]:.2f}, y = {x_well[1]:.2f}")
    print(f"   残差范数: ||Ax - b||₂ = {residual_well:.6f}")
    
    # 3.2 病态方程组：条件数很大，求解不稳定
    print("\nb) 病态方程组分析（对比演示）:")
    # Hilbert矩阵是典型病态矩阵
    A_ill = np.array([[1, 1/2, 1/3], 
                      [1/2, 1/3, 1/4], 
                      [1/3, 1/4, 1/5]])
    b_ill = np.array([1, 1, 1])
    
    cond_ill = np.linalg.cond(A_ill)
    print(f"   Hilbert矩阵 A（3阶）:\n{A_ill.round(3)}")
    print(f"   条件数 κ(A) = {cond_ill:.2e}（极端病态）")
    
    # 尝试求解（可能产生较大误差）
    try:
        x_ill = np.linalg.solve(A_ill, b_ill)
        residual_ill = np.linalg.norm(A_ill @ x_ill - b_ill)
        print(f"   名义解: {x_ill.round(3)}")
        print(f"   残差范数: {residual_ill:.6f}（尽管残差小，但解可能不可靠）")
    except np.linalg.LinAlgError as e:
        print(f"   求解失败: {e}")
    
    # 4. 矩阵的秩与零空间分析
    print("\n4. 矩阵的秩、零空间及其在AI中的意义:")
    
    # 创建一个秩亏矩阵（模拟线性相关特征）
    B = np.array([[1, 2, 3], 
                  [2, 4, 6],  # 第二行是第一行的2倍
                  [4, 5, 6]])
    
    rank_B = np.linalg.matrix_rank(B)
    print(f"a) 矩阵B（存在线性相关行）:\n{B}")
    print(f"   秩 rank(B) = {rank_B}，秩亏数 = {B.shape[0] - rank_B}")
    
    # 零空间计算（使用SVD）
    U_B, S_B, Vt_B = np.linalg.svd(B)
    
    # 零空间由Vt_B[rank_B:]的行张成
    null_space = Vt_B[rank_B:].T
    nullity = B.shape[1] - rank_B  # 零空间维度
    
    print(f"   零空间维度（零化度）: {nullity}")
    if nullity > 0:
        print(f"   零空间基向量: {null_space[:,0].round(3)}")
        
        # 验证：B × null_space ≈ 0
        verification = B @ null_space
        max_error = np.max(np.abs(verification))
        print(f"   验证: max|B × N| = {max_error:.6f} ≈ 0")
    
    # AI应用意义：秩亏表示特征冗余，零空间表示对输出无影响的输入方向
    print(f"\n   AI应用意义:")
    print(f"   - 秩亏提示特征存在线性相关，可考虑PCA降维")
    print(f"   - 零空间方向对模型输出无贡献，可识别无效特征")
    
    # 返回关键数据供后续使用
    return {
        'identity_matrix': I_3,
        'diagonal_matrix': D,
        'orthogonal_matrix': Q,
        'covariance_matrix': cov_matrix,
        'eigenvalues': eigvals,
        'eigenvectors': eigvecs,
        'singular_values': S,
        'U_matrix': U,
        'Vt_matrix': Vt,
        'rank_deficient_matrix': B,
        'matrix_rank': rank_B,
        'nullity': nullity
    }

# ============================================================================
# 第二部分：Pandas数据处理实战 - 成都教育科技数据集（增强版）
# ============================================================================

def create_chengdu_education_dataset():
    """
    创建模拟的成都教育科技数据集（增强版）
    
    数据特点：
    1. 成都真实中学名称和行政区划
    2. 学科成绩模拟（考虑正态分布和相关性）
    3. 学习行为与心理指标
    4. 成都教育特色字段
    
    返回：
    tuple: (DataFrame, 学科列名列表)
    """
    print("\n📊 第二部分：Pandas数据处理实战 - 成都教育科技数据集（增强版）")
    print("-" * 70)
    
    np.random.seed(42)  # 确保可复现
    
    # 成都教育资源分布（基于真实情况）
    schools_info = {
        '成都七中': {'district': '锦江区', 'tier': '顶尖', 'student_count': 3000},
        '树德中学': {'district': '青羊区', 'tier': '顶尖', 'student_count': 2800},
        '石室中学': {'district': '青羊区', 'tier': '顶尖', 'student_count': 3200},
        '成都外国语学校': {'district': '高新区', 'tier': '一流', 'student_count': 2500},
        '玉林中学': {'district': '武侯区', 'tier': '一流', 'student_count': 2000}
    }
    
    # 生成1000条学生记录
    n_students = 1000
    schools = list(schools_info.keys())
    districts = [schools_info[school]['district'] for school in schools]
    tiers = [schools_info[school]['tier'] for school in schools]
    
    # 按学校层级分配学生比例
    school_probs = [schools_info[school]['student_count'] for school in schools]
    school_probs = np.array(school_probs) / sum(school_probs)
    
    print(f"1. 成都教育资源分布:")
    for school, info in schools_info.items():
        print(f"   - {school}: {info['district']}区，{info['tier']}，约{info['student_count']}名学生")
    
    # 生成学生数据
    data = {
        '学生ID': range(1, n_students + 1),
        '姓名': [f'学生_{i:04d}' for i in range(1, n_students + 1)],
        '性别': np.random.choice(['男', '女'], n_students, p=[0.52, 0.48]),  # 成都中学生性别比例
        '年龄': np.random.choice([15, 16, 17, 18], n_students, p=[0.25, 0.35, 0.30, 0.10]),
        
        # 学校分配（考虑学校层级权重）
        '学校': np.random.choice(schools, n_students, p=school_probs),
        '所在区': np.random.choice(districts, n_students),
        '学校层级': np.random.choice(tiers, n_students),
        
        # 学科成绩生成（考虑学科相关性和学校差异）
        '语文': np.zeros(n_students),
        '数学': np.zeros(n_students),
        '英语': np.zeros(n_students),
        '物理': np.zeros(n_students),
        '化学': np.zeros(n_students),
        
        # 学习行为数据（成都特色）
        '每周学习时长(小时)': np.zeros(n_students),
        '每周辅导班时长(小时)': np.zeros(n_students),  # 成都补习文化
        '教育APP使用指数': np.zeros(n_students),  # 成都教育科技渗透率
        
        # 心理与健康指标
        '学习压力指数': np.zeros(n_students),
        '睡眠质量评分': np.zeros(n_students),
        '课外活动参与度': np.zeros(n_students),  # 成都素质教育特色
    }
    
    # 按学校层级生成差异化的成绩分布
    for i, school in enumerate(schools):
        mask = data['学校'] == school
        n_school_students = np.sum(mask)
        
        if n_school_students == 0:
            continue
            
        # 学校基准水平
        if schools_info[school]['tier'] == '顶尖':
            base_mean = 75
            base_std = 10
            pressure_mean = 70  # 顶尖学校压力较大
        else:
            base_mean = 65
            base_std = 15
            pressure_mean = 60
        
        # 生成相关学科成绩
        # 语文-数学相关系数约0.6，数学-物理约0.7
        cov_matrix = np.array([
            [1.0, 0.6, 0.5, 0.4, 0.3],  # 语文
            [0.6, 1.0, 0.7, 0.6, 0.5],  # 数学
            [0.5, 0.7, 1.0, 0.5, 0.4],  # 英语
            [0.4, 0.6, 0.5, 1.0, 0.8],  # 物理
            [0.3, 0.5, 0.4, 0.8, 1.0]   # 化学
        ])
        
        # 生成多元正态分布成绩
        means = np.array([base_mean, base_mean+3, base_mean-2, base_mean-5, base_mean-7])
        scores = np.random.multivariate_normal(means, cov_matrix * base_std**2, n_school_students)
        
        # 填充数据
        data['语文'][mask] = scores[:, 0].clip(0, 100)
        data['数学'][mask] = scores[:, 1].clip(0, 100)
        data['英语'][mask] = scores[:, 2].clip(0, 100)
        data['物理'][mask] = scores[:, 3].clip(0, 100)
        data['化学'][mask] = scores[:, 4].clip(0, 100)
        
        # 生成行为数据（与成绩相关）
        data['每周学习时长(小时)'][mask] = np.random.normal(30, 8, n_school_students).clip(10, 50)
        data['每周辅导班时长(小时)'][mask] = np.random.normal(8, 4, n_school_students).clip(0, 20)
        data['教育APP使用指数'][mask] = np.random.normal(7, 3, n_school_students).clip(0, 15)
        
        # 心理指标
        data['学习压力指数'][mask] = np.random.normal(pressure_mean, 15, n_school_students).clip(20, 100)
        data['睡眠质量评分'][mask] = np.random.normal(70, 12, n_school_students).clip(30, 100)
        data['课外活动参与度'][mask] = np.random.normal(60, 20, n_school_students).clip(0, 100)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 数据探索
    print(f"\n2. 数据集概览:")
    print(f"   总记录数: {df.shape[0]}名学生")
    print(f"   特征维度: {df.shape[1]}个字段")
    
    print(f"\n3. 数据结构:")
    print(df.dtypes.to_string())
    
    print(f"\n4. 描述性统计（学科成绩）:")
    score_columns = ['语文', '数学', '英语', '物理', '化学']
    print(df[score_columns].describe().round(2))
    
    print(f"\n5. 学校分布:")
    school_dist = df['学校'].value_counts()
    for school, count in school_dist.items():
        tier = schools_info[school]['tier']
        district = schools_info[school]['district']
        print(f"   - {school}（{district}，{tier}）: {count}人 ({count/n_students*100:.1f}%)")
    
    # 数据清洗与特征工程
    print(f"\n6. 数据清洗与特征工程:")
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    if len(missing_cols) > 0:
        print(f"   缺失值统计:\n{missing_cols}")
    else:
        print("   无缺失值 ✓")
    
    # 处理异常值（学科成绩限制在0-100）
    for col in score_columns:
        df[col] = df[col].clip(0, 100)
    print("   学科成绩异常值处理完成（限制在0-100范围内）")
    
    # 创建衍生特征
    print(f"\n7. 衍生特征创建:")
    
    # 学科总分与平均分
    df['总分'] = df[score_columns].sum(axis=1)
    df['平均分'] = df[score_columns].mean(axis=1)
    
    # 学科优势（最高分学科）
    df['优势学科'] = df[score_columns].idxmax(axis=1)
    
    # 文理倾向指数（文科=语文+英语，理科=数学+物理+化学）
    df['文科得分'] = df['语文'] + df['英语']
    df['理科得分'] = df['数学'] + df['物理'] + df['化学']
    df['文理倾向指数'] = (df['文科得分'] - df['理科得分']) / 500  # 归一化到[-1,1]
    
    # 学习效率指数 = 平均分 / sqrt(学习时长+辅导班时长)
    total_study_hours = df['每周学习时长(小时)'] + df['每周辅导班时长(小时)']
    df['学习效率指数'] = df['平均分'] / np.sqrt(total_study_hours)
    
    # 综合素质评分（成绩60% + 课外活动20% + 睡眠质量20%）
    df['综合素质评分'] = (df['平均分'] * 0.6 + 
                        df['课外活动参与度'] * 0.2 + 
                        df['睡眠质量评分'] * 0.2)
    
    print("   衍生特征创建完成，新增字段:")
    new_features = ['总分', '平均分', '文科得分', '理科得分', 
                   '文理倾向指数', '学习效率指数', '综合素质评分']
    for feat in new_features:
        print(f"     - {feat}: {df[feat].describe()['mean']:.2f} ± {df[feat].describe()['std']:.2f}")
    
    return df, score_columns, schools_info

def visualize_education_data(df, score_columns, schools_info):
    """
    可视化成都教育数据（增强版）
    
    包含成都特色可视化：
    1. 区域教育资源对比
    2. 学校层级分析
    3. 学科相关性热图
    4. 学习效率分布
    
    参数：
    df: 学生数据DataFrame
    score_columns: 学科列名列表
    schools_info: 学校信息字典
    """
    print("\n8. 数据可视化分析（成都特色增强）:")
    
    # 设置图形
    fig = plt.figure(figsize=(18, 14))
    
    # 1. 成都各区教育资源雷达图
    print("   a) 成都各区教育资源雷达图...")
    ax1 = plt.subplot(3, 4, 1, projection='polar')
    
    districts = df['所在区'].unique()
    n_districts = len(districts)
    
    # 计算各区平均指标
    district_stats = []
    for district in districts:
        district_data = df[df['所在区'] == district]
        stats = {
            '平均分': district_data['平均分'].mean(),
            '学习压力': district_data['学习压力指数'].mean(),
            '课外活动': district_data['课外活动参与度'].mean(),
            '睡眠质量': district_data['睡眠质量评分'].mean(),
            '辅导时长': district_data['每周辅导班时长(小时)'].mean()
        }
        district_stats.append(stats)
    
    # 标准化数据用于雷达图
    stats_array = np.array([[s['平均分'], s['学习压力'], s['课外活动'], 
                           s['睡眠质量'], s['辅导时长']] for s in district_stats])
    stats_normalized = (stats_array - stats_array.min(axis=0)) / (stats_array.max(axis=0) - stats_array.min(axis=0))
    
    # 绘制雷达图
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for i, district in enumerate(districts):
        values = stats_normalized[i].tolist()
        values += values[:1]  # 闭合
        
        ax1.plot(angles, values, 'o-', linewidth=2, label=district, 
                color=plt.cm.Set2(i/len(districts)))
        ax1.fill(angles, values, alpha=0.1)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(['学业成绩', '学习压力', '课外活动', '睡眠质量', '辅导投入'])
    ax1.set_title('成都各区教育资源雷达对比', fontsize=12, pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax1.grid(True)
    
    # 2. 学校层级与成绩关系
    print("   b) 学校层级与成绩关系分析...")
    ax2 = plt.subplot(3, 4, 2)
    
    # 按学校层级分组
    tier_data = []
    for school, info in schools_info.items():
        mask = df['学校'] == school
        if np.sum(mask) > 0:
            tier_data.append({
                '学校': school,
                '层级': info['tier'],
                '平均分': df.loc[mask, '平均分'].mean(),
                '学生数': np.sum(mask)
            })
    
    tier_df = pd.DataFrame(tier_data)
    
    # 绘制分组箱线图
    box_data = []
    tier_labels = []
    colors = {'顶尖': 'lightcoral', '一流': 'lightblue'}
    
    for tier in ['顶尖', '一流']:
        mask = tier_df['层级'] == tier
        if np.sum(mask) > 0:
            schools_in_tier = tier_df.loc[mask, '学校'].tolist()
            tier_scores = []
            for school in schools_in_tier:
                school_scores = df[df['学校'] == school]['平均分'].values
                tier_scores.extend(school_scores)
            
            box_data.append(tier_scores)
            tier_labels.append(tier)
    
    box = ax2.boxplot(box_data, labels=tier_labels, patch_artist=True)
    
    # 设置颜色
    for patch, tier in zip(box['boxes'], tier_labels):
        patch.set_facecolor(colors[tier])
    
    ax2.set_ylabel('平均分', fontsize=10)
    ax2.set_title('学校层级与成绩分布', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加统计标注
    for i, data in enumerate(box_data, 1):
        median = np.median(data)
        ax2.text(i, median+2, f'中位数: {median:.1f}', 
                ha='center', va='bottom', fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # 3. 学科相关性热图（增强版）
    print("   c) 学科相关性热图（成都教育特色）...")
    ax3 = plt.subplot(3, 4, 3)
    
    # 计算学科相关性矩阵
    corr_matrix = df[score_columns].corr()
    
    # 使用成都特色配色
    im = ax3.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # 设置刻度和标签
    ax3.set_xticks(range(len(score_columns)))
    ax3.set_yticks(range(len(score_columns)))
    ax3.set_xticklabels(score_columns, rotation=45, ha='right', fontsize=9)
    ax3.set_yticklabels(score_columns, fontsize=9)
    
    # 添加相关性数值
    for i in range(len(score_columns)):
        for j in range(len(score_columns)):
            value = corr_matrix.iloc[i, j]
            color = 'white' if abs(value) > 0.7 else 'black'
            ax3.text(j, i, f'{value:.2f}', 
                    ha='center', va='center', 
                    color=color, fontsize=8, fontweight='bold')
    
    ax3.set_title('成都中学生学科成绩相关性', fontsize=12, pad=15)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. 学习时长与成绩关系（分区域）
    print("   d) 学习时长与成绩关系（成都区域对比）...")
    ax4 = plt.subplot(3, 4, 4)
    
    # 按区域分组绘制
    colors_district = plt.cm.tab10(np.linspace(0, 1, len(districts)))
    
    for idx, district in enumerate(districts):
        mask = df['所在区'] == district
        if np.sum(mask) > 0:
            ax4.scatter(df.loc[mask, '每周学习时长(小时)'], 
                       df.loc[mask, '平均分'], 
                       alpha=0.6, s=30, 
                       color=colors_district[idx], 
                       label=district, 
                       edgecolors='white', linewidth=0.5)
    
    ax4.set_xlabel('每周学习时长(小时)', fontsize=10)
    ax4.set_ylabel('平均分', fontsize=10)
    ax4.set_title('学习时长与成绩关系（分区域）', fontsize=12)
    ax4.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax4.grid(alpha=0.3)
    
    # 5. 压力-成绩平衡分析
    print("   e) 学习压力与成绩平衡分析...")
    ax5 = plt.subplot(3, 4, 5)
    
    # 创建压力分组
    pressure_bins = pd.qcut(df['学习压力指数'], q=5, labels=['很低', '较低', '中等', '较高', '很高'])
    
    # 计算各压力组的平均成绩
    pressure_groups = df.groupby(pressure_bins).agg({
        '平均分': ['mean', 'std', 'count'],
        '学习效率指数': 'mean',
        '睡眠质量评分': 'mean'
    })
    
    # 绘制分组条形图
    x_pos = np.arange(len(pressure_groups))
    width = 0.25
    
    ax5.bar(x_pos - width, pressure_groups[('平均分', 'mean')], width, 
           label='平均分', color='skyblue', alpha=0.8)
    ax5.bar(x_pos, pressure_groups[('学习效率指数', 'mean')], width, 
           label='学习效率', color='lightcoral', alpha=0.8)
    ax5.bar(x_pos + width, pressure_groups[('睡眠质量评分', 'mean')], width, 
           label='睡眠质量', color='lightgreen', alpha=0.8)
    
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(pressure_groups.index, rotation=45, fontsize=9)
    ax5.set_ylabel('标准化得分', fontsize=10)
    ax5.set_title('学习压力与各项指标关系', fontsize=12)
    ax5.legend(fontsize=8)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. 教育APP使用效果分析
    print("   f) 教育APP使用效果分组分析...")
    ax6 = plt.subplot(3, 4, 6)
    
    # 创建APP使用分组
    app_bins = pd.qcut(df['教育APP使用指数'], q=4, labels=['很少', '较少', '中等', '频繁'])
    
    app_stats = df.groupby(app_bins).agg({
        '平均分': 'mean',
        '学习压力指数': 'mean',
        '学习效率指数': 'mean'
    })
    
    # 绘制分组条形图
    x_pos = np.arange(len(app_stats))
    
    for i, col in enumerate(['平均分', '学习压力指数', '学习效率指数']):
        ax6.bar(x_pos + i*0.25, app_stats[col], 0.25, 
               label=col, alpha=0.8)
    
    ax6.set_xticks(x_pos + 0.25)
    ax6.set_xticklabels(app_stats.index, fontsize=9)
    ax6.set_ylabel('得分', fontsize=10)
    ax6.set_title('教育APP使用时长分组效果', fontsize=12)
    ax6.legend(fontsize=8)
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. 文理倾向分布
    print("   g) 学生文理倾向分布（成都特色）...")
    ax7 = plt.subplot(3, 4, 7)
    
    # 绘制文理倾向分布直方图
    ax7.hist(df['文理倾向指数'], bins=30, edgecolor='black', 
            color='mediumpurple', alpha=0.7)
    
    # 添加正态分布拟合
    from scipy.stats import norm
    mu, std = df['文理倾向指数'].mean(), df['文理倾向指数'].std()
    x = np.linspace(df['文理倾向指数'].min(), df['文理倾向指数'].max(), 100)
    p = norm.pdf(x, mu, std)
    ax7.plot(x, p*len(df)*0.1, 'r-', linewidth=2, 
            label=f'正态拟合\nμ={mu:.2f}, σ={std:.2f}')
    
    ax7.axvline(x=0, color='gray', linestyle='--', alpha=0.7, 
               label='文理平衡线')
    
    ax7.set_xlabel('文理倾向指数', fontsize=10)
    ax7.set_ylabel('学生人数', fontsize=10)
    ax7.set_title('成都中学生文理倾向分布', fontsize=12)
    ax7.legend(fontsize=8)
    ax7.grid(alpha=0.3)
    
    # 8. 综合素质评分分布
    print("   h) 综合素质评分分布...")
    ax8 = plt.subplot(3, 4, 8)
    
    # 按学校层级分组绘制综合素质评分分布
    for tier in ['顶尖', '一流']:
        mask = df['学校层级'] == tier
        if np.sum(mask) > 0:
            ax8.hist(df.loc[mask, '综合素质评分'], bins=20, 
                    alpha=0.6, label=tier, density=True)
    
    ax8.set_xlabel('综合素质评分', fontsize=10)
    ax8.set_ylabel('密度', fontsize=10)
    ax8.set_title('综合素质评分分布（按学校层级）', fontsize=12)
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3)
    
    # 9. 关键洞察总结
    print(f"\n9. 成都教育数据关键洞察总结:")
    print("-" * 60)
    
    # 洞察1：区域教育资源差异
    print("   a) 区域教育资源差异显著:")
    for district in districts:
        district_avg = df[df['所在区'] == district]['平均分'].mean()
        district_pressure = df[df['所在区'] == district]['学习压力指数'].mean()
        print(f"      - {district}: 平均分{district_avg:.1f}，压力指数{district_pressure:.1f}")
    
    # 洞察2：学习效率最优区间
    efficiency_q3 = df['学习效率指数'].quantile(0.75)
    efficient_students = df[df['学习效率指数'] > efficiency_q3]
    avg_students = df[df['学习效率指数'] <= efficiency_q3]
    
    print(f"\n   b) 学习效率前25%学生特征:")
    print(f"      - 平均学习时长: {efficient_students['每周学习时长(小时)'].mean():.1f}h")
    print(f"      - 平均成绩: {efficient_students['平均分'].mean():.1f}分")
    print(f"      - 压力指数: {efficient_students['学习压力指数'].mean():.1f}")
    print(f"      - 睡眠质量: {efficient_students['睡眠质量评分'].mean():.1f}")
    
    # 洞察3：教育APP使用效果
    print(f"\n   c) 教育APP使用效果分析:")
    for app_group in ['很少', '较少', '中等', '频繁']:
        mask = df['教育APP使用指数'].groupby(pd.qcut(df['教育APP使用指数'], q=4, labels=['很少', '较少', '中等', '频繁'])).groups.get(app_group, [])
        if len(mask) > 0:
            group_data = df.iloc[mask]
            print(f"      - {app_group}使用组: {len(group_data)}人，平均分{group_data['平均分'].mean():.1f}")
    
    plt.tight_layout()
    plt.show()
    
    return {
        'district_stats': districts,
        'tier_comparison': tier_df,
        'correlation_matrix': corr_matrix,
        'pressure_analysis': pressure_groups,
        'app_analysis': app_stats
    }

# ============================================================================
# 第三部分：代码挑战详细框架（增强版）
# ============================================================================

def challenge_1_pca_framework(df, score_columns):
    """
    挑战1：学生能力PCA分析详细框架
    
    输入：
    df: 学生数据DataFrame
    score_columns: 学科列名列表
    
    输出：
    dict: PCA分析结果和可视化
    """
    print("\n🎯 挑战1：学生能力PCA分析详细框架")
    print("-" * 60)
    
    print("任务描述:")
    print("1. 对五科成绩进行PCA降维，提取2个主成分")
    print("2. 分析各主成分的学科权重和实际意义") 
    print("3. 可视化降维结果和教育洞察")
    print("4. 提出具体的教育改进建议")
    
    print("\n详细步骤:")
    
    steps = [
        "# 步骤1: 数据准备",
        "X = df[score_columns].values  # 提取成绩矩阵",
        "",
        "# 步骤2: 数据标准化（PCA前必须）",
        "from sklearn.preprocessing import StandardScaler",
        "scaler = StandardScaler()",
        "X_scaled = scaler.fit_transform(X)",
        "",
        "# 步骤3: PCA建模",
        "from sklearn.decomposition import PCA",
        "pca = PCA(n_components=2)",
        "X_pca = pca.fit_transform(X_scaled)",
        "",
        "# 步骤4: 结果分析",
        "print(f'方差解释率: {pca.explained_variance_ratio_}')",
        "print(f'累计解释率: {np.sum(pca.explained_variance_ratio_):.3f}')",
        "print(f'特征向量矩阵:\\n{pca.components_}')",
        "",
        "# 步骤5: 可视化（提供完整代码框架）",
        "fig, axes = plt.subplots(2, 3, figsize=(15, 10))",
        "",
        "# 5.1 PCA降维散点图",
        "scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30)",
        "axes[0, 0].set_xlabel('第一主成分 (解释率: {:.1%})'.format(pca.explained_variance_ratio_[0]))",
        "axes[0, 0].set_ylabel('第二主成分 (解释率: {:.1%})'.format(pca.explained_variance_ratio_[1]))",
        "axes[0, 0].set_title('PCA降维结果 - 学生分布')",
        "axes[0, 0].grid(alpha=0.3)",
        "",
        "# 5.2 方差解释率条形图",
        "axes[0, 1].bar(range(1, 3), pca.explained_variance_ratio_, color='skyblue')",
        "axes[0, 1].set_xlabel('主成分')",
        "axes[0, 1].set_ylabel('方差解释率')",
        "axes[0, 1].set_title('各主成分方差解释率')",
        "for i, v in enumerate(pca.explained_variance_ratio_):",
        "    axes[0, 1].text(i+1, v+0.01, f'{v:.1%}', ha='center')",
        "",
        "# 5.3 特征向量热力图",
        "im = axes[0, 2].imshow(pca.components_, cmap='RdYlBu', aspect='auto')",
        "axes[0, 2].set_xticks(range(len(score_columns)))",
        "axes[0, 2].set_xticklabels(score_columns, rotation=45)",
        "axes[0, 2].set_yticks([0, 1])",
        "axes[0, 2].set_yticklabels(['PC1', 'PC2'])",
        "axes[0, 2].set_title('主成分与学科关系热图')",
        "plt.colorbar(im, ax=axes[0, 2])",
        "",
        "# 步骤6: 教育洞察与建议",
        "# 请基于PCA结果，回答以下问题并提出建议:",
        "# 1. 第一主成分主要反映学生的什么能力？",
        "# 2. 第二主成分补充了哪些信息？",
        "# 3. 哪些学科相关性最强？这对教学安排有什么启示？",
        "# 4. 基于成都教育现状，提出2条具体的改进建议",
    ]
    
    for step in steps:
        print(step)
    
    print("\n输入数据规格:")
    print(f"- 数据形状: {df.shape}")
    print(f"- 学科列: {score_columns}")
    print(f"- 样本数: {len(df)}")
    
    print("\n预期输出:")
    expected_output = {
        'pca_model': 'sklearn.decomposition.PCA对象',
        'explained_variance_ratio': 'array([PC1解释率, PC2解释率])',
        'components': 'array([[PC1各学科权重], [PC2各学科权重]])',
        'pca_scores': 'array([[学生1的PC1得分, PC2得分], ...])',
        'visualizations': ['散点图', '方差解释率图', '热力图'],
        'insights': {
            'primary_factor': '第一主成分解释',
            'secondary_factor': '第二主成分解释',
            'subject_correlations': '学科相关性分析',
            'recommendations': '2条教育建议（每条需包含数据依据）'
        }
    }
    
    for key, value in expected_output.items():
        print(f"- {key}: {value}")
    
    return expected_output

def challenge_2_linear_programming_framework():
    """
    挑战2：教育资源优化模型详细框架
    
    输出：
    dict: 线性规划模型框架和求解指南
    """
    print("\n🎯 挑战2：教育资源优化模型详细框架")
    print("-" * 60)
    
    print("任务背景:")
    print("成都某区教育局有100万元预算，需在5所学校间分配，最大化学生总成绩提升。")
    print("每所学校有最小必要投入、最大有效投入、单位成本系数、成绩提升函数。")
    
    print("\n数学模型:")
    math_model = """
设决策变量: x₁, x₂, x₃, x₄, x₅ (各校新增投入，万元)
目标函数: max Σᵢ (wᵢxᵢ + bᵢ)  # 总成绩提升
约束条件:
1. 预算约束: Σᵢ aᵢxᵢ ≤ 100  # aᵢ为单位成本系数
2. 投入下限: xᵢ ≥ mᵢ       # 最小必要投入
3. 投入上限: xᵢ ≤ Mᵢ       # 最大有效投入
4. 非负约束: xᵢ ≥ 0
    """
    print(math_model)
    
    print("\n详细实现步骤:")
    
    steps = [
        "# 步骤1: 问题参数定义",
        "import numpy as np",
        "from scipy.optimize import linprog",
        "",
        "# 成都5所重点中学",
        "school_names = ['成都七中', '树德中学', '石室中学', '成都外国语学校', '玉林中学']",
        "n_schools = len(school_names)",
        "",
        "# 设置随机种子确保可复现",
        "np.random.seed(42)",
        "",
        "# 生成模拟参数",
        "resource_cost = np.random.uniform(0.8, 1.2, n_schools)      # 单位成本系数 aᵢ",
        "min_investment = np.random.uniform(5, 15, n_schools)         # 最小必要投入 mᵢ",
        "max_investment = np.random.uniform(20, 40, n_schools)        # 最大有效投入 Mᵢ",
        "improvement_coef = np.random.uniform(0.5, 1.5, n_schools)    # 成绩提升系数 wᵢ",
        "improvement_const = np.random.uniform(0, 5, n_schools)       # 成绩提升常数项 bᵢ",
        "",
        "# 步骤2: 转化为标准线性规划形式",
        "# 目标函数系数 (最大化问题取负)",
        "c = -improvement_coef  # 注意：linprog默认最小化",
        "",
        "# 不等式约束矩阵 A_ub * x ≤ b_ub",
        "# 预算约束: Σ aᵢxᵢ ≤ 100",
        "A_budget = resource_cost.reshape(1, -1)",
        "b_budget = np.array([100])",
        "",
        "# 上限约束: xᵢ ≤ Mᵢ  →  xᵢ - Mᵢ ≤ 0",
        "# 下限约束: xᵢ ≥ mᵢ  →  -xᵢ + mᵢ ≤ 0",
        "# 通常直接用bounds参数处理更简单",
        "",
        "# 步骤3: 定义变量边界",
        "bounds = [(min_i, max_i) for min_i, max_i in zip(min_investment, max_investment)]",
        "",
        "# 步骤4: 求解线性规划",
        "result = linprog(c, A_ub=A_budget, b_ub=b_budget, bounds=bounds, method='highs')",
        "",
        "# 步骤5: 结果分析与验证",
        "if result.success:",
        "    optimal_investment = result.x",
        "    total_improvement = -result.fun  # 取负得实际提升值",
        "    ",
        "    print('优化求解成功！')",
        "    print(f'最优投资方案:')",
        "    for i, school in enumerate(school_names):",
        "        print(f'  {school}: {optimal_investment[i]:.2f}万元')",
        "    print(f'总成绩提升: {total_improvement:.2f}分')",
        "    ",
        "    # 验证约束条件",
        "    budget_used = np.sum(resource_cost * optimal_investment)",
        "    print(f'预算使用: {budget_used:.2f}万元 (上限: 100万元)')",
        "    ",
        "    # 检查边界条件",
        "    for i in range(n_schools):",
        "        if not (min_investment[i] <= optimal_investment[i] <= max_investment[i]):",
        "            print(f'警告: {school_names[i]}投资超出边界')",
        "else:",
        "    print('优化失败:', result.message)",
        "",
        "# 步骤6: 敏感度分析（预算变化影响）",
        "budgets = np.linspace(80, 120, 9)  # 预算从80万到120万",
        "improvements = []",
        "",
        "for budget in budgets:",
        "    b_budget_adj = np.array([budget])",
        "    result_adj = linprog(c, A_ub=A_budget, b_ub=b_budget_adj, bounds=bounds, method='highs')",
        "    if result_adj.success:",
        "        improvements.append(-result_adj.fun)",
        "",
        "# 可视化敏感度分析结果",
        "plt.figure(figsize=(10, 6))",
        "plt.plot(budgets, improvements, 'bo-', linewidth=2)",
        "plt.xlabel('预算总额 (万元)')",
        "plt.ylabel('总成绩提升 (分)')",
        "plt.title('教育资源分配敏感度分析')",
        "plt.grid(True, alpha=0.3)",
        "plt.fill_between(budgets, improvements, alpha=0.2)",
        "plt.show()",
    ]
    
    for step in steps:
        print(step)
    
    print("\n预期输出:")
    expected_output = {
        'parameters': {
            'school_names': '成都5所中学名称列表',
            'resource_cost': '各单位成本系数数组',
            'min_investment': '各校最小投入数组',
            'max_investment': '各校最大投入数组',
            'improvement_coef': '成绩提升系数数组',
            'improvement_const': '成绩提升常数项数组'
        },
        'solution': {
            'optimal_investment': '最优投资分配数组',
            'total_improvement': '总成绩提升值',
            'budget_utilization': '预算利用率',
            'feasibility': '求解成功标志'
        },
        'analysis': {
            'sensitivity': '预算变化对结果的影响数据',
            'visualizations': ['预算-提升关系图', '投资分配柱状图'],
            'fairness_metrics': '分配公平性指标'
        }
    }
    
    for key, value in expected_output.items():
        print(f"- {key}: {value}")
    
    return expected_output

def challenge_3_recommendation_system_framework():
    """
    挑战3：智能教育推荐系统详细框架
    
    输出：
    dict: 推荐系统实现框架和评估指南
    """
    print("\n🎯 挑战3：智能教育推荐系统详细框架")
    print("-" * 60)
    
    print("任务背景:")
    print("开发成都教育科技平台的个性化推荐系统，使用SVD分析学生-学习资源交互矩阵。")
    print("学生200人（5所学校各40人），学习资源50个（视频、习题、电子书、模拟考试）。")
    
    print("\n数据规格:")
    data_spec = """
- 学生数量: 200 (成都5校各40人)
- 学习资源: 50个
  - 视频课程: 20个 (编号R001-R020)
  - 习题集: 15个 (编号R021-R035)
  - 电子书: 10个 (编号R036-R045)
  - 模拟考试: 5个 (编号R046-R050)
- 评分范围: 1-5分 (整数)
- 稀疏度: 80%缺失 (模拟真实场景)
- 资源元数据: 学科标签、难度等级、时长估计、成都课程标准匹配度
    """
    print(data_spec)
    
    print("\n详细实现步骤:")
    
    steps = [
        "# 步骤1: 数据生成与模拟",
        "import numpy as np",
        "from scipy.sparse.linalg import svds",
        "from sklearn.metrics import mean_squared_error",
        "",
        "# 参数设置",
        "n_students = 200",
        "n_resources = 50",
        "sparsity = 0.8  # 80%缺失",
        "",
        "# 初始化评分矩阵",
        "ratings = np.zeros((n_students, n_resources))",
        "",
        "# 生成学生能力水平（正态分布）",
        "np.random.seed(42)",
        "student_ability = np.random.normal(0.5, 0.2, n_students)",
        "student_ability = np.clip(student_ability, 0.1, 0.9)",
        "",
        "# 生成资源难度",
        "resource_difficulty = np.random.normal(0.5, 0.2, n_resources)",
        "resource_difficulty = np.clip(resource_difficulty, 0.1, 0.9)",
        "",
        "# 生成交互评分（考虑能力-难度匹配）",
        "n_ratings = int(n_students * n_resources * (1 - sparsity))",
        "student_indices = np.random.randint(0, n_students, n_ratings)",
        "resource_indices = np.random.randint(0, n_resources, n_ratings)",
        "",
        "# 评分生成逻辑：能力匹配度越高，评分越高",
        "for i, j in zip(student_indices, resource_indices):",
        "    ability = student_ability[i]",
        "    difficulty = resource_difficulty[j]",
        "    ",
        "    # 匹配度计算",
        "    match_score = 1.0 - abs(ability - difficulty)",
        "    ",
        "    # 加入随机性",
        "    noise = np.random.normal(0, 0.2)",
        "    raw_rating = match_score * 4 + 1 + noise  # 映射到1-5分",
        "    rating = int(np.clip(np.round(raw_rating), 1, 5))",
        "    ",
        "    ratings[i, j] = rating",
        "",
        "# 步骤2: SVD矩阵分解",
        "# 选择潜在特征维度",
        "k = 10",
        "",
        "# 执行SVD（处理稀疏矩阵）",
        "U, sigma, Vt = svds(ratings, k=k)",
        "",
        "# 重构矩阵",
        "sigma_matrix = np.diag(sigma)",
        "ratings_pred = U @ sigma_matrix @ Vt",
        "",
        "# 步骤3: 推荐系统评估",
        "# 计算RMSE（仅在有评分的部分）",
        "mask = ratings > 0",
        "rmse = np.sqrt(mean_squared_error(ratings[mask], ratings_pred[mask]))",
        "print(f'RMSE: {rmse:.4f}')",
        "",
        "# 计算MAE",
        "mae = np.mean(np.abs(ratings[mask] - ratings_pred[mask]))",
        "print(f'MAE: {mae:.4f}')",
        "",
        "# 步骤4: 个性化推荐生成",
        "# 为特定学生生成推荐",
        "def generate_recommendations(student_id, ratings, ratings_pred, top_n=5):",
        "    \"\"\"为指定学生生成top-N推荐\"\"\"",
        "    student_ratings = ratings[student_id]",
        "    student_predictions = ratings_pred[student_id]",
        "    ",
        "    # 找出未评分但预测评分高的资源",
        "    unrated_mask = student_ratings == 0",
        "    unrated_indices = np.where(unrated_mask)[0]",
        "    ",
        "    if len(unrated_indices) == 0:",
        "        return []",
        "    ",
        "    # 获取预测评分",
        "    unrated_predictions = student_predictions[unrated_indices]",
        "    ",
        "    # 按预测评分排序",
        "    sorted_indices = np.argsort(unrated_predictions)[::-1]",
        "    top_indices = unrated_indices[sorted_indices[:top_n]]",
        "    ",
        "    return top_indices",
        "",
        "# 示例：为第0号学生生成推荐",
        "student_id = 0",
        "recommendations = generate_recommendations(student_id, ratings, ratings_pred)",
        "print(f'学生{student_id}的top-5推荐资源索引: {recommendations}')",
        "",
        "# 步骤5: 推荐质量分析",
        "# 评估推荐多样性、新颖性、教育价值",
        "def analyze_recommendation_quality(recommendations, resource_metadata):",
        "    \"\"\"分析推荐列表的质量\"\"\"",
        "    quality_metrics = {}",
        "    ",
        "    # 多样性：推荐资源类型的分布",
        "    resource_types = resource_metadata['type']",
        "    rec_types = resource_types[recommendations]",
        "    unique_types = np.unique(rec_types)",
        "    diversity = len(unique_types) / len(np.unique(resource_types))",
        "    ",
        "    quality_metrics['diversity'] = diversity",
        "    ",
        "    # 新颖性：推荐资源的热门程度（模拟）",
        "    # 假设热门资源被更多学生评分",
        "    resource_popularity = np.sum(ratings > 0, axis=0)",
        "    rec_popularity = resource_popularity[recommendations]",
        "    novelty = 1.0 - np.mean(rec_popularity) / np.max(resource_popularity)",
        "    ",
        "    quality_metrics['novelty'] = novelty",
        "    ",
        "    # 教育价值：难度匹配度和知识覆盖度",
        "    # 这里需要根据具体业务逻辑实现",
        "    ",
        "    return quality_metrics",
        "",
        "# 步骤6: 成都特色融入",
        "# 资源标注成都课程标准匹配度",
        "# 考虑区域教育资源差异",
        "# 推荐时兼顾升学政策导向",
    ]
    
    for step in steps:
        print(step)
    
    print("\n预期输出:")
    expected_output = {
        'data_generation': {
            'ratings_matrix': '200×50评分矩阵',
            'student_ability': '200维学生能力数组',
            'resource_difficulty': '50维资源难度数组',
            'sparsity': '实际缺失率'
        },
        'svd_results': {
            'U_matrix': '200×10左奇异矩阵',
            'sigma': '10维奇异值数组',
            'Vt_matrix': '10×50右奇异矩阵转置',
            'reconstructed_matrix': '200×50重构评分矩阵'
        },
        'evaluation_metrics': {
            'RMSE': '均方根误差',
            'MAE': '平均绝对误差',
            'precision_at_k': '前k推荐准确率'
        },
        'recommendation_output': {
            'top_recommendations': '为每位学生的推荐列表',
            'quality_analysis': '推荐质量评估结果',
            'chengdu_features': '成都特色融入分析'
        }
    }
    
    for key, value in expected_output.items():
        print(f"- {key}: {value}")
    
    return expected_output

# ============================================================================
# 第四部分：主执行流程
# ============================================================================

def main():
    """
    主执行函数（增强版）
    
    执行顺序：
    1. NumPy高级线性代数演示
    2. 成都教育数据集创建与可视化
    3. 代码挑战框架展示
    """
    print("\n🚀 开始执行Day 1学习内容（调整版）...")
    
    try:
        # 第一部分：NumPy高级线性代数
        print("\n" + "="*80)
        linear_algebra_data = numpy_advanced_linear_algebra()
        
        # 第二部分：Pandas数据处理
        print("\n" + "="*80)
        df, score_columns, schools_info = create_chengdu_education_dataset()
        visualization_results = visualize_education_data(df, score_columns, schools_info)
        
        # 第三部分：代码挑战框架展示
        print("\n" + "="*80)
        print("\n📋 代码挑战框架展示（详细实现指南）")
        
        # 挑战1框架
        challenge_1_framework = challenge_1_pca_framework(df, score_columns)
        
        # 挑战2框架  
        challenge_2_framework = challenge_2_linear_programming_framework()
        
        # 挑战3框架
        challenge_3_framework = challenge_3_recommendation_system_framework()
        
        # 总结与下一步指引
        print("\n" + "="*80)
        print("\n✅ Day 1代码示例执行完成（调整版）!")
        print("="*80)
        
        print("\n📊 数据摘要:")
        print(f"- 学生数据集: {df.shape[0]}行 × {df.shape[1]}列")
        print(f"- 学科数量: {len(score_columns)}个")
        print(f"- 学校数量: {len(schools_info)}所")
        print(f"- 矩阵运算: {linear_algebra_data['covariance_matrix'].shape}协方差矩阵")
        
        print("\n🎯 今日学习重点:")
        print("1. 深入理解向量空间的8条公理及其AI应用意义")
        print("2. 掌握矩阵作为线性变换的几何与代数统一")
        print("3. 理解特征值分解在PCA中的数学本质")
        print("4. 应用线性代数解决成都教育实际问题")
        
        print("\n📝 下一步操作:")
        print("1. 仔细阅读《Day1-线性代数核心概念与Python实战_v2.md》理论部分")
        print("2. 选择至少1个代码挑战进行详细实现")
        print("3. 运行并调试你的代码，确保理解每一行")
        print("4. 在18:00前提交《今日执行卡片》（3个收获、1个疑问、1个感悟）")
        print("5. 记录学习过程中的数学理解和技术难点")
        
        print("\n🔗 资源提醒:")
        print("- 主文档: outputs/每日内容/Day1-线性代数核心概念与Python实战_v2.md")
        print("- 代码文件: outputs/每日内容/Day1-代码挑战_v2.py")
        print("- 学习大纲: outputs/学习大纲/AI全栈工程师转型学习大纲.md")
        
        # 返回所有数据供后续使用
        return {
            'linear_algebra': linear_algebra_data,
            'education_data': {
                'dataframe': df,
                'score_columns': score_columns,
                'schools_info': schools_info
            },
            'visualization': visualization_results,
            'challenge_frameworks': {
                'challenge_1': challenge_1_framework,
                'challenge_2': challenge_2_framework,
                'challenge_3': challenge_3_framework
            }
        }
        
    except Exception as e:
        print(f"\n❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 执行入口
# ============================================================================

if __name__ == "__main__":
    # 执行主函数
    print("\n" + "="*80)
    print("开始Day 1学习内容演示...")
    print("="*80)
    
    data = main()
    
    if data is not None:
        print("\n📊 数据访问示例:")
        print(f"- 协方差矩阵形状: {data['linear_algebra']['covariance_matrix'].shape}")
        print(f"- 学生数据行数: {len(data['education_data']['dataframe'])}")
        print(f"- 学科列数: {len(data['education_data']['score_columns'])}")
        print(f"- 学校数量: {len(data['education_data']['schools_info'])}")
        
        print("\n💡 技术提示:")
        print("- 理论部分建议反复阅读，结合几何直观理解")
        print("- 代码挑战先从理解框架开始，再逐步实现")
        print("- 遇到数学概念困惑时，可查阅扩展学习资源")
        
        print("\n🎓 祝你学习顺利，早日成为AI全栈工程师!")
        print("="*80)
    else:
        print("\n⚠️ 执行失败，请检查错误信息并修正代码。")