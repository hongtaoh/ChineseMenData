import numpy as np
import pandas as pd
import json

# 基于人口抽样调查生成年龄分布
# 考虑教育程度和收入的相关关系:
# P(education|age, hometown)
# P(income|age, education)
# P(current_location|hometown, education)
# P(property_status|age, income, current_location)
# P(height|age) 
# P(vision|education)
# P(personal_assets|age, income, education, current_location)


def get_age_probs(age_distribution):
    total_population = sum(list(age_distribution.values()))
    # key is each age from 1 to 99 and value is the prob
    age_probs = dict()
    for age_group, group_count in age_distribution.items():
        group_prob = group_count/total_population
        start, end = [int(x) for x in age_group.split('-')]
        for i in range(start, end + 1):
            age_probs[i] = group_prob/5
    return age_probs 

def sample_ages(age_ranges, n_samples, age_probs):
    return np.random.choice(
        age_ranges,
        size = n_samples,
        p = list(age_probs.values())
    )

def sample_hometowns(hometown_probs, n_samples):
    return np.random.choice(
        list(hometown_probs.keys()),
        size=n_samples,
        p=list(hometown_probs.values())
    )

def get_education_probabilities(age, hometown):
    """
    根据年龄和家乡返回教育程度的概率分布
    
    教育程度：[高中及以下, 大专, 本科, 研究生及以上]
    """
    # 基准概率 (全国总体教育程度分布)
    # 高中及以下 = 没上过学 + 小学 + 初中 + 高中 = 80.66%
    # 大专 = 9.69%
    # 本科 = 8.13%
    # 研究生 = 0.93%
    base_probs = np.array([0.8066, 0.0969, 0.0813, 0.0093])
    
    # 年龄调整系数 (年轻人受教育程度普遍更高)
    age_factors = np.ones(4)
    if age < 30:
        age_factors = np.array([0.5, 1.5, 2.0, 2.5])  # 年轻人高学历比例更高
    elif 30 <= age < 45:
        age_factors = np.array([0.7, 1.3, 1.5, 1.8])  # 中年人学历分布相对均衡
    elif 45 <= age < 60:
        age_factors = np.array([1.2, 0.8, 0.6, 0.4])  # 中老年人低学历比例更高
    else:
        age_factors = np.array([1.5, 0.5, 0.3, 0.1])  # 老年人以低学历为主
    
    # 地域调整系数
    location_factors = {
        '一线城市': np.array([0.4, 1.2, 1.8, 2.5]),    # 一线城市高学历比例最高
        '二线城市': np.array([0.6, 1.5, 1.5, 1.8]),    # 二线城市学历分布相对均衡
        '三线城市': np.array([0.8, 1.2, 1.2, 1.0]),    # 三线城市中等学历为主
        '县城': np.array([1.2, 0.8, 0.6, 0.3]),       # 县城以低学历为主
        '农村': np.array([1.5, 0.5, 0.3, 0.1])        # 农村以低学历为主
    }
    
    # 特殊约束条件
    if age < 22:  # 22岁以下研究生概率接近0
        age_factors[3] = 0.001
    if age < 19:  # 19岁以下本科概率很低
        age_factors[2] = 0.001
    
    # 计算最终概率
    final_probs = base_probs * age_factors * location_factors[hometown]
    
    # 归一化
    final_probs = final_probs / final_probs.sum()
    
    return final_probs

def generate_education(ages, hometowns):
    """为给定的年龄和家乡生成教育程度"""
    education_levels = ['高中及以下', '大专', '本科', '研究生及以上']
    education = []
    
    for age, hometown in zip(ages, hometowns):
        probs = get_education_probabilities(age, hometown)
        edu = np.random.choice(education_levels, p=probs)
        education.append(edu)
    
    return education

def get_income_probabilities(age, education):
    """根据年龄和教育程度返回收入的概率分布"""
    retirement_age = 60
    
    # 退休后的收入分布
    if age >= retirement_age:
        if education == '研究生及以上':
            # 退休金较高，集中在中等偏上收入段
            return np.array([0.05, 0.15, 0.40, 0.25, 0.12, 0.03])
        elif education == '本科':
            # 退休金中等偏上
            return np.array([0.10, 0.25, 0.35, 0.20, 0.08, 0.02])
        elif education == '大专':
            # 退休金中等
            return np.array([0.15, 0.40, 0.30, 0.10, 0.04, 0.01])
        else:  # 高中及以下
            # 退休金较低
            return np.array([0.40, 0.45, 0.12, 0.02, 0.01, 0.00])
    
    # 在职人员的收入分布
    if education == '研究生及以上':
        base_probs = np.array([0.05, 0.35, 0.35, 0.15, 0.08, 0.02])
    elif education == '本科':
        base_probs = np.array([0.10, 0.45, 0.30, 0.10, 0.04, 0.01])
    elif education == '大专':
        base_probs = np.array([0.20, 0.50, 0.20, 0.07, 0.02, 0.01])
    else:  # 高中及以下
        base_probs = np.array([0.40, 0.45, 0.10, 0.04, 0.01, 0.00])
    
    # 年龄对在职人员收入的影响
    if age < 25:
        # 刚工作，收入普遍较低
        age_effect = np.array([1.5, 1.2, 0.6, 0.3, 0.1, 0.0])
    elif 25 <= age < 35:
        # 职业发展期，收入快速增长
        age_effect = np.array([0.7, 1.0, 1.2, 1.1, 0.8, 0.5])
    elif 35 <= age < 45:
        # 事业上升期，高收入比例增加
        age_effect = np.array([0.5, 0.8, 1.0, 1.2, 1.1, 1.0])
    else:  # 45-60
        # 事业稳定期，保持较高收入
        age_effect = np.array([0.4, 0.7, 1.0, 1.1, 1.2, 1.2])
    
    # 计算在职人员最终概率
    final_probs = base_probs * age_effect
    final_probs = final_probs / final_probs.sum()
    
    # 特殊约束
    if age < 18:
        final_probs = np.array([0.8, 0.18, 0.02, 0.0, 0.0, 0.0])
    
    return final_probs

def generate_income(ages, education_levels):
    """为给定的年龄和教育程度生成收入"""
    income_levels = ['<5万', '5-15万', '15-30万', '30-50万', '50-100万', '>100万']
    income = []
    
    for age, edu in zip(ages, education_levels):
        probs = get_income_probabilities(age, edu)
        inc = np.random.choice(income_levels, p=probs)
        income.append(inc)
    
    return income

def get_location_probabilities(hometown, education):
    """
    获取当前居住地的概率分布
    location_levels = ['一线城市', '二线城市', '三线城市', '县城', '农村']
    """
    # 基础流动矩阵（从hometown到current_location的概率）
    # 行：hometown，列：current_location
    base_mobility = {
        '一线城市': [0.80, 0.15, 0.03, 0.01, 0.01],  # 一线城市更可能留在一线
        '二线城市': [0.20, 0.65, 0.10, 0.03, 0.02],  # 二线城市有机会去一线
        '三线城市': [0.15, 0.20, 0.55, 0.07, 0.03],  # 三线城市向上流动机会适中
        '县城': [0.10, 0.15, 0.20, 0.45, 0.10],      # 县城有一定向上流动
        '农村': [0.05, 0.10, 0.15, 0.30, 0.40]       # 农村向上流动难度较大
    }
    
    # 教育水平对流动的影响系数
    education_factors = {
        '研究生及以上': {
            '一线城市': 2.0,    # 研究生更可能去一线城市
            '二线城市': 1.5,    # 二线城市次之
            '三线城市': 0.8,
            '县城': 0.3,
            '农村': 0.1
        },
        '本科': {
            '一线城市': 1.5,
            '二线城市': 1.3,
            '三线城市': 1.0,
            '县城': 0.5,
            '农村': 0.2
        },
        '大专': {
            '一线城市': 1.0,
            '二线城市': 1.2,
            '三线城市': 1.2,
            '县城': 0.8,
            '农村': 0.4
        },
        '高中及以下': {
            '一线城市': 0.5,
            '二线城市': 0.8,
            '三线城市': 1.0,
            '县城': 1.2,
            '农村': 1.2
        }
    }
    
    # 获取基础流动概率
    probs = np.array(base_mobility[hometown])
    
    # 应用教育水平的影响
    factors = np.array([education_factors[education][loc] for loc in 
                       ['一线城市', '二线城市', '三线城市', '县城', '农村']])
    
    # 计算最终概率
    final_probs = probs * factors
    
    # 归一化
    final_probs = final_probs / final_probs.sum()
    
    return final_probs

def generate_current_location(hometowns, education_levels):
    """为给定的家乡和教育程度生成当前居住地"""
    location_levels = ['一线城市', '二线城市', '三线城市', '县城', '农村']
    current_locations = []
    
    for hometown, edu in zip(hometowns, education_levels):
        probs = get_location_probabilities(hometown, edu)
        loc = np.random.choice(location_levels, p=probs)
        current_locations.append(loc)
    
    return current_locations


def get_property_probabilities(income, current_location, age):
    """
    获取房产状态的概率分布
    property_status = ['无房产', '有房有贷款', '有房无贷款']
    """
    # 基础购房难度系数（越大表示越难买房）
    location_difficulty = {
        '一线城市': 2.5,
        '二线城市': 1.8,
        '三线城市': 1.3,
        '县城': 1.0,
        '农村': 0.7
    }
    
    # 收入档位对应的基础有房概率
    income_base_probs = {
        '<5万': [0.80, 0.15, 0.05],      # [无房产, 有房有贷款, 有房无贷款]
        '5-15万': [0.70, 0.25, 0.05],
        '15-30万': [0.50, 0.40, 0.10],
        '30-50万': [0.30, 0.50, 0.20],
        '50-100万': [0.20, 0.45, 0.35],
        '>100万': [0.10, 0.35, 0.55]
    }
    
    # 年龄影响系数
    def get_age_factor(age):
        if age < 25:
            return [1.5, 0.3, 0.1]  # 年轻人更可能无房
        elif 25 <= age < 30:
            return [1.2, 0.8, 0.4]  # 开始买房
        elif 30 <= age < 40:
            return [0.8, 1.2, 0.8]  # 买房高峰期
        elif 40 <= age < 50:
            return [0.7, 1.0, 1.2]  # 开始还清贷款
        else:
            return [0.6, 0.7, 1.5]  # 更可能已还清贷款
    
    # 获取基础概率
    base_probs = np.array(income_base_probs[income])
    
    # 应用城市难度系数
    difficulty = location_difficulty[current_location]
    adjusted_probs = np.array([
        base_probs[0] * difficulty,  # 无房概率增加
        base_probs[1],              # 有贷款概率保持
        base_probs[2] / difficulty  # 无贷款概率降低
    ])
    
    # 应用年龄影响
    age_factors = np.array(get_age_factor(age))
    final_probs = adjusted_probs * age_factors
    
    # 归一化
    final_probs = final_probs / final_probs.sum()
    
    return final_probs

def generate_property_status(incomes, current_locations, ages):
    """为给定的收入和居住地生成房产状态"""
    property_status_levels = ['无房产', '有房有贷款', '有房无贷款']
    property_status = []
    
    for income, location, age in zip(incomes, current_locations, ages):
        probs = get_property_probabilities(income, location, age)
        status = np.random.choice(property_status_levels, p=probs)
        property_status.append(status)
    
    return property_status

def get_height_params(age):
    """
    根据年龄获取身高的均值和标准差参数
    考虑儿童青少年的生长发育特点
    """
    if age < 6:
        # 参考儿童生长曲线
        mean = 80 + age * 7  # 粗略估计，1岁80cm左右，每年增长7cm
        std = 5
    elif 6 <= age < 14:
        # 青少年快速生长期
        mean = 115 + (age - 6) * 5  # 粗略估计，每年增长5cm
        std = 6
    elif 14 <= age < 18:
        # 青春期生长期
        mean = 155 + (age - 14) * 3  # 增速放缓
        std = 6
    elif 18 <= age < 20:
        # 后青春期
        mean = 172
        std = 6
    elif 20 <= age < 30:
        # 90后
        mean = 174
        std = 6
    elif 30 <= age < 40:
        # 80后
        mean = 173
        std = 6
    elif 40 <= age < 50:
        # 70后
        mean = 171
        std = 6
    elif 50 <= age < 60:
        # 60后
        mean = 170
        std = 6
    else:
        # 60后以前
        mean = 168
        std = 6
    
    return mean, std

def generate_height(ages):
    """为给定的年龄生成身高"""
    heights = []
    
    for age in ages:
        mean, std = get_height_params(age)
        # 使用截断正态分布，限制身高在合理范围内
        min_height = max(mean - 3*std, 60)  # 确保儿童身高不会太低
        max_height = min(mean + 3*std, 200)  # 确保成年人身高不会太高
        
        height = np.random.normal(mean, std)
        height = np.clip(height, min_height, max_height)
        height = round(height)
        heights.append(height)
    
    return heights


def get_marital_status_probabilities(age):
    """
    根据年龄返回婚姻状况的概率分布
    marital_status = ['未婚', '离异无孩子', '离异有孩子', '已婚']
    """
    if age < 22:
        probs = [0.99, 0.0, 0.0, 0.01]
    elif 22 <= age < 26:
        probs = [0.95, 0.02, 0.01, 0.02]
    elif 26 <= age < 30:
        probs = [0.80, 0.05, 0.03, 0.12]
    elif 30 <= age < 35:
        probs = [0.60, 0.08, 0.07, 0.25]
    elif 35 <= age < 40:
        probs = [0.40, 0.10, 0.15, 0.35]
    elif 40 <= age < 50:
        probs = [0.25, 0.12, 0.23, 0.40]
    else:
        probs = [0.15, 0.13, 0.27, 0.45]
    
    # 确保概率和为1
    return np.array(probs) / np.sum(probs)

def get_health_status_probabilities(age):
    """
    根据年龄返回健康状况的概率分布
    health_status = ['健康', '亚健康', '慢性病', '重大疾病']
    """
    if age < 25:
        probs = [0.90, 0.08, 0.015, 0.005]
    elif 25 <= age < 35:
        probs = [0.85, 0.12, 0.03, 0.01]
    elif 35 <= age < 45:
        probs = [0.75, 0.15, 0.08, 0.02]
    elif 45 <= age < 55:
        probs = [0.65, 0.20, 0.12, 0.03]
    elif 55 <= age < 65:
        probs = [0.50, 0.25, 0.20, 0.05]
    else:
        probs = [0.35, 0.30, 0.28, 0.07]
    
    # 确保概率和为1
    return np.array(probs) / np.sum(probs)

def generate_personal_status(ages):
    """生成婚姻状况和健康状况"""
    marital_status_levels = ['未婚', '离异无孩子', '离异有孩子', '已婚']
    health_status_levels = ['健康', '亚健康', '慢性病', '重大疾病']
    marital_status = []
    health_status = []
    
    for age in ages:
        # 生成婚姻状况
        marital_probs = get_marital_status_probabilities(age)
        status = np.random.choice(marital_status_levels, p=marital_probs)
        marital_status.append(status)
        
        # 生成健康状况
        health_probs = get_health_status_probabilities(age)
        health = np.random.choice(health_status_levels, p=health_probs)
        health_status.append(health)
    
    return marital_status, health_status

def generate_independent_features(n_samples):
    """生成其他独立的特征"""
    # 其他分类特征
    return {
        'religion': np.random.choice(
            ['无信仰', '有宗教信仰'],
            size=n_samples,
            p=[0.95, 0.05]
        ),
        'smoking_habit': np.random.choice(
            ['不吸烟', '偶尔吸烟', '经常吸烟'],
            size=n_samples,
            p=[0.70, 0.20, 0.10]
        ),
        'drinking_habit': np.random.choice(
            ['禁酒', '偶尔喝', '经常喝'],
            size=n_samples,
            p=[0.50, 0.40, 0.10]
        ),
    }


def get_height_attraction_factor(height):
    """
    计算身高对性吸引力的影响因子
    180以上：极高加成
    175-180：高加成
    170-175：正常
    170以下：略微降低
    """
    if height >= 180:
        return 1.5  # 显著提升性吸引力
    elif 175 <= height < 180:
        return 1.3  # 较高提升
    elif 170 <= height < 175:
        return 1.0  # 标准水平
    elif 165 <= height < 170:
        return 0.9  # 略微降低
    else:
        return 0.8  # 明显降低

def get_age_attraction_factor(age):
    """
    计算年龄对性吸引力的影响因子
    考虑男性在不同年龄段的魅力特点
    """
    if 25 <= age < 35:
        return 1.2  # 黄金年龄段
    elif 35 <= age < 45:
        return 1.1  # 成熟魅力
    elif 45 <= age < 55:
        return 1.0  # 标准水平
    elif 55 <= age < 65:
        return 0.9  # 略微下降
    else:
        return 0.8  # 明显下降

def calculate_sex_attract_score(
        face_score, body_score, humor_score, height, age):
    """
    计算性吸引力得分
    权重: 
    - 颜值 0.3
    - 身材 0.2
    - 身高影响因子 0.3
    - 幽默感 0.2
    最后乘以年龄影响因子
    """
    # 基础分数计算
    base_score = (0.3 * face_score + 
                 0.2 * body_score + 
                 0.2 * humor_score)
    
    # 身高影响
    height_factor = get_height_attraction_factor(height)
    height_component = 0.3 * height_factor * 5  # 将身高影响标准化到5分制
    
    # 合并所有组件
    weighted_score = base_score + height_component
    
    # 应用年龄因子
    age_factor = get_age_attraction_factor(age)
    weighted_score *= age_factor
    
    # 加入随机波动（±0.3分）
    random_factor = np.random.uniform(-0.3, 0.3)
    final_score = round(weighted_score + random_factor)
    
    # 确保分数在1-5范围内
    return np.clip(final_score, 1, 5)

def generate_scores(n_samples, ages, heights):
    """生成所有评分特征"""
    # 生成基础分数
    basic_scores = {
        'face_score': np.random.randint(1, 6, size=n_samples),
        'humor_score': np.random.randint(1, 6, size=n_samples),
        'body_score': np.random.randint(1, 6, size=n_samples)
    }
    
    # 计算性吸引力得分
    sex_attract_scores = []
    for i in range(n_samples):
        score = calculate_sex_attract_score(
            basic_scores['face_score'][i],
            basic_scores['body_score'][i],
            basic_scores['humor_score'][i],
            heights[i],
            ages[i]
        )
        sex_attract_scores.append(score)
    
    return {
        **basic_scores,
        'sex_attract_score': sex_attract_scores
    }

def get_vision_probabilities(education):
    """
    根据教育程度返回视力状况的概率分布
    vision_status = ['不近视', '近视低于400度', '近视高于400度']
    """
    if education == '研究生及以上':
        # 高学历群体近视比例最高
        return np.array([0.15, 0.45, 0.40])
    elif education == '本科':
        # 本科生近视比例也较高
        return np.array([0.20, 0.50, 0.30])
    elif education == '大专':
        # 中等教育近视比例适中
        return np.array([0.30, 0.50, 0.20])
    else:  # 高中及以下
        # 低教育程度近视比例较低
        return np.array([0.50, 0.40, 0.10])

def calculate_personal_asset(age, income, education, current_location):
    """
    计算个人总资产
    考虑：年收入、工作年限、储蓄率、生活成本、城市房产等因素
    """
    # 收入等级到月收入的映射（取区间中位数）
    income_to_monthly = {
        '<5万': 0.3,      # 月收入0.3万
        '5-15万': 0.8,    # 月收入0.8万
        '15-30万': 1.9,   # 月收入1.9万
        '30-50万': 3.3,   # 月收入3.3万
        '50-100万': 6.2,  # 月收入6.2万
        '>100万': 12.5    # 月收入12.5万
    }
    
    # 城市等级对应的月生活成本（单位：万）
    living_cost = {
        '一线城市': 1.0,
        '二线城市': 0.7,
        '三线城市': 0.5,
        '县城': 0.3,
        '农村': 0.2
    }
    
    # 教育水平对应的基础储蓄率
    education_saving_rate = {
        '研究生及以上': 0.35,
        '本科': 0.30,
        '大专': 0.25,
        '高中及以下': 0.20
    }
    
    # 城市房产基准价值（单位：万）
    base_house_value = {
        '一线城市': 500,
        '二线城市': 300,
        '三线城市': 150,
        '县城': 80,
        '农村': 30
    }
    
    # 基础计算
    monthly_income = income_to_monthly[income]
    monthly_cost = living_cost[current_location]
    saving_rate = education_saving_rate[education]
    working_years = max(0, age - 22) if age > 22 else 0
    
    # 计算积累的现金资产
    monthly_saving = (monthly_income - monthly_cost) * saving_rate
    if monthly_saving < 0:  # 如果收入不足以支付生活成本
        monthly_saving = monthly_income * 0.1  # 假设至少储蓄10%
    
    cash_asset = monthly_saving * 12 * working_years
    
    # 房产资产（工作满5年且月收入大于生活成本2倍时考虑购房）
    house_asset = 0
    if working_years >= 5 and monthly_income > monthly_cost * 2:
        house_value = base_house_value[current_location]
        # 随机确定是否有房
        # 工作年限越长，有房概率越大，最高80%
        if np.random.random() < min(0.8, working_years/30):  
            house_asset = house_value * (1 + 0.03 * working_years)  # 每年3%房价增值
    
    # 投资资产（假设有一定比例的现金用于投资）
    investment_ratio = min(0.4, working_years/20)  # 工作年限越长，投资比例越大，最高40%
    # 投资回报率-20%到40%
    investment_return = cash_asset * investment_ratio * np.random.uniform(
        -0.2, 0.4)  
    
    # 总资产 = 现金 + 房产 + 投资收益
    total_asset = cash_asset + house_asset + investment_return
    
    # 加入小幅随机波动（±10%）
    total_asset *= np.random.uniform(0.9, 1.1)
    
    # 返回资产范围
    if total_asset < 10:
        return '<10万'
    elif total_asset < 50:
        return '10-50万'
    elif total_asset < 200:
        return '50-200万'
    elif total_asset < 500:
        return '200-500万'
    elif total_asset < 1000:
        return '500-1000万'
    else:
        return '>1000万'

def generate_vision_and_assets(ages, education_levels, incomes, current_locations):
    """生成视力状况和个人总资产"""
    vision_status = []
    personal_assets = []
    
    for age, edu, income, location in zip(
        ages, education_levels, incomes, current_locations):
        # 生成视力状况
        vision_probs = get_vision_probabilities(edu)
        vision = np.random.choice(
            ['不近视', '近视低于400度', '近视高于400度'],
            p=vision_probs
        )
        vision_status.append(vision)
        
        # 生成个人总资产
        asset = calculate_personal_asset(age, income, edu, location)
        personal_assets.append(asset)
    
    return vision_status, personal_assets


def export_data(df):
    """导出数据并保存映射"""
    data = df.copy()
    
    # 为有序变量定义映射
    ordered_mappings = {
        'education': {
            '高中及以下': 0,
            '大专': 1,
            '本科': 2,
            '研究生及以上': 3
        },
        'income': {
            '<5万': 0,
            '5-15万': 1,
            '15-30万': 2,
            '30-50万': 3,
            '50-100万': 4,
            '>100万': 5
        },
        'personal_assets': {
            '<10万': 0,
            '10-50万': 1,
            '50-200万': 2,
            '200-500万': 3,
            '500-1000万': 4,
            '>1000万': 5
        },
        'hometown': {
            '农村': 0,
            '县城': 1,
            '三线城市': 2,
            '二线城市': 3,
            '一线城市': 4
        },
        'current_location': {
            '农村': 0,
            '县城': 1,
            '三线城市': 2,
            '二线城市': 3,
            '一线城市': 4
        },
        'vision': {
            '不近视': 0,
            '近视低于400度': 1,
            '近视高于400度': 2
        },
        'health_status': {
            '健康': 0,
            '亚健康': 1,
            '慢性病': 2,
            '重大疾病': 3
        },
        'marital_status': {
            '未婚': 0,
            '离异无孩子': 1,
            '离异有孩子': 2,
            '已婚': 3
        },
        'property_status': {
            '无房产': 0,
            '有房有贷款': 1,
            '有房无贷款': 2
        },
        'smoking_habit': {
            '不吸烟': 0,
            '偶尔吸烟': 1,
            '经常吸烟': 2
        },
        'drinking_habit': {
            '禁酒': 0,
            '偶尔喝': 1,
            '经常喝': 2
        },
        'face_score': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
        'humor_score': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
        'sex_attract_score': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
        'body_score': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    }
    
    # 非有序变量的映射
    categorical_mappings = {
        'religion': {
            '无信仰': 0,
            '有宗教信仰': 1
        }
    }
    
    mappings = {**ordered_mappings, **categorical_mappings}
    
    # 应用映射
    for col, mapping in mappings.items():
        data[col] = data[col].map(mapping)
    
    # 变量描述和说明
    variable_descriptions = {
        'age': {
            'description': '年龄（连续值）',
            'type': 'continuous',
            'range': '0-100岁',
            'comments': '基于2022年人口抽样调查数据生成'
        },
        'height': {
            'description': '身高（厘米）',
            'type': 'continuous',
            'range': '150-200cm',
            'comments': '考虑年龄对身高的影响'
        },
        'education': {
            'description': '教育程度（有序）',
            'type': 'ordinal',
            'values': list(ordered_mappings['education'].keys()),
            'comments': '基于全国教育程度分布数据'
        },
        'income': {
            'description': '年收入（有序）',
            'type': 'ordinal',
            'values': list(ordered_mappings['income'].keys()),
            'comments': '考虑年龄、教育程度对收入的影响'
        },
        'personal_assets': {
            'description': '个人总资产（有序）',
            'type': 'ordinal',
            'values': list(ordered_mappings['personal_assets'].keys()),
            'comments': '考虑年龄、收入、城市等级的综合影响'
        },
        'hometown': {
            'description': '家乡所在地（有序）',
            'type': 'ordinal',
            'values': list(ordered_mappings['hometown'].keys()),
            'comments': '按城市等级排序'
        },
        'current_location': {
            'description': '当前居住地（有序）',
            'type': 'ordinal',
            'values': list(ordered_mappings['current_location'].keys()),
            'comments': '与hometown共同反映人口流动'
        },
        'vision': {
            'description': '视力状况（有序）',
            'type': 'ordinal',
            'values': list(ordered_mappings['vision'].keys()),
            'comments': '与教育程度相关'
        },
        'health_status': {
            'description': '健康状况（有序）',
            'type': 'ordinal',
            'values': list(ordered_mappings['health_status'].keys()),
            'comments': '与年龄相关'
        },
        'marital_status': {
            'description': '婚姻状况（有序）',
            'type': 'ordinal',
            'values': list(ordered_mappings['marital_status'].keys()),
            'comments': '与年龄相关'
        },
        'property_status': {
            'description': '房产状况（有序）',
            'type': 'ordinal',
            'values': list(ordered_mappings['property_status'].keys()),
            'comments': '与收入、年龄、城市等级相关'
        },
        'smoking_habit': {
            'description': '吸烟习惯（有序）',
            'type': 'ordinal',
            'values': list(ordered_mappings['smoking_habit'].keys())
        },
        'drinking_habit': {
            'description': '饮酒习惯（有序）',
            'type': 'ordinal',
            'values': list(ordered_mappings['drinking_habit'].keys())
        },
        'face_score': {
            'description': '颜值评分（有序）',
            'type': 'ordinal',
            'values': [1,2,3,4,5],
            'comments': '影响sex_attract_score'
        },
        'humor_score': {
            'description': '幽默感评分（有序）',
            'type': 'ordinal',
            'values': [1,2,3,4,5],
            'comments': '影响sex_attract_score'
        },
        'sex_attract_score': {
            'description': '性吸引力评分（有序）',
            'type': 'ordinal',
            'values': [1,2,3,4,5],
            'comments': '由face_score、body_score、humor_score、height和age共同影响'
        },
        'body_score': {
            'description': '身材评分（有序）',
            'type': 'ordinal',
            'values': [1,2,3,4,5],
            'comments': '影响sex_attract_score'
        },
        'religion': {
            'description': '宗教信仰（分类）',
            'type': 'categorical',
            'values': list(categorical_mappings['religion'].keys())
        }
    }
    
    # 保存详细的映射信息
    mapping_info = {
        'mappings': mappings,
        'ordered_variables': list(ordered_mappings.keys()),
        'categorical_variables': list(categorical_mappings.keys()),
        'continuous_variables': ['age', 'height'],
        'variable_descriptions': variable_descriptions,
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }
    
    # 保存数据和映射
    data.to_csv("data/data_numeric.csv", index=False)
    data.to_parquet("data/data_numeric.parquet", index=False, compression="snappy")
    data.to_json("data/data_numeric.json", orient="records", force_ascii=False)
    
    with open("data/mappings.json", "w", encoding="utf-8") as f:
        json.dump(mapping_info, f, ensure_ascii=False, indent=4)
    
    return data, mapping_info

if __name__=="__main__":
    age_distribution = {
        '0-4': 32589,
        '5-9': 48010,
        '10-14': 49065,
        '15-19': 42687,
        '20-24': 39146,
        '25-29': 44806,
        '30-34': 60488,
        '35-39': 56397,
        '40-44': 50422,
        '45-49': 52930,
        '50-54': 64419,
        '55-59': 58880,
        '60-64': 35813,
        '65-69': 39343,
        '70-74': 28355,
        '75-79': 16956,
        '80-84': 10137,
        '85-89': 5223,
        '90-94': 1654,
        '95-99': 311
    }
    age_ranges = range(0, 100)
    n_samples = 1_000_000 # 1M
    hometown_probs = {
            '一线城市': 0.1,
            '二线城市': 0.15,
            '三线城市': 0.2,
            '县城': 0.25,
            '农村': 0.3
        }
    age_probs = get_age_probs(age_distribution)
    ages = sample_ages(age_ranges, n_samples, age_probs)
    hometowns = sample_hometowns(hometown_probs, n_samples)
    education = generate_education(ages, hometowns)
    income = generate_income(ages, education_levels = education)
    current_location = generate_current_location(
        hometowns, 
        education_levels=education)
    property_status = generate_property_status(income, current_location, ages)
    heights = generate_height(ages)
    # 生成婚姻和健康状况
    marital_status, health_status = generate_personal_status(ages)
    # 生成其他独立特征
    independent_features = generate_independent_features(n_samples)
    scores = generate_scores(n_samples, ages, heights)
    vision, personal_assets = generate_vision_and_assets(
            ages, education, income, current_location
        )

    df = pd.DataFrame({
        'age': ages,
        'height': heights,
        'hometown': hometowns,
        'education': education,
        'income': income,
        'current_location': current_location,
        'property_status': property_status,
        'marital_status': marital_status,
        'health_status': health_status,
        'vision': vision,
        'personal_assets': personal_assets,
        **independent_features,
        **scores
    })

    data, mapping_info = export_data(df)

