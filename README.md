# 中国男性人口合成数据生成

## 目录

1. [简介](#简介)
2. [数据集概览](#数据集概览)
   - [数据集用途](#数据集用途)
   - [主要特征](#主要特征)
3. [数据生成过程](#数据生成过程)
   - [年龄分布](#年龄分布)
   - [身高分布](#身高分布)
   - [教育水平](#教育水平)
   - [收入水平](#收入水平)
   - [地理流动性](#地理流动性)
   - [房产状况](#房产状况)
   - [健康和婚姻状况](#健康和婚姻状况)
   - [视力和个人资产](#视力和个人资产)
   - [个人评分（颜值、幽默感等）](#个人评分颜值幽默感等)
4. [数据验证](#数据验证)
   - [与真实统计数据的对比](#与真实统计数据的对比)
   - [合成数据的洞察](#合成数据的洞察)
5. [导出文件](#导出文件)
   - [文件格式](#文件格式)
   - [数据映射信息](#数据映射信息)
6. [使用说明](#使用说明)
   - [如何加载数据](#如何加载数据)
   - [分析示例](#分析示例)
7. [模型限制](#模型限制)
8. [参考文献](#参考文献)


## 简介

本项目旨在生成一个包含 **100 万条数据** 的中国男性人口合成数据集。数据集设计基于真实人口统计数据，通过条件概率分布和随机采样技术生成，涵盖了以下具体特征：

1. **年龄**：基于全国人口年龄分布的抽样数据生成，确保样本符合实际人口年龄段比例。
2. **身高**：按年龄段定义身高的均值和标准差，结合正态分布模拟不同年龄段的身高差异。
3. **家乡**：作为独立变量，通过随机采样生成，包括“农村”、“县城”、“三线城市”、“二线城市”、“一线城市”五类。
4. **教育水平**：考虑年龄与家乡对教育水平的影响，通过 $P(\text{教育水平}|\text{年龄}, \text{家乡})$ 模拟生成，教育水平包括“高中及以下”、“大专”、“本科”、“研究生及以上”四类。
5. **收入**：基于年龄与教育水平的组合，通过 $P(\text{收入}|\text{年龄}, \text{教育水平})$ 模拟年收入分布，收入档次包括“<5万”、“5-15万”、“15-30万”、“30-50万”、“50-100万”、“>100万”。
6. **现居住地**：结合家乡与教育水平，通过 $P(\text{现居住地}|\text{家乡}, \text{教育水平})$ 模拟生成，现居住地包括“农村”、“县城”、“三线城市”、“二线城市”、“一线城市”五类。
7. **房产状况**：通过 $P(\text{房产状况}|\text{年龄}, \text{收入}, \text{现居住地})$ 生成房产拥有情况，包括“无房产”、“有房有贷款”、“有房无贷款”三类。
8. **健康状况**：模拟年龄与健康之间的关系，通过 $P(\text{健康状况}|\text{年龄})$ 生成，包括“健康”、“亚健康”、“慢性病”、“重大疾病”四类。
9. **婚姻状况**：模拟年龄与婚姻之间的关系，通过 $P(\text{婚姻状况}|\text{年龄})$ 生成，包括“未婚”、“离异无孩子”、“离异有孩子”、“已婚”四类。
10. **视力**：结合教育水平生成视力分布，通过 $P(\text{视力}|\text{教育水平})$ 模拟，包括“不近视”、“近视低于400度”、“近视高于400度”三类。
11. **个人资产**：结合年龄、收入、教育水平和现居住地，通过 $P(\text{个人资产}|\text{年龄}, \text{收入}, \text{教育水平}, \text{现居住地})$ 计算，包括“<10万”、“10-50万”、“50-200万”、“200-500万”、“500-1000万”、“>1000万”。
12. **生活习惯**：
    - **吸烟习惯**：通过随机生成，包括“不吸烟”、“偶尔吸烟”、“经常吸烟”三类。
    - **饮酒习惯**：通过随机生成，包括“禁酒”、“偶尔喝”、“经常喝”三类。
    - **宗教信仰**：通过随机生成，包括“无信仰”、“有宗教信仰”两类。
13. **个人评分**：生成以下具体评分：
    - **颜值评分**：随机生成，范围为1到5分。
    - **幽默感评分**：随机生成，范围为1到5分。
    - **身材评分**：随机生成，范围为1到5分。
    - **性吸引力评分**：通过 $P(\text{性吸引力评分}|\text{颜值评分}, \text{身材评分}, \text{幽默感评分}, \text{身高}, \text{年龄})$ 计算，范围为1到5分。

### 项目特色

- **真实感**：所有变量的分布均基于实际统计数据或合理假设，变量间的依赖关系通过条件概率建模。
- **覆盖全面**：涵盖年龄、身高、家乡、现居住地、教育水平、收入、房产状况、健康状况、婚姻状况、视力、个人资产、生活习惯（吸烟习惯、饮酒习惯、宗教信仰）以及个人评分（颜值评分、幽默感评分、身材评分、性吸引力评分）。
- **灵活易用**：提供多种格式的数据文件（CSV、JSON、Parquet），便于加载和分析。

本项目适合以下用途：
1. **教学与研究**：用于机器学习模型训练、统计分析和数据可视化课程的示例数据。
2. **模型测试**：验证分类、回归、聚类等算法在复杂数据集上的性能。
3. **数据探索**：进行人口特征的探索性分析或创建交互式可视化。

下一部分将详细说明数据集结构和每个变量的生成逻辑。

## 年龄

### 数据来源

年龄分布基于全国人口统计数据，结合 2022 年中国人口抽样调查结果（千分之一样本），按以下每个年龄段的实际人数生成。

| 年龄段  | 人数   |
|---------|--------|
| 0-4     | 32589  |
| 5-9     | 48010  |
| 10-14   | 49065  |
| 15-19   | 42687  |
| 20-24   | 39146  |
| 25-29   | 44806  |
| 30-34   | 60488  |
| 35-39   | 56397  |
| 40-44   | 50422  |
| 45-49   | 52930  |
| 50-54   | 64419  |
| 55-59   | 58880  |
| 60-64   | 35813  |
| 65-69   | 39343  |
| 70-74   | 28355  |
| 75-79   | 16956  |
| 80-84   | 10137  |
| 85-89   | 5223   |
| 90-94   | 1654   |
| 95-99   | 311    |

### 生成方法

1. **年龄段分布**：直接使用统计数据，计算每个年龄段的概率分布。
2. **具体年龄分布**：假设每个年龄段内的年龄均匀分布，细化到每岁的概率。
3. **随机采样**：按照计算的概率分布生成目标样本数量的年龄数据。

#### Python 实现代码

```python
# 年龄分布数据
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

# 将年龄段人数转换为每岁概率分布
def get_age_probs(age_distribution):
    total_population = sum(age_distribution.values())
    age_probs = {}
    for age_group, group_count in age_distribution.items():
        group_prob = group_count / total_population
        start, end = map(int, age_group.split('-'))
        for i in range(start, end + 1):
            age_probs[i] = group_prob / (end - start + 1)  # 平均分配到每岁
    return age_probs

# 采样
def sample_ages(age_ranges, n_samples, age_probs):
    return np.random.choice(
        age_ranges,
        size=n_samples,
        p=list(age_probs.values())
    )

# 定义年龄范围和样本量
age_ranges = range(0, 100)
n_samples = 1_000_000  # 样本总量

# 计算每岁的概率分布并采样
age_probs = get_age_probs(age_distribution)
ages = sample_ages(age_ranges, n_samples, age_probs)
```

### 验证分布 （待完成）

生成的数据经过对比验证，确保与统计数据的年龄分布一致。以下为生成数据与统计数据的对比图：

- 左图：生成数据的年龄分布直方图。

- 右图：生成数据与统计数据的年龄段对比条形图。

```py
import matplotlib.pyplot as plt

# 汇总生成的每岁年龄到对应年龄段
def summarize_generated_age_distribution(ages, age_distribution):
    generated_counts = {age: list(ages).count(age) for age in range(0, 100)}
    generated_age_distribution = {}
    for age_group in age_distribution.keys():
        start, end = map(int, age_group.split('-'))
        generated_age_distribution[age_group] = sum(
            generated_counts.get(age, 0) for age in range(start, end + 1)
        )
    return generated_age_distribution

# 汇总结果
generated_age_distribution = summarize_generated_age_distribution(ages, age_distribution)

# 对比实际分布与生成分布
actual_counts = list(age_distribution.values())
generated_counts = list(generated_age_distribution.values())
age_groups = list(age_distribution.keys())

# 绘制图表
plt.figure(figsize=(12, 5))

# 左图：生成的年龄分布
plt.subplot(1, 2, 1)
plt.hist(ages, bins=100, color='skyblue', alpha=0.7, edgecolor='black')
plt.xlabel('年龄')
plt.ylabel('人数')
plt.title('生成数据的年龄分布')

# 右图：年龄段对比
plt.subplot(1, 2, 2)
plt.bar(age_groups, actual_counts, alpha=0.6, label='实际分布', color='orange')
plt.bar(age_groups, generated_counts, alpha=0.6, label='生成分布', color='blue')
plt.xlabel('年龄段')
plt.ylabel('人数')
plt.xticks(rotation=45)
plt.legend()
plt.title('实际分布与生成分布的对比')

plt.tight_layout()
plt.show()
```

## 身高

### 数据来源

身高分布基于不同年龄段的人体生长规律。通过定义各年龄段的身高均值和标准差，结合正态分布模拟真实的身高分布。以下是生成逻辑：

1. **儿童阶段（0-6 岁）**：
   - 基准均值随年龄递增，1 岁约 80 cm，每年增长约 7 cm。
   - 标准差为 5 cm。
2. **青少年阶段（6-18 岁）**：
   - 身高增长逐步放缓：
     - 6-14 岁每年增长约 5 cm。
     - 14-18 岁每年增长约 3 cm。
   - 标准差为 6 cm。
3. **成年人阶段（18 岁及以上）**：
   - 不同年龄段的身高均值逐渐下降：
     - 18-20 岁均值为 172 cm。
     - 20-30 岁均值为 174 cm，随后每 10 年略微降低。
   - 标准差均为 6 cm。
4. **老年阶段（60 岁及以上）**：
   - 均值降至 168 cm，身高略微下降。

---

### 生成方法

1. **按年龄段定义身高分布**：
   - 不同年龄段的均值和标准差如下表所示：

| 年龄段         | 均值（cm）         | 标准差（cm） |
|----------------|--------------------|-------------|
| 0-6 岁         | $80 + \text{age} \times 7$ | 5           |
| 6-14 岁        | $115 + (\text{age} - 6) \times 5$ | 6           |
| 14-18 岁       | $155 + (\text{age} - 14) \times 3$ | 6           |
| 18-20 岁       | 172                | 6           |
| 20-30 岁       | 174                | 6           |
| 30-40 岁       | 173                | 6           |
| 40-50 岁       | 171                | 6           |
| 50-60 岁       | 170                | 6           |
| 60 岁及以上    | 168                | 6           |

2. **正态分布生成**：
   - 使用定义的均值和标准差，为每个年龄生成对应的身高。
   - 为避免极端值，使用截断正态分布，限制范围为 [60, 200] cm。

3. **随机生成**：
   - 通过 `np.random.normal` 随机生成每个样本的身高，并使用 `np.clip` 截断至合理范围。

---

### 示例代码

以下为实现逻辑的代码片段：

```python
def get_height_params(age):
    """根据年龄获取身高的均值和标准差"""
    if age < 6:
        mean = 80 + age * 7
        std = 5
    elif 6 <= age < 14:
        mean = 115 + (age - 6) * 5
        std = 6
    elif 14 <= age < 18:
        mean = 155 + (age - 14) * 3
        std = 6
    elif 18 <= age < 20:
        mean = 172
        std = 6
    elif 20 <= age < 30:
        mean = 174
        std = 6
    elif 30 <= age < 40:
        mean = 173
        std = 6
    elif 40 <= age < 50:
        mean = 171
        std = 6
    elif 50 <= age < 60:
        mean = 170
        std = 6
    else:
        mean = 168
        std = 6
    return mean, std

def generate_height(ages):
    """为给定年龄生成身高"""
    heights = []
    for age in ages:
        mean, std = get_height_params(age)
        min_height = max(mean - 3 * std, 60)
        max_height = min(mean + 3 * std, 200)
        height = np.random.normal(mean, std)
        height = np.clip(height, min_height, max_height)
        heights.append(round(height))
    return heights
```


## 家乡

### 数据来源

家乡分布基于经验数据，考虑中国不同区域的人口比例和城镇化水平。家乡被定义为个体的出生地或长期生活地，分为以下五类：

| 家乡分类     | 比例 (%) | 
|--------------|----------|
| 农村         | 30.0     |
| 县城         | 25.0     |
| 三线城市     | 20.0     |
| 二线城市     | 15.0     |
| 一线城市     | 10.0     |

### 生成方法

1. **确定家乡类别**：按照以上比例分布随机生成家乡。
2. **随机采样**：使用 `numpy.random.choice` 按概率生成目标样本的家乡类别。

#### Python 实现代码

```python
# 家乡分布数据
hometown_probs = {
    '农村': 0.30,
    '县城': 0.25,
    '三线城市': 0.20,
    '二线城市': 0.15,
    '一线城市': 0.10
}

# 采样
def sample_hometowns(hometown_probs, n_samples):
    return np.random.choice(
        list(hometown_probs.keys()),  # 家乡类别
        size=n_samples,              # 样本总量
        p=list(hometown_probs.values())  # 每类的生成概率
    )

# 生成年龄对应的家乡分布
hometowns = sample_hometowns(hometown_probs, n_samples=1_000_000)
```

### 验证分布 （待完成）

生成的数据经过对比验证，确保与家乡的目标分布一致。以下为生成分布与目标分布的对比条形图：

```python
import matplotlib.pyplot as plt

# 汇总生成数据的家乡分布
from collections import Counter
generated_counts = Counter(hometowns)
categories = list(hometown_probs.keys())
actual_counts = [p * len(hometowns) for p in hometown_probs.values()]
generated_counts_list = [generated_counts[category] for category in categories]

# 绘制对比图
plt.figure(figsize=(8, 5))
x = range(len(categories))
plt.bar(x, actual_counts, alpha=0.6, label='目标分布', color='orange')
plt.bar(x, generated_counts_list, alpha=0.6, label='生成分布', color='blue')
plt.xticks(x, categories)
plt.xlabel('家乡类别')
plt.ylabel('人数')
plt.title('家乡分布目标与生成对比')
plt.legend()
plt.tight_layout()
plt.show()
```

## 教育水平

### 数据来源

教育水平分布基于中国全国教育统计数据。教育水平分为以下四类：

| 教育水平     | 全国总体比例 (%) | 
|--------------|------------------|
| 高中及以下   | 80.66            |
| 大专         | 9.69             |
| 本科         | 8.13             |
| 研究生及以上 | 0.93             |

实际生成中，教育水平受到以下因素的影响：
1. **年龄**：年轻人接受高等教育的比例更高。
2. **家乡**：发达地区（如一线城市）的高学历比例更高。

### 生成方法

1. **基准分布**：以全国总体分布为基础，确定每类教育水平的初始比例。
2. **年龄调整**：为不同年龄段设置教育水平的修正系数，模拟年龄对学历的影响。
3. **家乡调整**：根据家乡类别设置修正系数，模拟地域对教育水平的影响。
4. **概率归一化**：结合基准分布、年龄修正系数和家乡修正系数，计算最终的条件概率分布。
5. **随机采样**：按照条件概率分布生成教育水平数据。

### 代码实现

```py
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
```

### 收入

#### 数据来源

收入分布基于中国劳动力市场的经验数据，同时结合以下关键因素生成：

1. **年龄**：
   - 在职人员（18-59岁）的收入分布会受到年龄修正系数的动态影响。
   - 60 岁及以上人群的收入分布直接取决于其学历。
2. **教育水平**：
   - 不同学历的收入分布在所有年龄段均保持一致，学历越高，高收入档次占比越大。

#### 收入档次划分

| 收入档次      | 定义              |
|---------------|-------------------|
| `<5万`        | 年收入低于 5 万   |
| `5-15万`      | 年收入在 5 到 15 万之间 |
| `15-30万`     | 年收入在 15 到 30 万之间 |
| `30-50万`     | 年收入在 30 到 50 万之间 |
| `50-100万`    | 年收入在 50 到 100 万之间 |
| `>100万`      | 年收入高于 100 万 |

#### 生成方法

1. **学历的基础分布**：
   - 收入的静态分布直接由学历决定：
     - **研究生及以上**：高收入档次比例显著，例如 `[0.05, 0.35, 0.35, 0.15, 0.08, 0.02]`。
     - **高中及以下**：集中在低收入档次，例如 `[0.40, 0.45, 0.10, 0.04, 0.01, 0.00]`。

2. **年龄的动态修正（在职人员 18-59 岁）**：
   - 不同年龄段的修正系数：
     - **<25 岁**：以低收入为主，例如 `[1.5, 1.2, 0.6, 0.3, 0.1, 0.0]`。
     - **35-45 岁**：高收入比例达到最大，例如 `[0.5, 0.8, 1.0, 1.2, 1.1, 1.0]`。
     - **45-59 岁**：收入保持稳定，高收入档次维持较高比例。

3. **退休阶段（≥60 岁）**：
   - 60 岁及以上的收入分布直接采用学历的静态分布，无动态调整。

4. **条件概率计算**：
   - 结合学历的基础分布和年龄修正系数，计算每个收入档次的条件概率。
   - 概率归一化，确保总和为 1。

5. **随机采样**：
   - 按条件概率生成样本的收入档次数据。

### 代码实现

```py
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
```
## 现居住地

### 数据来源

现居住地的生成基于家乡所在地与教育水平的条件概率分布。我们模拟了从家乡（出生地）到当前居住地的迁移模式，结合教育水平的影响，生成样本的现居住地。现居住地包含以下五个类别：

1. **一线城市**（北上广深）。
2. **二线城市**（省会城市及计划单列市）。
3. **三线城市**（普通地级市）。
4. **县城**。
5. **农村**。

### 生成方法

现居住地的分布由以下几个关键因素决定：

1. **基础流动模式**：
   - 从家乡到当前居住地的流动矩阵，表示人口从某个家乡类型流动到目标居住地类型的基础概率。例如：
     - **农村**到**一线城市**的概率较低，而**一线城市**到**一线城市**的概率较高。

2. **教育水平的影响**：
   - 教育水平越高，迁移到高等级城市的概率越大。例如：
     - 具有“研究生及以上”学历的人更可能居住在一线或二线城市。
     - 具有“高中及以下”学历的人更可能居住在县城或农村。

3. **最终条件概率计算**：
   - 通过结合基础流动概率和教育水平的影响系数，生成条件概率 $P(\text{现居住地} \mid \text{家乡, 教育水平})$。

### 逻辑细节

1. **基础流动矩阵**：
   - 每一行表示从家乡迁移到现居住地的基础概率。例如：
     | 家乡        | 一线城市 | 二线城市 | 三线城市 | 县城 | 农村  |
     |-------------|----------|----------|----------|-------|-------|
     | 一线城市    | 0.80     | 0.15     | 0.03     | 0.01  | 0.01  |
     | 二线城市    | 0.20     | 0.65     | 0.10     | 0.03  | 0.02  |
     | 三线城市    | 0.15     | 0.20     | 0.55     | 0.07  | 0.03  |
     | 县城        | 0.10     | 0.15     | 0.20     | 0.45  | 0.10  |
     | 农村        | 0.05     | 0.10     | 0.15     | 0.30  | 0.40  |

2. **教育水平调整系数**：
   - 不同教育水平对迁移概率的修正系数。例如：
     - **研究生及以上**：
       - 一线城市：2.0。
       - 农村：0.1。
     - **高中及以下**：
       - 一线城市：0.5。
       - 农村：1.2。

3. **最终概率计算**：

- 使用以下公式计算条件概率：

$$
P(\text{现居住地} \mid \text{家乡, 教育水平}) = P(\text{现居住地} \mid \text{家乡}) \times \text{教育修正系数}
$$

- 并归一化以确保总和为 1。

### 示例代码

以下是用于生成现居住地的代码片段：

```python
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
```

## 房产状况

### 数据来源

房产状况模拟基于个体的 **收入**、**现居住地** 和 **年龄**。我们定义了三个房产状态：

1. **无房产**。
2. **有房有贷款**。
3. **有房无贷款**。

### 生成方法

房产状况的生成遵循以下逻辑：

1. **基础购房难度**：
   - 不同城市的房价和购房难度差异显著，按城市等级定义购房难度系数。例如：
     - 一线城市：购房难度系数为 2.5。
     - 农村：购房难度系数为 0.7。

2. **收入的影响**：
   - 收入档次决定了购房能力，每个收入档次对应的基础概率为：
     | 收入档次  | 无房产 | 有房有贷款 | 有房无贷款 |
     |-----------|--------|------------|------------|
     | <5万      | 80%    | 15%        | 5%         |
     | 5-15万    | 70%    | 25%        | 5%         |
     | 15-30万   | 50%    | 40%        | 10%        |
     | 30-50万   | 30%    | 50%        | 20%        |
     | 50-100万  | 20%    | 45%        | 35%        |
     | >100万    | 10%    | 35%        | 55%        |

3. **年龄的影响**：
   - 年龄影响购房意愿和还贷进度：
     - **<25岁**：无房产的比例较高。
     - **25-40岁**：买房高峰期，有房有贷款的比例增加。
     - **>40岁**：还贷进程加快，有房无贷款的比例显著增加。

4. **计算条件概率**：
   - 综合购房难度、收入和年龄影响系数，生成条件概率 $P(\text{房产状态} \mid \text{收入, 现居住地, 年龄})$。

### 逻辑细节

1. **购房难度调整**：
   - 每个城市等级的购房难度系数用于调整基础概率。例如：
     - 一线城市：无房产的概率乘以 2.5，有房无贷款的概率除以 2.5。

2. **收入基础概率**：
每个收入档次的基础概率为：

$$
\text{调整后概率} = 
\begin{bmatrix}
P(\text{无房产}) \times \text{购房难度系数} \\
P(\text{有房有贷款}) \\
P(\text{有房无贷款}) \div \text{购房难度系数}
\end{bmatrix}
$$

3. **年龄调整系数**：
   - 年龄对房产状态的调整系数为：
     | 年龄段      | 无房产 | 有房有贷款 | 有房无贷款 |
     |-------------|--------|------------|------------|
     | <25岁       | 1.5    | 0.3        | 0.1        |
     | 25-30岁     | 1.2    | 0.8        | 0.4        |
     | 30-40岁     | 0.8    | 1.2        | 0.8        |
     | 40-50岁     | 0.7    | 1.0        | 1.2        |
     | >50岁       | 0.6    | 0.7        | 1.5        |

$$
\text{年龄调整后的概率} = \text{调整后概率} \times \text{年龄系数}
$$

4. **条件概率计算公式**：

综合购房难度调整、收入基础概率和年龄调整系数：

$$
P(\text{房产状态} \mid \text{收入, 现居住地, 年龄}) = 
\frac{\text{年龄调整后的概率}}{\sum \text{年龄调整后的概率}}
$$

最终结果归一化为概率分布。


### 代码

```py
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
```




