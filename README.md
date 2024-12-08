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







