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

本项目旨在生成一个包含 **100 万条数据** 的中国男性人口合成数据集。数据集设计基于真实人口统计数据，通过条件概率分布和随机采样技术生成，涵盖了多种人口统计学和社会经济特征，具体如下：

1. **年龄**：基于全国人口年龄分布的抽样数据生成，确保样本符合实际人口年龄段比例。
2. **身高**：按年龄段定义身高的均值和标准差，结合正态分布模拟不同年龄段的身高差异。
3. **教育水平**：考虑年龄与地域对教育水平的影响，通过 $P(\text{education}|\text{age}, \text{hometown})$ 模拟生成。
4. **收入**：基于年龄与教育水平的组合，通过 $P(\text{income}|\text{age}, \text{education})$ 模拟年收入分布。
5. **地理流动性**：结合家乡与当前居住地的流动模式，通过 $P(\text{current\_location}|\text{hometown}, \text{education})$ 模拟地理迁移情况。
6. **房产状况**：通过 $P(\text{property\_status}|\text{age}, \text{income}, \text{current\_location})$ 生成房产拥有情况，考虑收入、年龄和居住地对购房的影响。
7. **健康与婚姻状况**：模拟年龄与健康及婚姻状况之间的关系，通过 $P(\text{health\_status}|\text{age})$` 和 `$P(\text{marital\_status}|\text{age})$ 生成。
8. **视力和个人资产**：结合教育、收入、地域和年龄等变量，模拟 $P(\text{vision}|\text{education})$` 和 `$P(\text{personal\_assets}|\text{age}, \text{income}, \text{education}, \text{current\_location})$。
9. **个人评分**：生成如颜值、幽默感、性吸引力等评分数据，性吸引力通过 $P(\text{sex\_attract\_score}|\text{face\_score}, \text{body\_score}, \text{humor\_score}, \text{height}, \text{age})$ 计算。
10. **生活习惯和宗教信仰**：模拟吸烟习惯、饮酒习惯及宗教信仰，假设这些特征与其他变量独立。

### 项目特色

- **真实感**：所有变量的分布均基于实际统计数据或合理假设，变量间的依赖关系通过条件概率建模。
- **覆盖全面**：涵盖人口统计学、社会经济和个人评分等特征，适用于多种研究场景。
- **灵活易用**：提供多种格式的数据文件（CSV、JSON、Parquet），便于加载和分析。

本项目适合以下用途：
1. **教学与研究**：用于机器学习模型训练、统计分析和数据可视化课程的示例数据。
2. **模型测试**：验证分类、回归、聚类等算法在复杂数据集上的性能。
3. **数据探索**：进行人口特征的探索性分析或创建交互式可视化。

下一部分将详细说明数据集结构和每个变量的生成逻辑。


