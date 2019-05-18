
# coding: utf-8

# ## 探索电影数据集
# 
# 在这个项目中，你将尝试使用所学的知识，使用 `NumPy`、`Pandas`、`matplotlib`、`seaborn` 库中的函数，来对电影数据集进行探索。
# 
# 下载数据集：
# [TMDb电影数据](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/explore+dataset/tmdb-movies.csv)
# 

# 
# 数据集各列名称的含义：
# <table>
# <thead><tr><th>列名称</th><th>id</th><th>imdb_id</th><th>popularity</th><th>budget</th><th>revenue</th><th>original_title</th><th>cast</th><th>homepage</th><th>director</th><th>tagline</th><th>keywords</th><th>overview</th><th>runtime</th><th>genres</th><th>production_companies</th><th>release_date</th><th>vote_count</th><th>vote_average</th><th>release_year</th><th>budget_adj</th><th>revenue_adj</th></tr></thead><tbody>
#  <tr><td>含义</td><td>编号</td><td>IMDB 编号</td><td>知名度</td><td>预算</td><td>票房</td><td>名称</td><td>主演</td><td>网站</td><td>导演</td><td>宣传词</td><td>关键词</td><td>简介</td><td>时常</td><td>类别</td><td>发行公司</td><td>发行日期</td><td>投票总数</td><td>投票均值</td><td>发行年份</td><td>预算（调整后）</td><td>票房（调整后）</td></tr>
# </tbody></table>
# 

# **请注意，你需要提交该报告导出的 `.html`、`.ipynb` 以及 `.py` 文件。**

# 
# 
# ---
# 
# ---
# 
# ## 第一节 数据的导入与处理
# 
# 在这一部分，你需要编写代码，使用 Pandas 读取数据，并进行预处理。

# 
# **任务1.1：** 导入库以及数据
# 
# 1. 载入需要的库 `NumPy`、`Pandas`、`matplotlib`、`seaborn`。
# 2. 利用 `Pandas` 库，读取 `tmdb-movies.csv` 中的数据，保存为 `movie_data`。
# 
# 提示：记得使用 notebook 中的魔法指令 `%matplotlib inline`，否则会导致你接下来无法打印出图像。

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')
movie_data = pd.read_csv('tmdb-movies.csv')
movie_data.head()


# ---
# 
# **任务1.2: ** 了解数据
# 
# 你会接触到各种各样的数据表，因此在读取之后，我们有必要通过一些简单的方法，来了解我们数据表是什么样子的。
# 
# 1. 获取数据表的行列，并打印。
# 2. 使用 `.head()`、`.tail()`、`.sample()` 方法，观察、了解数据表的情况。
# 3. 使用 `.dtypes` 属性，来查看各列数据的数据类型。
# 4. 使用 `isnull()` 配合 `.any()` 等方法，来查看各列是否存在空值。
# 5. 使用 `.describe()` 方法，看看数据表中数值型的数据是怎么分布的。
# 
# 

# In[2]:


# access the first and second row of movie_data
print("Show the first and second row of movie_data:")
print(movie_data.loc[[0,1]])
print('-' * 100)
print()

# return the titile of the first and second movie in movie_data
print("Return the title of the first and second movie in movie_data:")
print(movie_data[['original_title']].iloc[[0,1]])
print('-' * 100)
print()

# return the top 5 rows, tail 5 rows and random row of movie_data
print("Return the top 5 rows:")
print(movie_data.head())
print("Return the tail 5 rows:")
print(movie_data.tail())
print("Return a random row:")
print(movie_data.sample())
print('-' * 100)
print()

# Return the data type of some certain columns
print("Return the data type of some certain columns:")
print('data type of id column is {}'.format(movie_data['id'].dtypes))
print('data type of homepage column is {}'.format(movie_data['homepage'].dtypes))
print('data type of genres column is {}'.format(movie_data['genres'].dtypes))
print('data type of release_date column is {}'.format(movie_data['release_date'].dtypes))
print('data type of vote_average column is {}'.format(movie_data['vote_average'].dtypes))
print('data type of release_year column is {}'.format(movie_data['release_year'].dtypes))
print('data type of budget_adj column is {}'.format(movie_data['budget_adj'].dtypes))
print('data type of popularity column is {}'.format(movie_data['popularity'].dtypes))
print('-' * 100)
print()

# check if there's NaN value in movie_data
print("Check if there's NaN value in each column of movie_data:")
print(movie_data.isnull().any(axis = 0))
print('-' * 100)
print()

# Return the statistic data of movie_data
print("Return the statistic data of movie_data:")
print(movie_data.describe())


# ---
# 
# **任务1.3: ** 清理数据
# 
# 在真实的工作场景中，数据处理往往是最为费时费力的环节。但是幸运的是，我们提供给大家的 tmdb 数据集非常的「干净」，不需要大家做特别多的数据清洗以及处理工作。在这一步中，你的核心的工作主要是对数据表中的空值进行处理。你可以使用 `.fillna()` 来填补空值，当然也可以使用 `.dropna()` 来丢弃数据表中包含空值的某些行或者列。
# 
# 任务：使用适当的方法来清理空值，并将得到的数据保存。

# In[3]:


# Use value 0 to replace the NaN value. There's no effect on the data because all the columns with NaN value are
# of non-numerical data type
movie_data.fillna(0, inplace = True)
print(movie_data.isnull().any())


# ---
# 
# ---
# 
# ## 第二节 根据指定要求读取数据
# 
# 
# 相比 Excel 等数据分析软件，Pandas 的一大特长在于，能够轻松地基于复杂的逻辑选择合适的数据。因此，如何根据指定的要求，从数据表当获取适当的数据，是使用 Pandas 中非常重要的技能，也是本节重点考察大家的内容。
# 
# 

# ---
# 
# **任务2.1: ** 简单读取
# 
# 1. 读取数据表中名为 `id`、`popularity`、`budget`、`runtime`、`vote_average` 列的数据。
# 2. 读取数据表中前1～20行以及48、49行的数据。
# 3. 读取数据表中第50～60行的 `popularity` 那一列的数据。
# 
# 要求：每一个语句只能用一行代码实现。

# In[4]:


# Read data of column 'id', 'popularity', 'budget', 'runtime' and 'vote_average'
print(movie_data[['id', 'popularity', 'budget', 'runtime', 'vote_average']]);
print('-' * 100)
print()

# Read data of row 1~20, 48, 49
print(movie_data.iloc[np.append(np.arange(0,20),[47, 48])])
print('-' * 100)
print()

# Read the data from row 50th to 60th from column 'popularity'
print(movie_data['popularity'].iloc[np.arange(49,60)])


# ---
# 
# **任务2.2: **逻辑读取（Logical Indexing）
# 
# 1. 读取数据表中 **`popularity` 大于5** 的所有数据。
# 2. 读取数据表中 **`popularity` 大于5** 的所有数据且**发行年份在1996年之后**的所有数据。
# 
# 提示：Pandas 中的逻辑运算符如 `&`、`|`，分别代表`且`以及`或`。
# 
# 要求：请使用 Logical Indexing实现。

# In[5]:


# Read all data with popularity of greater than 5
print(movie_data[movie_data['popularity'] > 5])
print('-' * 100)
print()

# Read all data with popularity of greater than 5 and movie release date after 1996
print(movie_data[(movie_data['popularity'] > 5) & (movie_data['release_year'] > 1996)])


# ---
# 
# **任务2.3: **分组读取
# 
# 1. 对 `release_year` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `revenue` 的均值。
# 2. 对 `director` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `popularity` 的均值，从高到低排列。
# 
# 要求：使用 `Groupby` 命令实现。

# In[6]:


# Group by column 'release_year' and take average value of 'revenue'
print(movie_data.groupby(['release_year']).agg({'revenue':'mean'}))
print('-' * 70)
print()

# Group by column 'director' and take average value of 'popularity' with descending order
print(movie_data.groupby(['director']).agg({'popularity':'mean'})
            .sort_values('popularity', axis = 0, ascending = False))


# ---
# 
# ---
# 
# ## 第三节 绘图与可视化
# 
# 接着你要尝试对你的数据进行图像的绘制以及可视化。这一节最重要的是，你能够选择合适的图像，对特定的可视化目标进行可视化。所谓可视化的目标，是你希望从可视化的过程中，观察到怎样的信息以及变化。例如，观察票房随着时间的变化、哪个导演最受欢迎等。
# 
# <table>
# <thead><tr><th>可视化的目标</th><th>可以使用的图像</th></tr></thead><tbody>
#  <tr><td>表示某一属性数据的分布</td><td>饼图、直方图、散点图</td></tr>
#  <tr><td>表示某一属性数据随着某一个变量变化</td><td>条形图、折线图、热力图</td></tr>
#  <tr><td>比较多个属性的数据之间的关系</td><td>散点图、小提琴图、堆积条形图、堆积折线图</td></tr>
# </tbody></table>
# 
# 在这个部分，你需要根据题目中问题，选择适当的可视化图像进行绘制，并进行相应的分析。对于选做题，他们具有一定的难度，你可以尝试挑战一下～

# **任务3.1：**对 `popularity` 最高的20名电影绘制其 `popularity` 值。

# In[11]:


#movie_data[['popularity', 'original_title']].sort_values('popularity', ascending = False).iloc[np.arange(0,20)].plot(kind='barh', y='original_title', x = 'popularity', legend = False)

movie_data.sort_values('popularity', ascending = False).iloc[np.arange(0,20)].plot(kind = 'barh', x = 'original_title', y = 'popularity', legend = False);
plt.xlabel('popularity');


# ---
# **任务3.2：**分析电影净利润（票房-成本）随着年份变化的情况，并简单进行分析。

# In[12]:


movie_data['profit'] = movie_data['revenue_adj'] - movie_data['budget_adj']
base_color = sb.color_palette()[0]
plt.figure(figsize = [20,5])
sb.pointplot(data = movie_data, x = 'release_year', y = 'profit', color = base_color);
plt.xticks(rotation = 35);
plt.xlabel('movie release year');
plt.ylabel('movie net profit');


# 分析：
# 1. 从1960-1980年，电影的利润平均水平相比于最近35年的平均水平较高，但是起伏很大；在1980年-2015年的35年间，每一年的电影利润平均水平趋于稳定；
# 2. 虽然1960-1980年的平均利润水平相对较高，但是同一年内的电影利润差别却非常大，一些电影利润很可观的同时，也有一些电影利润很惨淡；

# ---
# 
# **[选做]任务3.3：**选择最多产的10位导演（电影数量最多的），绘制他们排行前3的三部电影的票房情况，并简要进行分析。

# In[13]:


# 
#movie_data.groupby('director').count()
director_top10 = movie_data['director'].value_counts().drop(0).head(10)
director_top10_all = movie_data[movie_data['director'].isin(director_top10.index)]
director_top3_movie = director_top10_all.sort_values(['director', 'revenue'], ascending = False).groupby('director').head(3)
director_top3_movie_cut = director_top3_movie[['director','original_title', 'revenue']]
#director_top3_movie_cut
sb.barplot(data = director_top3_movie_cut, y = 'director',x = 'revenue', color = base_color, );
plt.xlabel('mean revenue of top 3 movies');
plt.ylabel('director');


# 分析：
# 1.从最多产的前10位导演排行前3的电影平均票房来看，基本处在2~5亿的水平；
# 2.平均票房最高的导演是几乎是第二名票房的2倍之多

# ---
# 
# **[选做]任务3.4：**分析1968年~2015年六月电影的数量的变化。

# In[14]:


movie_year_range = movie_data[(movie_data['release_year'] <= 2015) & (movie_data['release_year'] >= 1968)]
movie_month = pd.to_datetime(movie_year_range['release_date']).dt.month
movie_atJune_Series = movie_year_range[movie_month == 6]['release_year'].value_counts().sort_index()
movie_atJune_df = pd.DataFrame({'Count':pd.Series(data = movie_atJune_Series.values, index = movie_atJune_Series.index)}).reset_index()
plt.figure(figsize = [20,5])
sb.pointplot(x = 'index', y = 'Count', data = movie_atJune_df);
plt.xticks(rotation = 30);
plt.xlabel('movie release year');
plt.ylabel('number of movies');


# 分析：
# 1. 整体来看，从1968-2015年，电影数量呈稳步上升趋势，除了90年代增长相对平缓；
# 2. 2015年与2005年相比，电影数量将近翻了一倍；

# ---
# 
# **[选做]任务3.5：**分析1968年~2015年六月电影 `Comedy` 和 `Drama` 两类电影的数量的变化。

# In[16]:


movie_atJune = movie_year_range[movie_month == 6]
#movie_atJune_Comedy = movie_atJune.str.contains('Comedy', regex = True)
#option 2movie_atJune[movie_atJune['genres'].str.contains('Comedy')]
movie_atJune_Comedy_Count = movie_atJune[movie_atJune['genres'].isin(['Comedy'])]
movie_atJune_comedy_series = movie_atJune_Comedy_Count['release_year'].value_counts().sort_index()
movie_comedy_df = pd.DataFrame({'Count':pd.Series(data = movie_atJune_comedy_series.values, index = movie_atJune_comedy_series.index)}).reset_index()
plt.figure(figsize = [20,5])
sb.pointplot(x='index', y = 'Count', data = movie_comedy_df)
plt.xticks(rotation = 30);
plt.xlabel('movie release year');
plt.ylabel('number of Comedy movies');


# In[17]:


movie_atJune_Drama_Count = movie_atJune[movie_atJune['genres'].isin(['Drama'])]
movie_atJune_drama_series = movie_atJune_Drama_Count['release_year'].value_counts().sort_index()
movie_drama_df = pd.DataFrame({'Count':pd.Series(data = movie_atJune_drama_series.values, index = movie_atJune_drama_series.index)}).reset_index()
plt.figure(figsize = [20,5])
sb.pointplot(x='index', y = 'Count', data = movie_drama_df)
plt.xticks(rotation = 30);
plt.xlabel('movie release year');
plt.ylabel('number of drama movies');


# > 注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出**File -> Download as -> HTML (.html)、Python (.py)** 把导出的 HTML、python文件 和这个 iPython notebook 一起提交给审阅者。
