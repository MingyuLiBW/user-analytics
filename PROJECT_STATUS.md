# User Analytics 项目状态

> 本文档用于后续开发交接和维护。环境、Git 与功能状态根据 2026-11-10 对项目实际检查所得；测试结果特别注明是否为本次运行。

## 1. 项目概述

- **项目名称：** User Analytics
- **项目目标：** 基于 Python 和 Streamlit 构建交互式数据分析 Web 应用。
- **主要用户流程：** 上传 CSV → 数据质量检查 → EDA → Customer Segmentation。
- **应用入口：** 根目录 `app.py`。
- **主要依赖：** Streamlit、Pandas、NumPy、scikit-learn、Plotly、SciPy；测试使用 pytest。
- **统计模块依赖情况：** `cluster_statistics.py` 直接导入 `scipy.stats`；SciPy 已在 `pyproject.toml` 的运行时依赖中直接声明。

## 2. 开发环境

| 项目 | 实际检查结果 |
| --- | --- |
| 操作系统 | Windows（PowerShell 环境） |
| 项目路径 | `C:\AI_Projects\User_Analytics` |
| 全局 Python | Python 3.10.6；通过 `C:\Users\kevin\AppData\Local\Programs\Python\Python310\python.exe --version` 验证 |
| 项目虚拟环境 | `.venv` |
| 虚拟环境解释器 | `.venv\Scripts\python.exe`，版本 Python 3.10.6 |
| 项目布局 | `src` layout |
| 项目安装方式 | `pyproject.toml` 配置 setuptools；虚拟环境安装元数据确认是 editable 安装，项目 URL 为本地项目目录 |
| 依赖安装 | 推荐 `python -m pip install -e ".[test]"`；兼容入口为 `python -m pip install -r requirements.txt` |
| Web 启动命令 | `python -m streamlit run app.py`（先激活 `.venv`） |
| 默认本地地址 | `http://localhost:8501`；这是 Streamlit 默认地址，本次未检查正在运行的服务。端口被占用时 Streamlit 可能选用其他端口 |
| 测试命令 | `python -m pytest -q` |

已检查虚拟环境中的版本：Streamlit 1.65.0、Pandas 2.3.3、NumPy 2.2.6、scikit-learn 1.7.2、Plotly 7.1.0、pytest 9.2.0。以上是当前环境快照，不是依赖锁定版本。

**待确认：** 本机 `py -3.10 --version` 返回 `No installed Python found!`，但全局 Python 3.10.6 的解释器绝对路径可以直接运行。项目日常操作建议继续使用 `.venv\Scripts\python.exe`，之后如需依赖 Python Launcher，再检查其注册配置。

## 3. 项目目录结构

以下为主要项目文件，省略 `.venv`、缓存、egg-info 和其他生成文件：

```text
User_Analytics/
├── app.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── .gitkeep
├── src/
│   └── user_analytics/
│       ├── __init__.py
│       ├── config.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── cluster_statistics.py
│       │   ├── clustering.py
│       │   ├── eda.py
│       │   ├── outliers.py
│       │   └── profiling.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── validation.py
│       └── ui/
│           ├── upload_state.py
│           └── pages/
│               ├── __init__.py
│               ├── customer_segmentation.py
│               ├── data_quality.py
│               └── eda.py
└── tests/
    ├── test_cluster_profiles.py
    ├── test_cluster_statistics.py
    ├── test_clustering.py
    ├── test_customer_segmentation_ui.py
    ├── test_data_quality.py
    ├── test_eda.py
    ├── test_eda_bivariate.py
    ├── test_kmeans.py
    ├── test_loader.py
    ├── test_home_upload_state.py
    └── test_pca.py
```

### 主要文件职责

- `app.py`：设置 Streamlit 页面配置、侧栏导航、Home 页面和 CSV 上传，并把已上传 DataFrame 传给各页面。
- `.streamlit/config.toml`：项目级 Streamlit 配置；当前将单文件上传上限设为 50 MB。
- `src/user_analytics/`：应用 Python 包，使用 `src` layout；由 setuptools 在 `pyproject.toml` 中配置。
- `analysis/`：不依赖 Streamlit 的分析逻辑；包括 EDA、数据质量、聚类、PCA、画像和聚类统计检验。
- `data/`：CSV 读取与基础数据验证。
- `ui/pages/`：Home 之外的 Streamlit 页面呈现和交互。
- `src/user_analytics/ui/upload_state.py`：Home 上传数据的 session state 更新、显式清除和新文件替换逻辑。
- `data/`：本地数据目录；Git 中用 `.gitkeep` 保留目录，`.gitignore` 忽略本地 CSV 和表格数据文件。
- `tests/`：pytest 单元测试，按数据加载、分析模块和页面辅助逻辑划分。
- `pyproject.toml`：项目元数据、Python 版本范围、依赖声明和 setuptools 的 `src` package discovery 配置。
- `requirements.txt`：兼容安装入口，引用 `pyproject.toml` 项目及 `test` extra；依赖只在 `pyproject.toml` 维护。

## 4. 已完成功能

以下按当前代码实现记录；测试文件说明对应的主要单元测试位置。

### CSV Upload

- 首页使用 Streamlit file uploader 接收 CSV，Pandas 读取为 DataFrame。
- 使用 `utf-8-sig` 处理 UTF-8 BOM；首行是否为列名由用户选项控制。无表头时使用 `header=None` 并生成 `Column_1` 等列名。
- 在 Pandas 解析前使用标准库 `csv.reader` 校验逻辑记录字段数：有表头时以表头字段数为准，无表头时以首条数据记录为准；不一致时拒绝读取，并用 `index_col=False` 防止额外字段被误当作索引。
- Streamlit 的 `server.maxUploadSize` 和 loader 后端均限制单个文件最大为 50 MB（52,428,800 bytes）。后端优先检查 Streamlit 上传对象的 `size` 元数据，在读取内容前拒绝超限文件，并对实际字节长度再次校验。
- 空文件、不可解析格式、仅有表头而无数据行等情况由 `CSVLoadError` 和校验结果反馈。
- 上传文件发生变化时先使旧数据及 Customer Segmentation 缓存失效；用户清除文件时清空数据。页面切换导致 uploader 暂时不在脚本中的 rerun 不会清空有效 DataFrame。
- 上传后展示前 10 行、行列数、字段名及 Pandas dtype。
- **浏览器手工验证（用户本轮报告）：** 正常 CSV 上传、字段数量异常拒绝、页面切换后保留数据、用户主动清除、新文件替换旧文件、无效文件替换后旧数据失效，均已验证。上传 55 MB 文件时，Streamlit 控件拒绝上传并提示 `File must be 50.0MB or smaller.`。这些是用户报告的浏览器结果，并非本轮自动化浏览器测试。
- 主要代码：`app.py`、`src/user_analytics/ui/upload_state.py`、`src/user_analytics/data/loader.py`、`src/user_analytics/data/validation.py`、`.streamlit/config.toml`。
- 主要测试：`tests/test_loader.py`、`tests/test_home_upload_state.py`。

#### Cross-page Integration Regression

- **User-reported browser verification（用户手动浏览器回归，本轮报告）：** 以数据集 A 完成 K-means、PCA、Cluster Profile 和 Statistical Analysis 后，连续执行 A → B → C 替换，所有检查均通过。
- Home 预览、字段和行列数，以及 Data Quality、EDA 的分析数据均与当前上传的数据集一致。
- 替换数据集后，旧 K-means 聚类结果、PCA 坐标、Cluster Profile 和 Statistical Analysis 结果不再显示。
- A、B、C 存在部分同名字段时，特征选择和统计字段选择控件状态正常，未发现不适用的旧字段残留。
- 在 C 已加载时上传 `invalid_mismatched_columns.csv`，文件被拒绝，旧数据和旧分析结果均未继续显示。
- 上述内容是用户报告的手动浏览器验证，不是 Codex 自动执行的浏览器测试，也不计入 pytest 通过数量。

### Data Quality

- Missing Values：统计每列数量/比例和全数据集缺失情况，并绘图。
- Duplicate Rows：按完整重复行统计，不自动删除。
- Data Type Classification：展示 Pandas dtype，并按数值、类别、日期时间或其他类型分类。
- IQR Outlier Detection：用 1.5 × IQR 标记潜在异常，不修改或删除记录。
- 主要代码：`src/user_analytics/analysis/profiling.py`、`src/user_analytics/analysis/outliers.py`、`src/user_analytics/ui/pages/data_quality.py`。
- 主要测试：`tests/test_data_quality.py`。

### EDA

- Descriptive Statistics：数值变量和类别变量分别汇总。
- Numeric Univariate Analysis：数值直方图、箱线图。
- Categorical Univariate Analysis：频数柱状图、占比和 Top N / Other。
- Numeric-Numeric Analysis：散点图、可选线性趋势线和类别着色。
- Categorical-Numeric Analysis：箱线图或小提琴图。
- Categorical-Categorical Analysis：频数或比例分组图。
- Correlation Matrix：Pearson / Spearman 数值相关矩阵及热力图；布尔字段排除，缺失值使用成对有效观测策略。
- 主要代码：`src/user_analytics/analysis/eda.py`、`src/user_analytics/ui/pages/eda.py`。
- 主要测试：`tests/test_eda.py`、`tests/test_eda_bivariate.py`。

### Customer Segmentation

- Numeric/Categorical Preprocessing：特征选择、日期和布尔字段排除、缺失值填补、类别 One-Hot Encoding、数值 StandardScaler。
- Missing Value Imputation：数值字段使用中位数；类别字段使用众数，全缺失类别列使用占位值 `**MISSING**`。
- StandardScaler 只用于数值特征；OneHotEncoder 未再次标准化，并配置未知类别安全处理。
- K-means Clustering：固定 `n_init` 和用户提供的 `random_state`，返回模型结果、标签、中心和 inertia。
- Elbow Method、Silhouette Score：在用户选择的 K 范围评估并展示；不声称单一绝对最优 K。
- PCA Visualization：对已预处理特征矩阵投影到二维，仅用于可视化，不改变 K-means 标签。
- Cluster Profile：从原始字段汇总各簇数值统计和类别组成。
- ANOVA、Levene Test、Kruskal-Wallis Test、Chi-square Test、Eta-squared、Epsilon-squared、Cramér's V 和 Benjamini-Hochberg FDR Correction：后端提供探索性聚类间比较和效应量；UI 分表展示主检验结果与诊断/效应量。
- 主要代码：`src/user_analytics/analysis/clustering.py`、`src/user_analytics/analysis/cluster_statistics.py`、`src/user_analytics/ui/pages/customer_segmentation.py`。
- 主要测试：`tests/test_clustering.py`、`tests/test_kmeans.py`、`tests/test_pca.py`、`tests/test_cluster_profiles.py`、`tests/test_cluster_statistics.py`、`tests/test_customer_segmentation_ui.py`。

## 5. 重要技术设计

- **页面与分析分层：** 页面交互位于 `ui/pages/`，分析函数位于 `analysis/`，CSV 读入和基础校验位于 `data/`。
- **上传数据状态：** `app.py` 将当前 DataFrame 放入 `st.session_state.uploaded_dataframe`，并传递给各个页面。`csv_upload` 的 `on_change` 回调代表用户主动更改/清除上传；非 Home 页面卸载 uploader 时不调用清除逻辑。新文件解析前会清掉旧 DataFrame 与 Customer Segmentation 相关缓存，解析失败后不保留旧数据。
- **聚类结果缓存：** Customer Segmentation 将聚类输出保存在 session state。结果记录数据指纹、选定特征、K 和 random state；页面检查这些值与当前输入一致后才展示。数据更换时清除聚类和评估状态；设置变化时提示重新运行聚类，避免展示过期结果。
- **评估与可视化缓存：** K 值评估和 PCA 结果分别保存在 session state，并通过数据/字段/参数签名校验。普通 rerun 不会重新拟合 K-means。
- **统计结果缓存：** Statistical Analysis 结果绑定数据指纹、聚类字段、K、random state 和统计字段；签名不匹配时不展示旧结果。统计检验调用已有聚类标签，不重新训练模型。
- **索引对齐：** `fit_kmeans()` 的聚类标签为以原始样本 index 为索引的 Series。画像、聚类汇总、PCA 和统计分析对 Series labels 按 index 对齐；统计后端对 Series 要求标签 index 与 DataFrame index 唯一且完全匹配。array-like labels 则按行位置对应。
- **预处理：** `prepare_clustering_data()` 产出预处理后的特征和元数据；数值缺失以中位数填补并标准化，类别缺失以众数或占位类别填补并 One-Hot 编码。PCA 接收预处理后的特征，不重新从原始数据构造输入。
- **统计检验限制：** ANOVA、Levene、Kruskal-Wallis、Chi-square 等是当前数据上的探索性比较。若用于构造 K-means 的同一批特征又参与检验，P-value 不是独立确认性推断。Levene 是方差齐性诊断，不替换普通 ANOVA；卡方结果需结合期望频数警告。效应量与 P-value 提供不同信息，不应自动转成业务结论。
- **源数据保护：** 当前分析逻辑以副本/派生结果进行计算；异常值和缺失值分析不自动删除源数据记录。

## 6. Git 状态和提交历史

以下信息在记录 `e51c18f` 提交后核对：

- **当前分支：** `feature/customer-segmentation`
- **工作区状态：** 本轮更新本文档前，`git status` 显示工作区干净。本文档修改后会成为唯一的未提交修改；本轮没有修改代码或 Git 历史。
- **本地分支：** `feature/customer-segmentation`、`feature/data-quality`、`feature/eda`、`master`。
- **远端：** 本次 `git remote -v` 未输出 remote；是否计划配置 GitHub remote 待后续确认。

最近 5 个提交（完整 hash）：

| Hash | Message |
| --- | --- |
| `e51c18feebb925fc3aeaff668e1e9062f5d72a4c` | Improve CSV upload reliability and validation |
| `585ed958217b3761ac07e68b3b95114d76a5920a` | Add project development status documentation |
| `92ae702c7c215da63edb0e8fe9d758cd5e3a4718` | Add cluster statistical analysis |
| `23d67cff41e9670c409b176fc8bf068d54ab8c47` | Add customer segment profile insights |
| `48bf879325b84ae5a589904396bd7a16c4619e31` | Add PCA visualization for customer segmentation |

## 7. 测试情况

- **测试框架：** pytest。
- **CSV loader 定向测试：** 14 passed（2026-11-10，本轮重新运行）。
- **项目 `.venv` 全量测试记录：** 106 passed（2026-11-10，先前运行 `.venv\Scripts\python.exe -m pytest -q`）。
- **上传状态与 Customer Segmentation UI 定向测试：** 17 passed（2026-11-10，先前只读审查期间运行 `tests/test_home_upload_state.py` 与 `tests/test_customer_segmentation_ui.py` 得到；不是本轮跨页面浏览器测试结果）。
- **最近一次 `git diff --check`：** 通过（2026-11-10，本轮运行；Git 仅提示部分文件工作副本 LF 将规范化为 CRLF，无 whitespace 错误）。
- **依赖管理临时环境验证（Codex 自动执行）：** 在项目目录外的 `C:\AI_Projects\_ua_dependency_check_20261110` 使用 Python 3.10.6 创建全新虚拟环境；`python -m pip install -e ".[test]"` 和 `python -m pip install -r requirements.txt` 均成功；`python -m pip check` 输出 `No broken requirements found.`；`user_analytics`、`scipy`、`streamlit`、`pandas`、`numpy`、`sklearn`、`plotly` 导入检查成功；`python -m pytest -q` 为 106 passed；`python -m streamlit run app.py` 成功启动于端口 8502，验证后已停止。项目原有 `.venv` 未被修改。
- **自动化与手动验证区分：** CSV loader 14 passed、上传状态与 Customer Segmentation UI 17 passed 为此前 pytest 自动化记录；全量 106 passed 既有项目 `.venv` 记录，也在本轮临时环境中重新验证。Cross-page Integration Regression 为用户报告的手动浏览器验证，不是自动化测试结果。临时环境安装与启动验证也不代表跨页面浏览器测试。

## 8. 当前开发进度

核心功能开发、CSV 上传可靠性改进和跨页面手动浏览器回归均已完成。CSV Upload 阶段记录在提交 `e51c18f`（Improve CSV upload reliability and validation）中，包含字段数量一致性校验、session state 清除/替换处理和 50 MB 上传限制。A → B → C 数据替换及 Data Quality、EDA、Customer Segmentation 跨页面一致性由用户报告验证通过。本轮完成依赖声明和安装文档修正，并在项目目录外的全新 Python 3.10 环境验证两种安装入口、依赖一致性、模块导入、全量 pytest 和 Streamlit 启动；项目原有 `.venv` 未修改。该依赖验证不属于新的跨页面浏览器回归。

## 9. 下一步开发计划

按优先级推进工程化收尾：

1. 继续完善 README 和项目文档，使其与当前功能、配置及验证状态一致。
2. 检查部署配置；SciPy 直接依赖及 pytest 测试 extra 已在 `pyproject.toml` 声明并通过全新环境安装验证。
3. 补充必要的异常数据和边界条件测试，并修复检查中发现的问题。
4. 准备部署所需配置；配置 GitHub remote 并上传 GitHub（待用户确认后执行）。
5. 部署 Streamlit Community Cloud。
6. 浏览器验证线上应用。

## 10. 开发约定

- 修改代码前先阅读现有实现。
- 优先模块化设计，避免把所有逻辑堆在 `app.py`。
- 数据分析逻辑与 UI 分离。
- 新功能必须配套测试。
- 修改后运行 `pytest -q` 和 `git diff --check`。
- 不擅自修改统计方法和既有业务逻辑。
- 不在未经验证的情况下宣称功能正常。
- 浏览器测试通过后再 Commit。
- 不擅自切换分支、Commit 或 Push。

## 11. 下次如何继续项目

在 PowerShell 中启动项目：

```powershell
cd C:\AI_Projects\User_Analytics
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

打开 Streamlit 输出的地址；默认通常为 `http://localhost:8501`。

检查 Git 状态和提交历史：

```powershell
git status
git branch
git log --oneline -5
```

建议在未来 Codex 对话中使用：

> 请先完整阅读项目根目录的 PROJECT_STATUS.md，再检查当前 Git 状态和实际代码结构。确认文档与代码是否一致，指出当前开发进度和下一项任务。不要覆盖现有代码，不要擅自切换分支、Commit 或 Push。
