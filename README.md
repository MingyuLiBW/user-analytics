# User Analytics Dashboard

An interactive Python and Streamlit platform for CSV exploration, data quality checks, descriptive statistics, K-means clustering, and visualizations.

## 全新环境安装（PowerShell）

项目要求 Python 3.10。以下命令从项目根目录运行；如果 `.venv` 已存在，请复用它并跳过创建步骤。先用 Python Launcher 创建环境：

```powershell
py -3.10 -m venv .venv
```

如果 `py -3.10` 启动器不可用，可使用本机 Python 3.10 解释器的绝对路径创建环境。例如：

```powershell
& "C:\Users\kevin\AppData\Local\Programs\Python\Python310\python.exe" -m venv .venv
```

环境创建完成后，激活环境并安装项目及测试依赖：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

启动应用并运行测试：

```powershell
python -m streamlit run app.py
python -m pytest -q
```

Streamlit 启动后会持续占用当前终端。运行 pytest 时，可以另开一个 PowerShell 窗口并激活同一虚拟环境；也可以先在当前窗口按 `Ctrl+C` 停止 Streamlit，再运行 pytest。

## Dependency configuration

`pyproject.toml` 是运行依赖及测试依赖的唯一维护来源；`test` extra 提供 pytest。上面的 `python -m pip install -e ".[test]"` 是推荐安装命令。`requirements.txt` 仅保留兼容安装入口，内容引用本地项目及其 `test` extra。需要使用该入口时，在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
```

以上两种安装方式都会安装项目包、运行依赖和 pytest。启动应用后，Streamlit 通常会在终端显示 <http://localhost:8501>。
