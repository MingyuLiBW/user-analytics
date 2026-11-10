# User Analytics Dashboard

An interactive Python and Streamlit platform for CSV exploration, data quality checks, descriptive statistics, K-means clustering, and visualizations.

## Environment setup

This project uses Python 3.10 and a project-local virtual environment. From the project root in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `.venv` already exists, reuse it instead of creating it again.

## Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run the app

```powershell
streamlit run app.py
```

Streamlit prints the local URL in the terminal, usually <http://localhost:8501>.
