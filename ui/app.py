"""Benchmark console - start with `make ui`."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from experiments.common.env import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

st.set_page_config(page_title="LLM Benchmark Console", page_icon=":material/calculate:", layout="wide")

pages = [
    st.Page("views/runs.py", title="Runs", icon=":material/monitoring:", default=True),
    st.Page("views/estimate.py", title="Estimate cost", icon=":material/calculate:"),
    st.Page("views/cache.py", title="Cache", icon=":material/database:"),
]
st.navigation(pages).run()
