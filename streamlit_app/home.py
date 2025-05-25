import streamlit as st
import pathlib

# MUST come before st.title or st.markdown
st.set_page_config(
    page_title="Wikipedia Graph Clustering",
    page_icon="📚",
    layout="centered"
)

readme_path = pathlib.Path("README.md")
st.title("Wikipedia Graph Classification Dashboard")

if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()
        st.markdown(readme_text, unsafe_allow_html=False)
else:
    st.warning("README.md not found.")