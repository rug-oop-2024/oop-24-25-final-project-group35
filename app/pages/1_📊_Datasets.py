import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

artifacts = automl.registry.list(type="dataset")

datasets = [
    Dataset(
        name=artifact.name,
        asset_path=artifact.asset_path,
        data=artifact.data,
        version=artifact.version,
        metadata=artifact.metadata,
        tags=artifact.tags
    )
    for artifact in artifacts
]

st.title("Dataset Management")

st.header("Upload a New Dataset")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of the uploaded dataset:")
    st.write(data.head())

    name = st.text_input("Dataset Name", value=uploaded_file.name)
    asset_path = st.text_input("Asset Path", value=f"objects/{uploaded_file.name}")
    version = st.text_input("Version", value="1.0.0")
    tags = st.text_input("Tags (comma-separated)", value="")

    if st.button("Save Dataset"):
        dataset = Dataset.from_dataframe(
            name=name,
            asset_path=asset_path,
            data=data,
            version=version
        )
        dataset.tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        automl.registry.register(dataset)
        st.success(f"Dataset {name} saved successfully.")
        st.experimental_rerun()

st.header("Existing Datasets")

if datasets:
    for ds in datasets:
        st.subheader(f"Dataset: {ds.name}")
        st.write(f"ID: {ds.id}")
        st.write(f"Asset Path: {ds.asset_path}")
        st.write(f"Version: {ds.version}")
        st.write(f"Type: {ds.type}")
        st.write(f"Tags: {', '.join(ds.tags)}")

        if st.checkbox(f"Preview {ds.name}", key=f"preview_{ds.id}"):
            df = ds.read()
            st.write(df)

        if st.button(f"Delete {ds.name}", key=f"delete_{ds.id}"):
            automl.registry.delete(ds.id)
            st.success(f"Dataset {ds.name} deleted.")
            st.experimental_rerun()
else:
    st.write("No datasets available.")
