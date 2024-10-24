import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types

st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

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


if datasets:
    dataset_names = [ds.name for ds in datasets]
    dataset_dict = {ds.name: ds for ds in datasets}

    selected_dataset_name = st.selectbox("Select a Dataset", dataset_names)

    selected_dataset = dataset_dict[selected_dataset_name]

    df = selected_dataset.read()

    st.write(f"**Dataset Name:** {selected_dataset.name}")
    st.write(f"**Version:** {selected_dataset.version}")
    st.write(f"**Tags:** {', '.join(selected_dataset.tags)}")


    if st.checkbox("Preview Dataset"):
        st.write(df.head())

    feature_types = detect_feature_types(selected_dataset)
    features = []
    for feature in feature_types:
        features.append({'name': feature.name, 'type': feature.type})


    st.subheader("Detected Features")
    feature_df = pd.DataFrame(features)
    st.dataframe(feature_df)

    all_feature_names = [feature['name'] for feature in features]

    target_feature_name = st.selectbox("Select Target Feature", all_feature_names)

    available_input_features = [name for name in all_feature_names if name != target_feature_name]

    input_feature_names = st.multiselect(
        "Select Input Features",
        options=available_input_features,
    )

    if input_feature_names:

        st.write(f"**Selected Input Features:** {', '.join(input_feature_names)}")

        st.write(f"**Selected Target Feature:** {target_feature_name}")


        target_feature_type = next(
            (feature['type'] for feature in features if feature['name'] == target_feature_name),
            None
        )

        if target_feature_type == 'categorical':
            task_type = 'Classification'
        elif target_feature_type == 'numerical':
            task_type = 'Regression'
        else:
            task_type = 'Unknown'

        st.write(f"**Detected Task Type:** {task_type}")

    else:
        st.warning("Please select at least one input feature.")

else:
    st.warning("No datasets available. Please upload a dataset in the Datasets page.")



