# 3_🚀_Deployment.py

import streamlit as st
import pickle
from app.core.system import AutoMLSystem

st.set_page_config(page_title="Deployment", page_icon="🚀")

st.title("🚀 Deployment")
st.write("Here you can view and manage your saved pipelines")

automl = AutoMLSystem.get_instance()

"""
This app allows users to view saved pipelines, delete them, and use them for
making predictions on new data.
"""

all_artifacts = automl.registry.list()

"""
Retrieves pipeline names from the list of all artifacts.
"""

pipeline_names = set()
for artifact in all_artifacts:
    for tag in artifact.tags:
        if tag.startswith('pipeline:'):
            pipeline_name = tag.split('pipeline:')[1]
            pipeline_names.add(pipeline_name)

pipeline_names = list(pipeline_names)

if pipeline_names:
    selected_pipeline_name = st.selectbox("Select a Pipeline", pipeline_names)

    pipeline_artifacts = [
        artifact for artifact in all_artifacts
        if f'pipeline:{selected_pipeline_name}' in artifact.tags
    ]

    """
    Displays information about the selected pipeline.
    """

    st.write(f"## Pipeline: {selected_pipeline_name}")

    model_artifact = next(
        (artif for artif in pipeline_artifacts if artif.type == 'model'),
        None
    )

    if model_artifact:
        model_type = model_artifact.metadata.get("model_type", "unknown")
        model_version = model_artifact.version
        model_tags = ", ".join(model_artifact.tags) or "No tags"

        st.write(f"**Model Type:** {model_type.capitalize()}")
        st.write(f"**Version:** {model_version}")
        st.write(f"**Tags:** {model_tags}")
    else:
        st.error("Model artifact not found for this pipeline.")
        st.stop()

    if st.session_state.get('delete_confirmation') == selected_pipeline_name:
        st.warning(f"Are you sure you want to delete the pipeline "
                   f"'{selected_pipeline_name}'?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, delete it"):
                for artifact in pipeline_artifacts:
                    automl.registry.delete(artifact.id)
                st.success(
                    f"Pipeline '{selected_pipeline_name}' deleted.")
                pipeline_names.remove(selected_pipeline_name)
                del st.session_state['delete_confirmation']
                st.rerun()
        with col2:
            if st.button("Cancel"):
                del st.session_state['delete_confirmation']
    else:
        """
        Allows the user to use the selected pipeline for making predictions.
        """
        st.subheader("Use Pipeline for Prediction")

        if model_type == 'classification':
            st.write("This is a **classification** pipeline. "
                     "Please upload data appropriate for classification tasks")
        elif model_type == 'regression':
            st.write("This is a **regression** pipeline. "
                     "Please upload data appropriate for regression tasks.")
        else:
            st.error("Unknown model type.")

        uploaded_file = st.file_uploader(
            "Upload a CSV file for prediction", type=["csv"])

        if uploaded_file is not None:
            import pandas as pd
            import numpy as np

            new_data_df = pd.read_csv(uploaded_file)

            st.write("### Uploaded Data")
            st.write(new_data_df.head())

            pipeline_config_artifact = next(
                (artifact for artifact in pipeline_artifacts
                 if artifact.name.endswith("pipeline_config")),
                None
            )

            if pipeline_config_artifact:
                pipeline_config = pickle.loads(pipeline_config_artifact.data)
                input_features = pipeline_config['input_features']
                target_feature = pipeline_config['target_feature']

                preprocessing_artifacts = {
                    artifact.name: artifact
                    for artifact in pipeline_artifacts
                    if artifact.type == 'preprocessing'
                }

                """
                Preprocesses input features using the saved preprocessing
                artifacts.
                """

                preprocessed_inputs = []
                for feature in input_features:
                    feature_name = feature if isinstance(
                        feature, str) else feature.name
                    try:
                        feature_data = new_data_df[
                            feature_name].values.reshape(-1, 1)
                    except KeyError:
                        st.error(f"Change the upload type to the correct one,"
                                 f" which should be a {model_type} dataset.")
                        st.stop()
                    artifact_name = f"{selected_pipeline_name}_{feature_name}"
                    if artifact_name in preprocessing_artifacts:
                        preprocessor_artifact = preprocessing_artifacts[
                            artifact_name]
                        preprocessor = pickle.loads(preprocessor_artifact.data)
                        feature_data = preprocessor.transform(feature_data)
                    preprocessed_inputs.append(feature_data)

                X_new = np.concatenate(preprocessed_inputs, axis=1)

                model = pickle.loads(model_artifact.data)

                predictions = model.predict(X_new)
                prediction_df = pd.DataFrame(predictions, columns=[
                    "Prediction"])

                st.write("### Predictions")
                if model_type == 'classification':
                    st.write("**Classification Results:**")
                    st.write(prediction_df)
                elif model_type == 'regression':
                    st.write("**Regression Results:**")
                    st.write(prediction_df)
                else:
                    st.error("Unknown model type.")
            else:
                st.error("Pipeline configuration not found.")
else:
    st.warning("No pipelines found.")
