import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.feature import Feature
from autoop.core.ml.model.classification import (
    KNearestNeighbors,
    LogisticRegression as ClassifierLogisticRegression,
    DecisionTree,
)
from autoop.core.ml.model.regression import (
    MultipleLinearRegression,
    Lasso,
    SupportVectorRegression,
)
from autoop.core.ml.metric import (
    MeanSquaredError,
    MeanAbsoluteError,
    R2,
    Accuracy,
    CohensKappa,
    CSI,
)
from autoop.core.ml.pipeline import Pipeline

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine "
    "learning pipeline to train a model on a dataset."
)

automl = AutoMLSystem.get_instance()

artifacts = automl.registry.list(type="dataset")

datasets = [
    Dataset(
        name=artifact.name,
        asset_path=artifact.asset_path,
        data=artifact.data,
        version=artifact.version,
        metadata=artifact.metadata,
        tags=artifact.tags,
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

    features = detect_feature_types(selected_dataset)
    feature_list = [{'name': feature.name,
                     'type': feature.type} for feature in features]

    st.subheader("Detected Features")
    feature_df = pd.DataFrame(feature_list)
    st.dataframe(feature_df)

    all_feature_names = [feature['name'] for feature in feature_list]

    target_feature_name = st.selectbox(
        "Select Target Feature", all_feature_names)

    available_input_features = [
        name for name in all_feature_names if name != target_feature_name
    ]

    input_feature_names = st.multiselect(
        "Select Input Features",
        options=available_input_features,
    )

    if input_feature_names:

        st.write(
            f"**Selected Input Features:** {', '.join(input_feature_names)}")

        st.write(f"**Selected Target Feature:** {target_feature_name}")

        target_feature_type = next(
            (feature['type'] for feature in feature_list if feature[
                'name'] == target_feature_name),
            None,
        )

        if target_feature_type == 'categorical':
            task_type = 'Classification'
        elif target_feature_type == 'numerical':
            task_type = 'Regression'
        else:
            task_type = 'Unknown'

        st.write(f"**Detected Task Type:** {task_type}")

        if task_type == 'Classification':
            model_options = ['K-Nearest Neighbors',
                             'Logistic Regression', 'Decision Tree']
            model_mapping = {
                'K-Nearest Neighbors': KNearestNeighbors,
                'Logistic Regression': ClassifierLogisticRegression,
                'Decision Tree': DecisionTree,
            }
        elif task_type == 'Regression':
            model_options = [
                'Multiple Linear Regression',
                'Lasso Regression',
                'Support Vector Regression',
            ]
            model_mapping = {
                'Multiple Linear Regression': MultipleLinearRegression,
                'Lasso Regression': Lasso,
                'Support Vector Regression': SupportVectorRegression,
            }
        else:
            st.error("Unknown task type. Cannot proceed.")
            st.stop()

        selected_model_name = st.selectbox("Select a Model", model_options)

        ModelClass = model_mapping[selected_model_name]
        model = ModelClass()

        split_ratio = st.slider(
            'Select the training data ratio',
            min_value=0.1,
            max_value=0.9,
            value=0.8,
            step=0.05,
        )

        if task_type == 'Regression':
            metric_options = {
                'Mean Squared Error': MeanSquaredError,
                'Mean Absolute Error': MeanAbsoluteError,
                'R2 Score': R2,
            }
        else:
            metric_options = {
                'Accuracy': Accuracy,
                'Cohen\'s Kappa': CohensKappa,
                'CSI': CSI,
            }

        metric_names = list(metric_options.keys())

        selected_metric_names = st.multiselect(
            'Select Metrics',
            metric_names,
            default=metric_names,
        )

        metrics = [metric_options[name]() for name in selected_metric_names]

        st.subheader("Pipeline Summary:")

        st.markdown(f"""
        #### Pipeline Configuration

        - **Dataset:** {selected_dataset.name}
        - **Input Features:** {', '.join(input_feature_names)}
        - **Target Feature:** {target_feature_name}
        - **Task Type:** {task_type}
        - **Selected Model:** {selected_model_name}
        - **Training Data Ratio:** {split_ratio}
        - **Selected Metrics:** {', '.join(selected_metric_names)}
        """)

        if st.button("Train Model"):
            input_features = [
                Feature(
                    name=name,
                    type=next(
                        (f.type for f in features if f.name == name),
                        '',
                    ),
                )
                for name in input_feature_names
            ]
            target_feature = Feature(name=target_feature_name,
                                     type=target_feature_type)

            pipeline = Pipeline(
                metrics=metrics,
                dataset=selected_dataset,
                model=model,
                input_features=input_features,
                target_feature=target_feature,
                split=split_ratio,
            )

            with st.spinner('Training the model...'):
                results = pipeline.execute()

            st.session_state['trained_pipeline'] = pipeline
            st.session_state['pipeline_results'] = results
            st.session_state['input_feature_names'] = input_feature_names
            st.session_state['target_feature_name'] = target_feature_name
            st.session_state['selected_model_name'] = selected_model_name
            st.session_state['selected_dataset_name'] = selected_dataset_name
            st.session_state['split_ratio'] = split_ratio
            st.session_state['selected_metric_names'] = selected_metric_names

            st.success("Model training completed!")

            st.write("### Evaluation Results")
            st.write("#### Training Metrics")
            for metric, value in results['train_metrics']:
                st.write(f"- **{metric.__class__.__name__}:** {value}")

            st.write("#### Testing Metrics")
            for metric, value in results['test_metrics']:
                st.write(f"- **{metric.__class__.__name__}:** {value}")

        if 'trained_pipeline' in st.session_state:
            st.subheader("Save Pipeline")
            pipeline_name = st.text_input("Pipeline Name")
            pipeline_version = st.text_input("Pipeline Version", value="1.0.0")
            pipeline_tags = st.text_input("Tags (comma-separated)", value="")

            if st.button("Save Pipeline"):
                if pipeline_name:
                    pipeline = st.session_state['trained_pipeline']
                    artifacts = pipeline.artifacts

                    for artifact in artifacts:
                        artifact.name = f"{pipeline_name}_{artifact.name}"
                        artifact.version = pipeline_version
                        tags = pipeline_tags.split(",")
                        artifact.tags = [
                            tag.strip() for tag in tags if tag.strip()]
                        artifact.tags.append(f"pipeline:{pipeline_name}")
                        artifact.asset_path = (
                            f"pipelines/{pipeline_name}/{artifact.name}"
                            f"_{pipeline_version}.pkl"
                        )
                        automl.registry.register(artifact)

                    st.success(
                        f"Pipeline '{pipeline_name}' saved successfully.")

                else:
                    st.error("Please provide a pipeline name.")

    else:
        st.warning("Please select at least one input feature")

else:
    st.warning(
        "No datasets available. Please upload a dataset in the Datasets page.")
