import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"
SCHEMA_PATH = BASE_DIR / "models" / "input_schema.json"
EVALUATION_PATH = BASE_DIR / "data" / "evaluation" / "model_comparison.csv"
INTERPRETATION_PATH = (
    BASE_DIR
    / "data"
    / "interpretation"
    / "random_forest_importances.csv"
)


st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_evaluation_results():
    if not EVALUATION_PATH.exists():
        return None

    return pd.read_csv(EVALUATION_PATH)


@st.cache_data
def load_feature_importance():
    if not INTERPRETATION_PATH.exists():
        return None

    return pd.read_csv(INTERPRETATION_PATH)


def create_numeric_input(feature):
    minimum = feature.get("min")
    maximum = feature.get("max")
    median = feature.get("median")

    if minimum is None:
        minimum = 0

    if maximum is None:
        maximum = max(minimum + 1, 100)

    if median is None:
        median = minimum

    integer_feature = feature["dtype"].startswith(("int", "uint"))

    if integer_feature:
        minimum = int(round(minimum))
        maximum = int(round(maximum))
        median = int(round(median))

        if maximum < minimum:
            maximum = minimum

        return st.number_input(
            feature["name"],
            min_value=minimum,
            max_value=maximum,
            value=min(max(median, minimum), maximum),
            step=1,
        )

    return st.number_input(
        feature["name"],
        min_value=float(minimum),
        max_value=float(maximum),
        value=min(max(float(median), float(minimum)), float(maximum)),
        step=0.01,
    )


def create_categorical_input(feature):
    values = feature.get("values", [])

    if not values:
        return ""

    default = feature.get("default", values[0])

    try:
        default_index = values.index(default)
    except ValueError:
        default_index = 0

    return st.selectbox(
        feature["name"],
        options=values,
        index=default_index,
    )


def build_input_form(schema):
    inputs = {}
    features = schema["features"]
    columns = st.columns(2)

    for index, feature in enumerate(features):
        with columns[index % 2]:
            if feature["type"] == "numeric":
                inputs[feature["name"]] = create_numeric_input(feature)
            else:
                inputs[feature["name"]] = create_categorical_input(feature)

    return inputs


def display_prediction(prediction):
    st.success(f"Estimated price: ${prediction:,.0f}")


def display_prediction_details(inputs, prediction):
    st.subheader("Prediction details")

    information = pd.DataFrame({
        "Metric": [
            "Estimated price",
            "Number of input features",
        ],
        "Value": [
            f"${prediction:,.0f}",
            len(inputs),
        ],
    })

    st.dataframe(
        information,
        use_container_width=True,
        hide_index=True,
    )


def display_model_performance():
    results = load_evaluation_results()

    if results is None:
        st.info("Model evaluation results are not available yet.")
        return

    st.subheader("Model performance")
    st.dataframe(
        results.round(2),
        use_container_width=True,
        hide_index=True,
    )

    metric = st.selectbox(
        "Metric to visualize",
        ["MAE", "RMSE", "R2"],
    )

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(results["Model"], results[metric])
    axis.set_ylabel(metric)
    axis.set_xlabel("Model")
    axis.set_title(f"Model comparison - {metric}")
    plt.xticks(rotation=20)
    plt.tight_layout()

    st.pyplot(figure)
    plt.close(figure)


def display_feature_importance():
    importance = load_feature_importance()

    if importance is None:
        st.info("Feature importance results are not available yet.")
        return

    if not {"feature", "importance"}.issubset(importance.columns):
        return

    st.subheader("Most important features")
    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).head(15)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        importance["feature"][::-1],
        importance["importance"][::-1],
    )
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.set_title("Top 15 features influencing predictions")
    plt.tight_layout()

    st.pyplot(figure)
    plt.close(figure)


def main():
    st.title("House Price Predictor")
    st.write("Enter the characteristics of a house to estimate its sale price.")

    if not MODEL_PATH.exists():
        st.error(
            "Saved model not found. Run: "
            "python src/models/save_model.py"
        )
        st.stop()

    if not SCHEMA_PATH.exists():
        st.error(
            "Input schema not found. Run: "
            "python src/models/save_model.py"
        )
        st.stop()

    model = load_model()
    schema = load_schema()

    with st.form("house_prediction_form"):
        inputs = build_input_form(schema)
        submitted = st.form_submit_button("Estimate price")

    if submitted:
        input_data = pd.DataFrame([inputs])

        try:
            prediction = float(model.predict(input_data)[0])
            display_prediction(prediction)
            display_prediction_details(inputs, prediction)
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    st.divider()

    tab1, tab2 = st.tabs([
        "Model performance",
        "Feature importance",
    ])

    with tab1:
        display_model_performance()

    with tab2:
        display_feature_importance()


if __name__ == "__main__":
    main()