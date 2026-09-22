from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ml.inference.predictor import FraudPredictor

st.set_page_config(page_title="Fraud Detection MVP", layout="wide")
st.title("Fraud Detection MVP")
st.caption(
    "MVP para demonstracao e estudo experimental do impacto do threshold em deteccao de fraude."
)


@st.cache_resource
def load_predictor() -> FraudPredictor:
    return FraudPredictor.from_path()


def render_prediction_tab(predictor: FraudPredictor) -> None:
    artifact = predictor.artifact
    st.subheader("Prediction")
    st.write(
        "Insira uma transacao usando os nomes reais das colunas do dataset armazenados no artefato."
    )
    default_payload = {feature: 0 for feature in artifact.feature_columns}
    payload_text = st.text_area(
        "Transaction payload (JSON)",
        value=json.dumps(default_payload, indent=2),
        height=260,
    )

    if st.button("Run prediction", type="primary"):
        try:
            payload = json.loads(payload_text)
            result = predictor.predict_transaction(payload)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")
            col2.metric("Threshold", f"{result['threshold']:.2f}")
            col3.metric("Risk", result["risk"])
            col4.metric("Decision", result["decision"])
            st.json(result)
        except Exception as exc:
            st.error(str(exc))


def render_threshold_tab(predictor: FraudPredictor) -> None:
    artifact = predictor.artifact
    st.subheader("Threshold Analysis")
    table = pd.DataFrame(artifact.threshold_table)

    st.dataframe(table, use_container_width=True)
    if not table.empty:
        chart_columns = st.columns(3)
        chart_columns[0].line_chart(table.set_index("threshold")[["precision", "recall"]])
        chart_columns[1].line_chart(table.set_index("threshold")[["f1", "f2"]])
        chart_columns[2].line_chart(
            table.set_index("threshold")[["false_positive_rate", "false_negative_rate"]]
        )

        chosen_threshold = st.select_slider(
            "Threshold selecionado para inspecao",
            options=[float(value) for value in table["threshold"].tolist()],
            value=float(artifact.threshold),
        )
        selected_row = table.loc[table["threshold"] == chosen_threshold].iloc[0]
        st.write(selected_row.to_dict())


def render_model_tab(predictor: FraudPredictor) -> None:
    artifact = predictor.artifact
    st.subheader("Model Metadata")
    st.json(
        {
            "model_name": artifact.model_name,
            "model_version": artifact.model_version,
            "trained_at": artifact.trained_at,
            "dataset_name": artifact.dataset_name,
            "target_column": artifact.target_column,
            "feature_count": len(artifact.feature_columns),
            "threshold": artifact.threshold,
            "best_thresholds": artifact.best_thresholds,
            "metrics": artifact.metrics,
            "notes": artifact.notes,
        }
    )


try:
    predictor = load_predictor()
    tab1, tab2, tab3 = st.tabs(["Prediction", "Threshold Analysis", "Model Info"])
    with tab1:
        render_prediction_tab(predictor)
    with tab2:
        render_threshold_tab(predictor)
    with tab3:
        render_model_tab(predictor)
except Exception as exc:
    st.error(
        "Nao foi possivel carregar o artefato do modelo. Execute o script de treinamento primeiro."
    )
    st.exception(exc)
