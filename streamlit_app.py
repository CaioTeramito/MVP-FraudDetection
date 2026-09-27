from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

import pandas as pd
import streamlit as st

from ml.inference.predictor import FraudPredictor

st.set_page_config(page_title="Fraud Detection MVP", page_icon="F", layout="wide")
st.markdown(
    """
    <style>
        .stApp { background: #f5f2ec; color: #1c2a2b; }
        [data-testid="stSidebar"] { background: #123436; }
        [data-testid="stSidebar"] * { color: #eef4ee; }
        .hero { padding: 1.2rem 0 1.8rem; }
        .eyebrow { color: #a04c2d; font-size: .8rem; font-weight: 700; letter-spacing: .12em; }
        .hero h1 { color: #123436; margin: .2rem 0; }
        .hero p { color: #52615d; font-size: 1.05rem; max-width: 48rem; }
        [data-testid="stWidgetLabel"] { color: #1c2a2b !important; }
        [data-testid="stMetric"] { background: #fffdf8; border: 1px solid #dfd8ca; border-radius: 10px; padding: .8rem; }
        .decision-card { border-radius: 12px; padding: 1rem 1.2rem; font-weight: 700; }
        .decision-approve { background: #e4f1e8; color: #1b6538; }
        .decision-review { background: #fff1d5; color: #855b11; }
        .decision-block { background: #f8e2df; color: #992d28; }
    </style>
    """,
    unsafe_allow_html=True,
)

FIELD_LABELS = {
    "amount": "Valor da transacao",
    "amt": "Valor da transacao",
    "transaction_amount": "Valor da transacao",
    "step": "Etapa da transacao",
    "oldbalanceorg": "Saldo anterior da origem",
    "newbalanceorig": "Saldo atual da origem",
    "oldbalancedest": "Saldo anterior do destino",
    "newbalancedest": "Saldo atual do destino",
    "isflaggedfraud": "Indicador de fraude sinalizado",
    "transaction_date": "Data da transacao",
    "transaction_datetime": "Data e hora da transacao",
    "trans_date_trans_time": "Data e hora da transacao",
    "transaction_type": "Tipo da transacao", "type": "Tipo da transacao",
    "merchant": "Estabelecimento", "category": "Categoria", "gender": "Genero",
    "city": "Cidade", "state": "Estado", "country": "Pais", "card_type": "Tipo de cartao",
}


@st.cache_resource
def load_predictor() -> FraudPredictor:
    return FraudPredictor.from_path()


def humanize_feature_name(feature: str) -> str:
    if label := FIELD_LABELS.get(feature.lower()):
        return label
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", feature)
    return re.sub(r"[_\s]+", " ", spaced).strip().capitalize()


def is_temporal_feature(feature: str) -> bool:
    return any(token in feature.lower() for token in ("date", "time", "datetime", "timestamp"))


def parse_numeric_input(feature: str, value: str) -> float:
    normalized = value.strip().replace(",", ".")
    label = humanize_feature_name(feature)
    if not normalized:
        raise ValueError(f"Informe o campo '{label}'.")
    try:
        number = float(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Digite um numero valido para '{label}', usando ponto ou virgula decimal."
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"O valor de '{label}' precisa ser um numero finito.")
    return number


def feature_schema(predictor: FraudPredictor) -> dict[str, dict[str, Any]]:
    """Recover field types and categorical options from the fitted preprocessor."""
    schema = {feature: {"kind": "numeric", "options": []} for feature in predictor.artifact.feature_columns}
    for name, transformer, columns in predictor.artifact.preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if name == "cat":
            categories = getattr(transformer.named_steps.get("encoder"), "categories_", [])
            for column, values in zip(columns, categories, strict=False):
                schema[column] = {"kind": "categorical", "options": [str(value) for value in values]}
        elif name == "num":
            for column in columns:
                schema[column] = {"kind": "numeric", "options": []}
    return schema


def render_input(feature: str, spec: dict[str, Any]) -> Any:
    label = humanize_feature_name(feature)
    help_text = "Campo usado pelo modelo na analise da transacao."
    if spec["kind"] == "categorical":
        return st.selectbox(label, spec["options"], help=help_text, key=f"field_{feature}")
    if is_temporal_feature(feature):
        selected_date = st.date_input(label, value=date.today(), help=help_text, key=f"field_{feature}")
        if "time" in feature.lower() or "datetime" in feature.lower():
            selected_time = st.time_input("Horario", help=help_text, key=f"time_{feature}")
            return datetime.combine(selected_date, selected_time).isoformat(sep=" ")
        return selected_date.isoformat()
    return st.text_input(
        label,
        value="",
        placeholder="Digite um valor",
        help=help_text,
        key=f"field_{feature}",
    )


def render_prediction_result(
    result: dict[str, Any],
    pass_through_pct: int,
    block_above_pct: int,
) -> None:
    probability = result["fraud_probability"]
    if probability <= pass_through_pct / 100:
        risk, action = "LOW", "APPROVE"
    elif probability <= block_above_pct / 100:
        risk, action = "MEDIUM", "REVIEW"
    else:
        risk, action = "HIGH", "BLOCK"

    decision = action.lower()
    st.subheader("Resultado da analise")
    col1, col2, col3 = st.columns(3)
    col1.metric("Probabilidade estimada de fraude", f"{probability:.2%}")
    col2.metric("Nivel de risco", risk)
    col3.metric("Decisao", action)
    st.markdown(
        f'<div class="decision-card decision-{decision}">Decisao recomendada: {action} | Risco: {risk}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Faixas aplicadas: ate {pass_through_pct}% passa direto; acima de "
        f"{pass_through_pct}% ate {block_above_pct}% requer avaliacao humana; "
        f"acima de {block_above_pct}% bloqueia."
    )
    with st.expander("Detalhes tecnicos da previsao"):
        st.write(
            f"Threshold binario interno do modelo: {result['threshold']:.2%}. "
            "A decisao operacional exibida usa os intervalos configurados nesta tela."
        )
    if result["top_contributors"]:
        st.subheader("Principais fatores da decisao")
        st.dataframe(pd.DataFrame(result["top_contributors"]), use_container_width=True)


def render_prediction_tab(predictor: FraudPredictor) -> None:
    artifact = predictor.artifact
    st.subheader("Analisar transacao")
    st.write("Preencha os dados abaixo. A aplicacao monta e valida o payload exigido pelo modelo automaticamente.")
    has_fraud_flag = any(
        feature.lower() == "isflaggedfraud" for feature in artifact.feature_columns
    )
    if has_fraud_flag:
        st.caption("O indicador externo de fraude sera considerado ausente (0) nesta simulacao.")
    if all(feature.startswith("feature_") for feature in artifact.feature_columns):
        st.warning(
            "Este artefato usa features anonimas de um dataset sintetico. Para ver campos como valor, data e tipo de transacao, retreine com um dataset que contenha essas colunas reais."
        )
    st.markdown("#### Intervalos de decisao")
    threshold_columns = st.columns(2)
    pass_through_pct = threshold_columns[0].slider(
        "Passa direto ate (%)",
        min_value=0,
        max_value=100,
        value=round(artifact.threshold_low * 100),
        help="Probabilidades ate este limite sao aprovadas automaticamente.",
        key="decision_pass_through_pct",
    )
    block_above_pct = threshold_columns[1].slider(
        "Bloqueia acima de (%)",
        min_value=0,
        max_value=100,
        value=round(artifact.threshold_high * 100),
        help="Probabilidades acima deste limite sao bloqueadas como fraude.",
        key="decision_block_above_pct",
    )
    if pass_through_pct >= block_above_pct:
        st.error("O limite para passar direto precisa ser menor que o limite para bloquear.")
    schema = feature_schema(predictor)
    with st.form("transaction_form", border=False):
        columns = st.columns(2)
        payload: dict[str, Any] = {}
        visible_index = 0
        for feature in artifact.feature_columns:
            if feature.lower() == "isflaggedfraud":
                payload[feature] = 0
                continue
            with columns[visible_index % 2]:
                payload[feature] = render_input(feature, schema[feature])
            visible_index += 1
        submitted = st.form_submit_button(
            "Analisar transacao",
            type="primary",
            use_container_width=True,
            disabled=pass_through_pct >= block_above_pct,
        )
    if submitted:
        try:
            prediction_payload = {
                feature: (
                    value
                    if feature.lower() == "isflaggedfraud"
                    else parse_numeric_input(feature, value)
                    if schema[feature]["kind"] == "numeric" and not is_temporal_feature(feature)
                    else value
                )
                for feature, value in payload.items()
            }
            st.session_state["prediction_result"] = predictor.predict_transaction(prediction_payload)
            st.session_state["prediction_payload"] = prediction_payload
        except Exception as exc:
            st.error(f"Nao foi possivel analisar a transacao: {exc}")
    if result := st.session_state.get("prediction_result"):
        render_prediction_result(result, pass_through_pct, block_above_pct)
        with st.expander("Payload tecnico enviado ao modelo"):
            st.json(st.session_state["prediction_payload"])


def render_threshold_tab(predictor: FraudPredictor) -> None:
    table = pd.DataFrame(predictor.artifact.threshold_table)
    st.subheader("Threshold Analysis")
    st.caption("Compare o custo de falsos positivos e fraudes nao detectadas antes de mudar a regra operacional.")
    if not table.empty:
        selected_threshold = st.select_slider("Threshold selecionado", [float(value) for value in table["threshold"]], float(predictor.artifact.threshold))
        row = table.loc[table["threshold"] == selected_threshold].iloc[0]
        metrics = st.columns(4)
        metrics[0].metric("Precision", f"{row['precision']:.2%}")
        metrics[1].metric("Recall", f"{row['recall']:.2%}")
        metrics[2].metric("F2", f"{row['f2']:.3f}")
        metrics[3].metric("Falsos positivos", int(row["false_positives"]))
        charts = st.columns(3)
        charts[0].line_chart(table.set_index("threshold")[["precision", "recall"]])
        charts[1].line_chart(table.set_index("threshold")[["f1", "f2"]])
        charts[2].line_chart(table.set_index("threshold")[["false_positive_rate", "false_negative_rate"]])
        with st.expander("Tabela completa de thresholds"):
            st.dataframe(table, use_container_width=True)


def render_model_tab(predictor: FraudPredictor) -> None:
    artifact = predictor.artifact
    st.subheader("Modelo em uso")
    col1, col2, col3 = st.columns(3)
    col1.metric("Modelo", artifact.model_name.replace("_", " ").title())
    col2.metric("Versao", artifact.model_version)
    col3.metric("Features", len(artifact.feature_columns))
    st.write(f"Dataset: `{artifact.dataset_name}` | Treinado em: `{artifact.trained_at}`")
    with st.expander("Detalhes tecnicos"):
        st.json({"target_column": artifact.target_column, "feature_columns": artifact.feature_columns, "threshold": artifact.threshold, "metrics": artifact.metrics, "best_thresholds": artifact.best_thresholds, "notes": artifact.notes})


try:
    predictor = load_predictor()
    st.markdown("""<div class="hero"><div class="eyebrow">DECISAO ASSISTIDA POR MODELO</div><h1>Fraud Detection MVP</h1><p>Analise uma transacao, interprete o risco e acompanhe como o threshold altera o equilibrio entre deteccao e falsos positivos.</p></div>""", unsafe_allow_html=True)
    with st.sidebar:
        st.header("Sessao")
        page = st.radio("Navegacao", ["Analisar transacao", "Threshold Analysis", "Modelo"], label_visibility="collapsed")
        st.divider()
        st.caption(f"Modelo: {predictor.artifact.model_name}")
        st.caption(f"Threshold binario do modelo: {predictor.artifact.threshold:.2f}")
    if page == "Analisar transacao":
        render_prediction_tab(predictor)
    elif page == "Threshold Analysis":
        render_threshold_tab(predictor)
    else:
        render_model_tab(predictor)
except Exception as exc:
    st.error("Nao foi possivel carregar o artefato do modelo. Execute o treinamento primeiro.")
    st.exception(exc)
