import random
import nltk
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from src.pqrs_service import clasificar_pqrs
from src.events import (
    load_events,
    add_event,
    delete_event,
    create_event,
    is_active,
    time_remaining,
)

nltk.download("stopwords", quiet=True)

st.set_page_config(
    page_title="Clasificador de PQRS",
    page_icon="📋",
    layout="wide",
)

CATEGORY_COLORS = {
    "Petición":   "#3B82F6",
    "Queja":      "#F59E0B",
    "Reclamo":    "#EF4444",
    "Sugerencia": "#10B981",
}
URGENCY_COLORS = {
    "Alta":  "#EF4444",
    "Media": "#F59E0B",
    "Baja":  "#10B981",
}
RISK_COLORS = {
    "Rojo":     "#EF4444",
    "Amarillo": "#F59E0B",
    "Verde":    "#10B981",
}
ID2LABEL = {0: "Petición", 1: "Queja", 2: "Reclamo", 3: "Sugerencia"}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0f1117; }
    .header-box {
        background: linear-gradient(135deg, #1e2433 0%, #0f1117 100%);
        border: 1px solid #2d3748;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .header-box h1 {
        font-family: 'IBM Plex Mono', monospace;
        color: #e2e8f0;
        font-size: 1.6rem;
        margin: 0 0 0.3rem 0;
    }
    .header-box p { color: #94a3b8; margin: 0; font-size: 0.9rem; font-weight: 300; }
    .metric-card {
        background: #1e2433;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.75rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0;
    }
    .result-row {
        background: #1e2433;
        border: 1px solid #2d3748;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


BATCH_TEMPLATES = {
    "Petición": [
        "Solicito respetuosamente una certificación de los pagos realizados durante el último año fiscal.",
        "Requiero información detallada sobre los requisitos para acceder al programa de subsidio de vivienda.",
        "Por favor necesito una copia de mi historial de atenciones médicas de los últimos 6 meses.",
        "Me dirijo a ustedes para pedir la habilitación del acceso a los servicios en línea de la entidad.",
        "Solicito copia del reglamento interno vigente y el organigrama actualizado de la institución.",
        "Requiero se me informe el estado actual de mi trámite con número de radicado 2024-08821.",
        "Pido amablemente que se actualicen mis datos de contacto en el sistema.",
        "Quisiera solicitar una audiencia con el director para tratar asuntos relacionados con mi contrato.",
    ],
    "Queja": [
        "El personal de ventanilla fue completamente grosero y se negó a atender mi solicitud sin justificación.",
        "La página web lleva tres días sin funcionar y nadie en soporte técnico responde mis correos.",
        "Es inaceptable que los tiempos de espera superen las cuatro horas para una simple consulta.",
        "La información que me dieron en la primera visita fue completamente diferente a la de la segunda.",
        "Los baños y salas de espera de las instalaciones están en condiciones deplorables.",
        "Nadie me ha dado razón del estado de mi proceso pese a consultarlo múltiples veces.",
        "El funcionario encargado ha sido completamente indiferente y poco profesional.",
        "Me informaron un horario de atención que al llegar no correspondía con la realidad.",
    ],
    "Reclamo": [
        "Exijo el reembolso inmediato del pago duplicado realizado el pasado mes. Adjunto comprobantes.",
        "Interpongo reclamo formal ya que han pasado 45 días hábiles sin respuesta a mi solicitud.",
        "Reclamo el reconocimiento de la garantía de mi producto, según la ley del consumidor.",
        "Han cobrado un valor diferente al cotizado y exijo se haga la devolución del excedente.",
        "Llevo 6 meses esperando la corrección de un error en mi facturación sin medidas concretas.",
        "Reclamo la prestación correcta del servicio acordado en el contrato firmado el pasado año.",
        "Exijo respuesta formal antes de acudir a los organismos de control competentes.",
        "El daño causado a mis bienes durante el procedimiento no ha sido reconocido ni reparado.",
    ],
    "Sugerencia": [
        "Sería muy útil implementar un sistema de citas en línea para evitar las largas filas.",
        "Propongo habilitar un canal de WhatsApp para consultas rápidas y reducir la carga en ventanilla.",
        "Sugiero que los formularios estén disponibles en formato digital para facilitar los trámites.",
        "Recomiendo ampliar el horario de atención los sábados para usuarios que trabajan entre semana.",
        "Propongo instalar señalización clara dentro de las instalaciones para orientar a los visitantes.",
        "Sería beneficioso ofrecer capacitaciones al personal sobre atención al cliente y empatía.",
        "Sugiero crear un portal de seguimiento donde los usuarios puedan rastrear su trámite en línea.",
        "Propongo habilitar una línea gratuita de atención para adultos mayores y personas con discapacidad.",
    ],
}


def generate_batch(n: int) -> list:
    random.seed()
    categorias = list(BATCH_TEMPLATES.keys())
    return [{"texto": random.choice(BATCH_TEMPLATES[c := random.choice(categorias)]), "cat_real": c} for _ in range(n)]


@st.cache_resource
def load_model():
    path = Path(__file__).parent / "models" / "best_pipeline.pkl"
    return joblib.load(path) if path.exists() else None

def predict_one(text: str, model, incluir_respuesta: bool = False) -> dict:
    resultado = clasificar_pqrs(
        texto=text,
        model=model,
        incluir_respuesta=incluir_respuesta,
    )

    probas = resultado.get("probabilidades")
    if probas:
        resultado["probas"] = [
            probas.get("Petición", 0) / 100,
            probas.get("Queja", 0) / 100,
            probas.get("Reclamo", 0) / 100,
            probas.get("Sugerencia", 0) / 100,
        ]
    else:
        resultado["probas"] = None

    return resultado


st.markdown("""
<div class="header-box">
    <h1>Clasificador Automático de PQRS</h1>
    <p>Sistema AutoML entrenado con Optuna · Clasifica Peticiones, Quejas, Reclamos y Sugerencias en español sin intervención humana</p>
</div>
""", unsafe_allow_html=True)

model = load_model()
if model is None:
    st.error("Modelo no encontrado. Sube `models/best_pipeline.pkl` al repositorio.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Clasificar texto", "Simulación en lote", "Cómo funciona", "Eventos institucionales"])

with tab1:
    st.markdown("##### Ingresa cualquier texto y el modelo lo clasificará automáticamente")
    st.markdown("")

    texto_input = st.text_area(
        label="Texto",
        height=140,
        placeholder="Escribe aquí la petición, queja, reclamo o sugerencia...\n\nEj: Llevo 3 meses esperando respuesta a mi solicitud y nadie me da información.",
        label_visibility="collapsed",
    )

    generar_respuesta = st.checkbox(
        "Generar respuesta institucional sugerida",
        value=True,
    )

    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        clasificar = st.button("Clasificar →", use_container_width=True, type="primary")

    if clasificar and texto_input.strip():
        with st.spinner("Procesando..."):
            res = predict_one(
                texto_input,
                model,
                incluir_respuesta=generar_respuesta,
            )

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        cat_c  = CATEGORY_COLORS.get(res["categoria"], "#6B7280")
        urg_c  = URGENCY_COLORS.get(res["urgencia"], "#6B7280")
        con_c  = "#10B981" if res["confianza"] and res["confianza"] > 75 else "#F59E0B"
        con_str = f"{res['confianza']:.1f}%" if res["confianza"] else "N/A"
        risk_c   = RISK_COLORS.get(res["nivel_riesgo"], "#6B7280")
        risk_val = (
            f"{res['nivel_riesgo']}"
            f"<br><span style='font-size:0.75rem;color:#64748b'>Score: {res['score_riesgo']}/100</span>"
        )

        for col, label, value, color in [
            (c1, "Categoría", res["categoria"], cat_c),
            (c2, "Urgencia",  res["urgencia"],  urg_c),
            (c3, "Confianza", con_str,           con_c),
            (c4, "Semáforo de riesgo", risk_val, risk_c),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-top:3px solid {color}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color}">{value}</div>
                </div>""", unsafe_allow_html=True)

        if res["probas"] is not None:
            st.markdown("")
            labels = [ID2LABEL[i] for i in range(len(res["probas"]))]
            fig = go.Figure(go.Bar(
                x=labels, y=res["probas"] * 100,
                marker_color=[CATEGORY_COLORS.get(label, "#6B7280") for label in labels],
                text=[f"{p*100:.1f}%" for p in res["probas"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Distribución de probabilidad por categoría",
                yaxis_title="Confianza (%)", yaxis_range=[0, 115], height=280,
                paper_bgcolor="#0f1117", plot_bgcolor="#1e2433",
                font_color="#94a3b8", margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    elif clasificar:
        st.warning("Escribe algún texto antes de clasificar.")


with tab2:
    st.markdown("##### Genera un lote de PQRS y obsérvalas clasificadas automáticamente")
    st.markdown("Simula cómo el sistema procesaría cientos de mensajes reales sin intervención humana.")
    st.markdown("")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        n_textos = st.slider("Cantidad de PQRS a generar", min_value=5, max_value=32, value=12)
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        generar = st.button("Generar y clasificar", use_container_width=True, type="primary")

    if generar:
        with st.spinner(f"Clasificando {n_textos} PQRS..."):
            batch   = generate_batch(n_textos)
            results = []
            for item in batch:
                res = predict_one(item["texto"], model)
                results.append({
                    "Texto":     item["texto"],
                    "Categoría": res["categoria"],
                    "Urgencia":  res["urgencia"],
                    "Confianza": f"{res['confianza']:.1f}%" if res["confianza"] else "N/A",
                    "Riesgo":    res["nivel_riesgo"],
                    "Score":     res["score_riesgo"],
                })

        st.session_state["pqrs_batch_df"] = pd.DataFrame(results)
        st.session_state["pqrs_batch_n"]  = n_textos

    if "pqrs_batch_df" in st.session_state:
        df = st.session_state["pqrs_batch_df"]
        n_textos_batch = st.session_state["pqrs_batch_n"]
        conteo = df["Categoría"].value_counts()

        st.markdown("---")
        st.markdown("**Resumen del lote**")
        m_cols = st.columns(4)
        for i, cat in enumerate(["Petición", "Queja", "Reclamo", "Sugerencia"]):
            color = CATEGORY_COLORS[cat]
            count = conteo.get(cat, 0)
            with m_cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="border-top:3px solid {color}">
                    <div class="metric-label">{cat}</div>
                    <div class="metric-value" style="color:{color}">{count}</div>
                    <div style="color:#64748b;font-size:0.8rem">{count/n_textos_batch*100:.0f}% del lote</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_pie = go.Figure(go.Pie(
                labels=conteo.index.tolist(), values=conteo.values.tolist(),
                marker_colors=[CATEGORY_COLORS.get(c, "#6B7280") for c in conteo.index],
                hole=0.45, textfont_size=13,
            ))
            fig_pie.update_layout(
                title="Distribución por categoría", height=280,
                paper_bgcolor="#0f1117", font_color="#94a3b8",
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_g2:
            urg_conteo = df["Urgencia"].value_counts()
            urg_order  = ["Alta", "Media", "Baja"]
            urg_vals   = [urg_conteo.get(u, 0) for u in urg_order]
            fig_urg = go.Figure(go.Bar(
                x=urg_order, y=urg_vals,
                marker_color=[URGENCY_COLORS[u] for u in urg_order],
                text=urg_vals, textposition="outside",
            ))
            fig_urg.update_layout(
                title="Distribución por urgencia",
                yaxis_range=[0, max(urg_vals) * 1.35 + 1], height=280,
                paper_bgcolor="#0f1117", plot_bgcolor="#1e2433",
                font_color="#94a3b8", margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_urg, use_container_width=True)

        st.markdown("---")
        st.markdown("**Filtrar y ordenar resultados**")
        st.markdown(
            "<span style='color:#64748b;font-size:0.85rem'>"
            "Elige qué categorías ver y cómo ordenar el detalle del lote: "
            "por urgencia o por semáforo de riesgo, de mayor a menor o de menor a mayor."
            "</span>",
            unsafe_allow_html=True,
        )

        col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
        with col_f1:
            categorias_mostrar = st.multiselect(
                "Categorías a mostrar",
                options=["Petición", "Queja", "Reclamo", "Sugerencia"],
                default=["Petición", "Queja", "Reclamo", "Sugerencia"],
            )
        with col_f2:
            ordenar_por = st.selectbox(
                "Ordenar por",
                ["Orden de generación", "Urgencia", "Semáforo de riesgo"],
            )
        with col_f3:
            orden_direccion = st.radio(
                "Orden",
                ["Mayor a menor", "Menor a mayor"],
                horizontal=True,
            )

        df_filtrado = df[df["Categoría"].isin(categorias_mostrar)].copy()

        if ordenar_por == "Urgencia":
            urgencia_rank = {"Alta": 3, "Media": 2, "Baja": 1}
            df_filtrado["_orden"] = df_filtrado["Urgencia"].map(urgencia_rank)
            df_filtrado = df_filtrado.sort_values(
                "_orden", ascending=(orden_direccion == "Menor a mayor"), kind="stable"
            )
        elif ordenar_por == "Semáforo de riesgo":
            df_filtrado = df_filtrado.sort_values(
                "Score", ascending=(orden_direccion == "Menor a mayor"), kind="stable"
            )

        st.markdown(
            f"**Detalle del lote clasificado** "
            f"<span style='color:#64748b;font-size:0.85rem;font-family:\"IBM Plex Mono\",monospace'>"
            f"({len(df_filtrado)} de {len(df)})</span>",
            unsafe_allow_html=True,
        )

        if df_filtrado.empty:
            st.info("No hay resultados que coincidan con las categorías seleccionadas.")
        else:
            for _, row in df_filtrado.iterrows():
                cat_c  = CATEGORY_COLORS.get(row["Categoría"], "#6B7280")
                urg_c  = URGENCY_COLORS.get(row["Urgencia"], "#6B7280")
                risk_c = RISK_COLORS.get(row["Riesgo"], "#6B7280")
                st.markdown(f"""
                <div class="result-row">
                    <span style="color:#94a3b8">{row['Texto']}</span><br>
                    <span style="margin-top:6px;display:inline-block">
                        <span class="badge" style="background:{cat_c}22;color:{cat_c};border:1px solid {cat_c}55">{row['Categoría']}</span>
                        &nbsp;
                        <span class="badge" style="background:{urg_c}22;color:{urg_c};border:1px solid {urg_c}55">Urgencia {row['Urgencia']}</span>
                        &nbsp;
                        <span class="badge" style="background:{risk_c}22;color:{risk_c};border:1px solid {risk_c}55">Riesgo {row['Riesgo']}</span>
                        &nbsp;
                        <span style="color:#475569;font-size:0.75rem;font-family:'IBM Plex Mono',monospace">{row['Confianza']}</span>
                    </span>
                </div>""", unsafe_allow_html=True)


with tab3:
    st.markdown("#### ¿Cómo funciona el AutoML?")
    st.markdown("""
    En lugar de elegir un modelo manualmente, este sistema usa **Optuna** para evaluar
    automáticamente decenas de combinaciones y seleccionar la que mejor funciona con los datos.
    El proceso completo ocurre sin intervención humana.
    """)

    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("**Modelos evaluados automáticamente**")
        for m in ["Logistic Regression", "Support Vector Machine (SVM)", "Random Forest", "Gradient Boosting", "Naive Bayes"]:
            st.markdown(f"- {m}")
    with col_y:
        st.markdown("**Vectorizadores evaluados**")
        for v in ["TF-IDF (unigramas y bigramas)", "CountVectorizer"]:
            st.markdown(f"- {v}")

    st.markdown("---")
    st.markdown("**Pipeline de preprocesamiento**")
    for i, paso in enumerate(["Conversión a minúsculas", "Eliminación de URLs y números", "Eliminación de puntuación", "Eliminación de stopwords en español"], 1):
        st.markdown(f"`{i}` {paso}")

    st.markdown("---")
    st.markdown("**Métrica de optimización:** F1-macro — penaliza el mal desempeño en clases minoritarias.")

with tab4:
    st.markdown("#### Eventos institucionales")
    st.markdown(
        "Registra incidentes o eventos vigentes (caídas del sistema, "
        "mantenimientos, fallas masivas, etc.). Mientras estén activos, "
        "el generador de respuestas los tendrá en cuenta para responder "
        "con empatía y disculpas institucionales cuando una PQRS esté relacionada."
    )
    st.markdown("")

    with st.form("nuevo_evento_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            titulo = st.text_input("Título del evento", placeholder="Ej: Caída del portal de pagos")
            area = st.text_input("Área responsable", value="General")
        with col2:
            vigencia_valor = st.number_input("Vigencia", min_value=1, value=24, step=1)
            vigencia_tipo = st.selectbox("Unidad de vigencia", ["horas", "días"])

        descripcion = st.text_area(
            "Descripción",
            placeholder="Ej: El portal de pagos en línea presentó intermitencias entre las 8am y las 11am del día de hoy.",
            height=100,
        )

        crear = st.form_submit_button("Registrar evento", type="primary", use_container_width=True)

        if crear:
            if not titulo.strip() or not descripcion.strip():
                st.warning("Completa el título y la descripción del evento.")
            else:
                # vigencia_tipo internamente se maneja como "horas" o "dias"
                tipo_interno = "horas" if vigencia_tipo == "horas" else "dias"
                evento = create_event(
                    titulo=titulo,
                    descripcion=descripcion,
                    vigencia_valor=int(vigencia_valor),
                    vigencia_tipo=tipo_interno,
                    area=area or "General",
                )
                add_event(evento)
                st.success("Evento registrado correctamente.")
                st.rerun()

    st.markdown("---")
    st.markdown("**Eventos registrados**")

    eventos = load_events()
    eventos_ordenados = sorted(eventos, key=lambda e: e["creado_en"], reverse=True)

    if not eventos_ordenados:
        st.info("No hay eventos registrados.")
    else:
        for evento in eventos_ordenados:
            activo = is_active(evento)
            estado_color = "#10B981" if activo else "#6B7280"
            estado_texto = time_remaining(evento) if activo else "Expirado"

            col_info, col_btn = st.columns([5, 1])
            with col_info:
                st.markdown(f"""
                <div class="result-row">
                    <span class="badge" style="background:{estado_color}22;color:{estado_color};border:1px solid {estado_color}55">{estado_texto}</span>
                    &nbsp;
                    <span class="badge" style="background:#3B82F622;color:#3B82F6;border:1px solid #3B82F655">{evento['area']}</span>
                    <br>
                    <strong style="color:#e2e8f0">{evento['titulo']}</strong><br>
                    <span style="color:#94a3b8">{evento['descripcion']}</span>
                </div>""", unsafe_allow_html=True)
            with col_btn:
                if st.button("Eliminar", key=f"del_{evento['id']}", use_container_width=True):
                    delete_event(evento["id"])
                    st.rerun()

st.markdown("---")
st.caption("Clasificador AutoML de PQRS · Construido con Optuna, scikit-learn y Streamlit · [GitHub](https://github.com/SanRom3/Clasificaci-n-de-PQRS)")