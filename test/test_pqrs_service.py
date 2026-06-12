import os

import pytest

from src.pqrs_service import (
    infer_urgencia,
    get_probabilidades,
    clasificar_pqrs,
    ID2LABEL,
)


# ---------- infer_urgencia ----------

@pytest.mark.parametrize("texto,categoria,esperado", [
    ("Exijo el reembolso inmediato de mi dinero.", "Reclamo", "Alta"),
    ("Llevo varios meses esperando una solución.", "Reclamo", "Alta"),
    ("Quisiera saber el estado de mi trámite.", "Reclamo", "Media"),
    ("El servicio fue muy malo.", "Queja", "Media"),
    ("Solicito información sobre mis certificados.", "Petición", "Baja"),
    ("Sugiero mejorar la atención al cliente.", "Sugerencia", "Baja"),
])
def test_infer_urgencia(texto, categoria, esperado):
    assert infer_urgencia(texto, categoria) == esperado


def test_infer_urgencia_is_case_insensitive():
    assert infer_urgencia("EXIJO una respuesta URGENTE", "Reclamo") == "Alta"


# ---------- get_probabilidades ----------

def test_get_probabilidades_returns_confidence_and_distribution(dummy_model):
    confianza, probabilidades = get_probabilidades(dummy_model, "exijo reembolso ya")

    assert confianza is not None
    assert 0 <= confianza <= 100
    assert set(probabilidades.keys()) == set(ID2LABEL.values())
    assert pytest.approx(sum(probabilidades.values()), abs=0.5) == 100


# ---------- clasificar_pqrs ----------

def test_clasificar_pqrs_sin_respuesta(dummy_model):
    resultado = clasificar_pqrs(
        texto="Exijo el reembolso de mi dinero, ya pagué y no recibí nada.",
        model=dummy_model,
        incluir_respuesta=False,
    )

    assert resultado["categoria"] == "Reclamo"
    assert resultado["urgencia"] == "Alta"
    assert resultado["confianza"] is not None
    assert resultado["probabilidades"] is not None
    assert resultado["respuesta_sugerida"] is None
    assert resultado["fuente_respuesta"] is None
    assert resultado["eventos_considerados"] == 0


def test_clasificar_pqrs_con_respuesta_usa_plantilla(dummy_model, isolated_events_file, monkeypatch):
    # Sin API key, debe caer a respuesta de plantilla
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    resultado = clasificar_pqrs(
        texto="Sugiero mejorar la atención en ventanilla.",
        model=dummy_model,
        incluir_respuesta=True,
    )

    assert resultado["categoria"] == "Sugerencia"
    assert resultado["respuesta_sugerida"] is not None
    assert resultado["fuente_respuesta"] == "Plantilla"
    assert resultado["eventos_considerados"] == 0


def test_clasificar_pqrs_cuenta_eventos_activos(dummy_model, isolated_events_file, monkeypatch):
    from src.events import create_event, add_event

    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    ev = create_event(
        titulo="Falla en pagos",
        descripcion="El sistema de pagos presentó intermitencias hoy.",
        vigencia_valor=1,
        vigencia_tipo="dias",
        area="Pagos",
    )
    add_event(ev)

    resultado = clasificar_pqrs(
        texto="Exijo respuesta urgente sobre mi reclamo de pagos.",
        model=dummy_model,
        incluir_respuesta=True,
    )

    assert resultado["eventos_considerados"] == 1


@pytest.mark.parametrize("texto,categoria_esperada", [
    ("Solicito información sobre mi certificado de estudios.", "Petición"),
    ("Estoy muy molesto con el servicio, es pésimo.", "Queja"),
    ("Exijo el reembolso de mi dinero por el reclamo presentado.", "Reclamo"),
    ("Sugiero implementar un sistema de citas en línea.", "Sugerencia"),
])
def test_clasificar_pqrs_categorias(dummy_model, texto, categoria_esperada):
    resultado = clasificar_pqrs(texto=texto, model=dummy_model, incluir_respuesta=False)
    assert resultado["categoria"] == categoria_esperada
