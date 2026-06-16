import pytest
from fastapi.testclient import TestClient

from api import main as api_main


@pytest.fixture
def client(dummy_model, isolated_events_file, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with TestClient(api_main.app) as test_client:
        # El modelo real puede no existir en el entorno de pruebas;
        # inyectamos el dummy directamente en el estado de la app.
        api_main.state["model"] = dummy_model
        yield test_client


# ---------- General ----------

def test_info_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["nombre"] == "Clasificador AutoML de PQRS"
    assert "GET  /health" in data["endpoints"]


def test_health_endpoint_with_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["estado"] == "ok"
    assert data["modelo_cargado"] is True
    assert data["eventos_activos"] == 0


# ---------- Clasificación ----------

def test_classify_endpoint(client):
    resp = client.post("/classify", json={"texto": "Exijo el reembolso de mi dinero ya."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["categoria"] == "Reclamo"
    assert data["urgencia"] == "Alta"
    assert data["respuesta_sugerida"] is None


def test_classify_endpoint_incluye_semaforo_riesgo(client):
    resp = client.post("/classify", json={"texto": "Exijo el reembolso de mi dinero ya."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["nivel_riesgo"] in {"Verde", "Amarillo", "Rojo"}
    assert 0 <= data["score_riesgo"] <= 100
    # Reclamo + Urgencia Alta -> riesgo alto
    assert data["nivel_riesgo"] == "Rojo"


def test_classify_batch_incluye_semaforo_riesgo(client):
    resp = client.post("/classify/batch", json={"textos": [
        "Solicito información sobre mi certificado.",
        "Exijo el reembolso de mi dinero.",
    ]})
    assert resp.status_code == 200
    for resultado in resp.json()["resultados"]:
        assert resultado["nivel_riesgo"] in {"Verde", "Amarillo", "Rojo"}
        assert 0 <= resultado["score_riesgo"] <= 100


def test_classify_endpoint_con_respuesta(client):
    resp = client.post(
        "/classify",
        params={"incluir_respuesta": True},
        json={"texto": "Sugiero mejorar la atención al cliente en ventanilla."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["categoria"] == "Sugerencia"
    assert data["respuesta_sugerida"] is not None
    assert data["fuente_respuesta"] == "Plantilla"


def test_classify_endpoint_validation_error(client):
    # Texto demasiado corto (< 5 caracteres)
    resp = client.post("/classify", json={"texto": "Hi"})
    assert resp.status_code == 422


def test_classify_batch(client):
    resp = client.post("/classify/batch", json={"textos": [
        "Solicito información sobre mi certificado.",
        "Exijo el reembolso de mi dinero.",
        "Sugiero mejorar el sistema de citas.",
    ]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert sum(data["resumen"].values()) == 3


def test_classify_batch_too_many_texts(client):
    textos = ["Solicito información sobre mi trámite." for _ in range(51)]
    resp = client.post("/classify/batch", json={"textos": textos})
    # Pydantic valida max_length=50 antes de llegar al handler
    assert resp.status_code == 422


# ---------- Events ----------

def test_create_list_and_delete_event(client):
    payload = {
        "titulo": "Caída del sistema de pagos",
        "descripcion": "El sistema presentó fallas entre las 10am y 2pm de hoy.",
        "area": "Pagos",
        "vigencia_valor": 2,
        "vigencia_tipo": "horas",
    }

    create_resp = client.post("/events", json=payload)
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["titulo"] == payload["titulo"]
    assert created["activo"] is True

    list_resp = client.get("/events")
    assert list_resp.status_code == 200
    events = list_resp.json()
    assert len(events) == 1
    assert events[0]["id"] == created["id"]

    active_resp = client.get("/events", params={"solo_activos": True})
    assert len(active_resp.json()) == 1

    delete_resp = client.delete(f"/events/{created['id']}")
    assert delete_resp.status_code == 200

    list_after = client.get("/events")
    assert list_after.json() == []


def test_delete_nonexistent_event_returns_404(client):
    resp = client.delete("/events/999999999")
    assert resp.status_code == 404


def test_create_event_invalid_vigencia_tipo(client):
    payload = {
        "titulo": "Evento inválido",
        "descripcion": "Descripcion de prueba suficientemente larga.",
        "area": "General",
        "vigencia_valor": 1,
        "vigencia_tipo": "semanas",
    }
    resp = client.post("/events", json=payload)
    assert resp.status_code == 422


def test_health_reflects_active_events(client):
    payload = {
        "titulo": "Mantenimiento programado",
        "descripcion": "Mantenimiento del portal durante la madrugada.",
        "area": "Infraestructura",
        "vigencia_valor": 1,
        "vigencia_tipo": "dias",
    }
    client.post("/events", json=payload)

    resp = client.get("/health")
    assert resp.json()["eventos_activos"] == 1


# ---------- Modelo no disponible ----------

def test_classify_returns_503_when_model_not_loaded(dummy_model, isolated_events_file, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with TestClient(api_main.app) as test_client:
        api_main.state["model"] = None
        resp = test_client.post("/classify", json={"texto": "Solicito información de mi trámite."})

    assert resp.status_code == 503