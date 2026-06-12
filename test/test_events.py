from datetime import datetime, timedelta

from src.events import (
    create_event,
    is_active,
    time_remaining,
    load_events,
    save_events,
    add_event,
    delete_event,
    get_active_events,
    purge_expired,
    build_context_prompt,
)


def test_create_event_horas(isolated_events_file):
    ev = create_event(
        titulo="Caída del portal",
        descripcion="El portal de pagos no responde.",
        vigencia_valor=2,
        vigencia_tipo="horas",
        area="Pagos",
    )

    assert ev["titulo"] == "Caída del portal"
    assert ev["area"] == "Pagos"
    assert ev["vigencia_valor"] == 2
    assert ev["vigencia_tipo"] == "horas"
    assert "id" in ev
    assert is_active(ev) is True


def test_create_event_strips_whitespace(isolated_events_file):
    ev = create_event(
        titulo="  Mantenimiento  ",
        descripcion="  Ventana de mantenimiento programado.  ",
        vigencia_valor=1,
        vigencia_tipo="dias",
    )

    assert ev["titulo"] == "Mantenimiento"
    assert ev["descripcion"] == "Ventana de mantenimiento programado."
    assert ev["area"] == "General"


def test_is_active_for_expired_event(isolated_events_file):
    ev = create_event(
        titulo="Evento viejo",
        descripcion="Este evento ya expiró hace tiempo.",
        vigencia_valor=1,
        vigencia_tipo="horas",
    )
    # Forzamos expiración manual
    ev["expira_en"] = (datetime.now() - timedelta(hours=1)).isoformat()

    assert is_active(ev) is False


def test_time_remaining_formats(isolated_events_file):
    ahora = datetime.now()

    ev_min = create_event("T", "Descripcion de prueba suficientemente larga.", 1, "horas")
    ev_min["expira_en"] = (ahora + timedelta(minutes=30)).isoformat()
    assert "min restantes" in time_remaining(ev_min)

    ev_horas = create_event("T", "Descripcion de prueba suficientemente larga.", 1, "horas")
    ev_horas["expira_en"] = (ahora + timedelta(hours=5)).isoformat()
    assert "h restantes" in time_remaining(ev_horas)

    ev_dias = create_event("T", "Descripcion de prueba suficientemente larga.", 1, "dias")
    ev_dias["expira_en"] = (ahora + timedelta(days=2, hours=3)).isoformat()
    assert "d" in time_remaining(ev_dias) and "h restantes" in time_remaining(ev_dias)

    ev_expirado = create_event("T", "Descripcion de prueba suficientemente larga.", 1, "horas")
    ev_expirado["expira_en"] = (ahora - timedelta(minutes=1)).isoformat()
    assert time_remaining(ev_expirado) == "Expirado"


def test_load_events_empty_when_no_file(isolated_events_file):
    assert load_events() == []


def test_add_and_load_event(isolated_events_file):
    ev = create_event("Incidente", "Descripcion de prueba suficientemente larga.", 1, "dias")
    add_event(ev)

    events = load_events()
    assert len(events) == 1
    assert events[0]["titulo"] == "Incidente"


def test_delete_event(isolated_events_file):
    ev1 = create_event("Uno", "Descripcion de prueba suficientemente larga.", 1, "dias")
    ev2 = create_event("Dos", "Otra descripcion de prueba suficientemente larga.", 1, "dias")
    add_event(ev1)
    add_event(ev2)

    delete_event(ev1["id"])
    events = load_events()

    assert len(events) == 1
    assert events[0]["id"] == ev2["id"]


def test_get_active_events_filters_expired(isolated_events_file):
    activo = create_event("Activo", "Descripcion de prueba suficientemente larga.", 1, "dias")
    expirado = create_event("Expirado", "Descripcion de prueba suficientemente larga.", 1, "horas")
    expirado["expira_en"] = (datetime.now() - timedelta(hours=1)).isoformat()

    save_events([activo, expirado])

    activos = get_active_events()
    assert len(activos) == 1
    assert activos[0]["titulo"] == "Activo"


def test_purge_expired_removes_old_events(isolated_events_file):
    reciente = create_event("Reciente", "Descripcion de prueba suficientemente larga.", 1, "dias")
    viejo = create_event("Viejo", "Descripcion de prueba suficientemente larga.", 1, "horas")
    viejo["expira_en"] = (datetime.now() - timedelta(days=10)).isoformat()

    save_events([reciente, viejo])
    restantes = purge_expired(load_events())

    titulos = [e["titulo"] for e in restantes]
    assert "Reciente" in titulos
    assert "Viejo" not in titulos


def test_build_context_prompt_empty_when_no_events():
    assert build_context_prompt([]) == ""


def test_build_context_prompt_includes_event_info(isolated_events_file):
    ev = create_event(
        titulo="Falla en pagos",
        descripcion="El sistema de pagos presentó intermitencias.",
        vigencia_valor=1,
        vigencia_tipo="dias",
        area="Pagos",
    )

    prompt = build_context_prompt([ev])

    assert "EVENTOS INSTITUCIONALES ACTIVOS" in prompt
    assert "Falla en pagos" in prompt
    assert "Pagos" in prompt
    assert "El sistema de pagos presentó intermitencias." in prompt
