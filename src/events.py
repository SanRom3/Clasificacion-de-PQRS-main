import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

EVENTS_FILE = Path(__file__).parent.parent / "data" / "events.json"

def create_event(
    titulo: str,
    descripcion: str,
    vigencia_valor: int,
    vigencia_tipo: str,   
    area: str = "General",
) -> dict:
    ahora = datetime.now()

    if vigencia_tipo == "horas":
        expira = ahora + timedelta(hours=vigencia_valor)
    else:
        expira = ahora + timedelta(days=vigencia_valor)

    return {
        "id":          uuid.uuid4().int >> 64,
        "titulo":      titulo.strip(),
        "descripcion": descripcion.strip(),
        "area":        area.strip(),
        "vigencia_valor": vigencia_valor,
        "vigencia_tipo":  vigencia_tipo,
        "creado_en":   ahora.isoformat(),
        "expira_en":   expira.isoformat(),
    }


def is_active(event: dict) -> bool:
    """Retorna True si el evento aún no ha expirado."""
    expira = datetime.fromisoformat(event["expira_en"])
    return datetime.now() < expira


def time_remaining(event: dict) -> str:
    """Retorna un string legible del tiempo restante."""
    expira  = datetime.fromisoformat(event["expira_en"])
    delta   = expira - datetime.now()
    total_s = int(delta.total_seconds())

    if total_s <= 0:
        return "Expirado"
    elif total_s < 3600:
        mins = total_s // 60
        return f"{mins} min restantes"
    elif total_s < 86400:
        horas = total_s // 3600
        return f"{horas}h restantes"
    else:
        dias = total_s // 86400
        horas = (total_s % 86400) // 3600
        return f"{dias}d {horas}h restantes"


def load_events() -> list:
    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not EVENTS_FILE.exists():
        return []
    try:
        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_events(events: list) -> None:
    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)


def add_event(event: dict) -> list:
    events = load_events()
    events.append(event)
    save_events(events)
    return events


def delete_event(event_id: int) -> list:
    events = load_events()
    events = [e for e in events if e["id"] != event_id]
    save_events(events)
    return events


def get_active_events() -> list:
    return [e for e in load_events() if is_active(e)]


def purge_expired(events: list) -> list:
    cutoff = datetime.now() - timedelta(days=7)
    alive  = [
        e for e in events
        if datetime.fromisoformat(e["expira_en"]) > cutoff
    ]
    save_events(alive)
    return alive


def build_context_prompt(active_events: list) -> str:
    if not active_events:
        return ""

    lines = ["EVENTOS INSTITUCIONALES ACTIVOS (ten estos en cuenta al responder):"]
    for e in active_events:
        creado = datetime.fromisoformat(e["creado_en"]).strftime("%d/%m/%Y %H:%M")
        lines.append(
            f"- [{e['area']}] {e['titulo']} "
            f"(registrado: {creado}, vigencia: {e['vigencia_valor']} {e['vigencia_tipo']}): "
            f"{e['descripcion']}"
        )
    lines.append(
        "Si la PQRS está relacionada con alguno de estos eventos, "
        "menciona el inconveniente en tu respuesta y ofrece disculpas institucionales."
    )
    return "\n".join(lines)
