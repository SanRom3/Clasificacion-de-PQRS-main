import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest


class DummyPipeline:
    """Pipeline falso con interfaz compatible (predict + predict_proba)
    para no depender de un modelo entrenado real en las pruebas."""

    classes_ = [0, 1, 2, 3]  # Petición, Queja, Reclamo, Sugerencia

    def predict(self, X):
        # Heurística simple basada en palabras clave para pruebas deterministas
        out = []
        for texto in X:
            t = texto.lower()
            if "exijo" in t or "reembolso" in t or "reclamo" in t:
                out.append(2)  # Reclamo
            elif "queja" in t or "pésimo" in t or "molesto" in t:
                out.append(1)  # Queja
            elif "sugiero" in t or "propongo" in t:
                out.append(3)  # Sugerencia
            else:
                out.append(0)  # Petición
        return np.array(out)

    def predict_proba(self, X):
        preds = self.predict(X)
        probs = []
        for p in preds:
            row = np.full(4, 0.05)
            row[p] = 0.85
            row = row / row.sum()
            probs.append(row)
        return np.array(probs)


@pytest.fixture
def dummy_model():
    return DummyPipeline()


@pytest.fixture
def isolated_events_file(tmp_path, monkeypatch):
    """Redirige EVENTS_FILE a un archivo temporal para no tocar datos reales."""
    from src import events as events_module

    temp_file = tmp_path / "events.json"
    monkeypatch.setattr(events_module, "EVENTS_FILE", temp_file)
    return temp_file
