
import argparse
import random
import re
import pandas as pd

random.seed(42)

RECLAMO_KEYWORDS = [
    "reembolso", "devuelvan", "devolución", "estafa", "exijo", "exigir",
    "no funciona", "roto", "defectuoso", "incumpl", "denuncia",
    "no ha llegado", "no llegó", "nunca llegó", "no recibí", "cancelar mi pedido",
    "quiero mi dinero", "fraude", "engaño", "no se lo recomiendo a nadie",
]

QUEJA_KEYWORDS = [
    "decepci", "pésimo", "pesimo", "horrible", "malísimo", "malisimo",
    "mala calidad", "no me gustó", "no me gusto", "desilusion", "lamentable",
    "mal servicio", "muy mal", "terrible", "vergüenza",
]

PETICION_PATTERNS = [
    r"\?",  # contiene un signo de interrogación
    r"\balguien sabe\b", r"\bme pueden\b", r"\bpodr[ií]an\b", r"\bnecesito saber\b",
    r"\bquisiera saber\b", r"\bcómo (se|puedo|hago)\b", r"\bsirve para\b",
    r"\bes compatible\b", r"\bcuánto\b", r"\bdónde\b",
]

SUGERENCIA_PATTERNS = [
    r"\bsería mejor\b", r"\bseria mejor\b", r"\bdebería\b", r"\bdeberia\b",
    r"\bsugiero\b", r"\brecomendar[íi]a\b", r"\bme gustaría que\b",
    r"\bechado en falta\b", r"\bfaltaría\b", r"\bfaltaria\b",
    r"\bestaría bien que\b", r"\bpero le falta\b", r"\bojalá\b", r"\bojala\b",
]


def contains_any(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def map_to_pqrs(texto: str, estrellas: int) -> str | None:
    texto_lower = texto.lower()

    if estrellas <= 2:
        if any(k in texto_lower for k in RECLAMO_KEYWORDS):
            return "Reclamo"
        if any(k in texto_lower for k in QUEJA_KEYWORDS):
            return "Queja"
        # 1-2 estrellas sin keyword claro: lo dejamos como Queja por defecto
        return "Queja"

    if estrellas == 3:
        if contains_any(texto, PETICION_PATTERNS):
            return "Petición"
        return None  # ambiguo, lo descartamos

    if estrellas >= 4:
        if contains_any(texto, SUGERENCIA_PATTERNS):
            return "Sugerencia"
        return None  # reseña puramente positiva, no es una PQRS

    return None


def infer_urgencia(texto: str, categoria: str) -> str:
    texto_lower = texto.lower()
    alta_kw = ["urgente", "inmediato", "exijo", "denuncia", "reembolso", "estafa"]

    if categoria == "Reclamo":
        return "Alta" if any(k in texto_lower for k in alta_kw) else "Media"
    if categoria == "Queja":
        return "Media"
    return "Baja"


def build_dataset(reviews_df: pd.DataFrame, n_per_class: int = 1000) -> pd.DataFrame:
    reviews_df = reviews_df.copy()
    reviews_df["estrellas"] = reviews_df["label"] + 1
    reviews_df["texto"] = reviews_df["text"].astype(str).str.strip()
    reviews_df = reviews_df[reviews_df["texto"].str.len() > 15]  # descarta reseñas muy cortas

    records = []
    for _, row in reviews_df.iterrows():
        categoria = map_to_pqrs(row["texto"], row["estrellas"])
        if categoria is None:
            continue
        urgencia = infer_urgencia(row["texto"], categoria)
        records.append({
            "texto": row["texto"],
            "categoria": categoria,
            "urgencia": urgencia,
            "fuente": "amazon_reviews_multi_es",
        })

    df = pd.DataFrame(records)

    balanced = []
    for cat in ["Petición", "Queja", "Reclamo", "Sugerencia"]:
        subset = df[df["categoria"] == cat]
        n = min(len(subset), n_per_class)
        balanced.append(subset.sample(n=n, random_state=42))
        print(f"  {cat}: {len(subset)} disponibles -> {n} tomadas")

    return pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/pqrs_dataset_real.csv")
    parser.add_argument("--review-output", default="data/raw/pqrs_dataset_real_para_revisar.csv")
    parser.add_argument("--sample", type=int, default=1000,
                         help="Máximo de ejemplos por categoría (default 1000)")
    parser.add_argument("--review-sample", type=int, default=60,
                         help="Cuántos ejemplos por categoría exportar para revisión manual")
    args = parser.parse_args()

    print("[*] Descargando amazon_reviews_multi_es desde HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("SetFit/amazon_reviews_multi_es", split="train")
    reviews_df = ds.to_pandas()[["text", "label"]]
    print(f"    {len(reviews_df)} reseñas cargadas.")

    print("\n[*] Aplicando mapeo heurístico a categorías PQRS...")
    df = build_dataset(reviews_df, n_per_class=args.sample)

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df[["texto", "categoria", "urgencia"]].to_csv(args.output, index=False, encoding="utf-8")
    print(f"\n[OK] Dataset completo guardado en: {args.output}")
    print(f"     Total: {len(df)} registros")
    print(df["categoria"].value_counts().to_string())

    review_parts = []
    for cat in ["Petición", "Queja", "Reclamo", "Sugerencia"]:
        subset = df[df["categoria"] == cat]
        n = min(len(subset), args.review_sample)
        review_parts.append(subset.sample(n=n, random_state=7))
    review_df = pd.concat(review_parts).reset_index(drop=True)
    review_df["categoria_correcta"] = ""  # columna vacía para que la llenes a mano
    review_df.to_csv(args.review_output, index=False, encoding="utf-8")
    print(f"\n[OK] Muestra para revisión manual guardada en: {args.review_output}")
    print(f"     ({len(review_df)} filas, columna 'categoria_correcta' vacía para corregir)")


if __name__ == "__main__":
    main()
