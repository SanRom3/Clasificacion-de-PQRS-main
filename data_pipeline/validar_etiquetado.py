import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", default="data/raw/pqrs_dataset_real_para_revisar.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.review, encoding="utf-8")

    if "categoria_correcta" not in df.columns:
        raise SystemExit("El CSV no tiene columna 'categoria_correcta'.")

    revisadas = df[df["categoria_correcta"].notna() & (df["categoria_correcta"] != "")]
    if revisadas.empty:
        raise SystemExit(
            "No hay filas revisadas todavía. Llena la columna 'categoria_correcta' "
            "con la categoría real para cada fila (Petición/Queja/Reclamo/Sugerencia)."
        )

    aciertos = (revisadas["categoria"] == revisadas["categoria_correcta"]).sum()
    total = len(revisadas)
    precision = aciertos / total * 100

    print(f"Filas revisadas: {total} de {len(df)}")
    print(f"Aciertos del mapeo heurístico: {aciertos} ({precision:.1f}%)")
    print()

    print("Matriz de confusión (heurístico vs. corregido):")
    matriz = pd.crosstab(
        revisadas["categoria"], revisadas["categoria_correcta"],
        rownames=["Heurístico"], colnames=["Corregido"],
    )
    print(matriz.to_string())

    errores = revisadas[revisadas["categoria"] != revisadas["categoria_correcta"]]
    if not errores.empty:
        print(f"\n{len(errores)} ejemplos mal etiquetados por el heurístico:")
        for _, row in errores.head(10).iterrows():
            print(f"  [{row['categoria']} -> {row['categoria_correcta']}] {row['texto'][:80]}")


if __name__ == "__main__":
    main()
