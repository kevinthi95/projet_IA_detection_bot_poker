from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "unsupervised_model.pkl"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "unsupervised_metrics.txt"
RESULTS_PATH = PROJECT_ROOT / "data" / "predictions" / "unsupervised_detection_results.csv"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "outputs" / "figures" / "unsupervised_confusion_matrix.png"
ANOMALY_SCORE_DISTRIBUTION_PATH = PROJECT_ROOT / "outputs" / "figures" / "anomaly_score_distribution.png"


def check_file_exists(path: Path, name: str) -> None:
    assert path.exists(), f"Fichier manquant : {name}"


def check_file_not_empty(path: Path, name: str) -> None:
    assert path.stat().st_size > 0, f"Fichier vide : {name}"


def check_results_file(path: Path) -> None:
    df = pd.read_csv(path)

    expected_columns = [
        "player_id",
        "is_bot",
        "isolation_prediction",
        "predicted_suspect",
        "anomaly_score",
        "risk_level",
    ]

    for column in expected_columns:
        assert column in df.columns, f"Colonne manquante : {column}"

    assert set(df["predicted_suspect"].unique()).issubset({0, 1}), (
        "predicted_suspect doit contenir uniquement 0 ou 1."
    )

    assert set(df["isolation_prediction"].unique()).issubset({-1, 1}), (
        "isolation_prediction doit contenir uniquement -1 ou 1."
    )

    assert set(df["risk_level"].unique()).issubset({"low", "medium", "high"}), (
        "risk_level doit contenir uniquement low, medium ou high."
    )

    assert df["anomaly_score"].notna().all(), (
        "La colonne anomaly_score contient des valeurs manquantes."
    )


def check_report_content(path: Path) -> None:
    content = path.read_text(encoding="utf-8")

    expected_keywords = [
        "ISOLATION FOREST",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Bot/Suspect",
        "risk_level",
    ]

    for keyword in expected_keywords:
        assert keyword in content, f"Mot-clé manquant dans le rapport : {keyword}"


def main():
    print("Vérification des fichiers générés par la partie 3...")

    files_to_check = {
        "unsupervised_model.pkl": MODEL_PATH,
        "unsupervised_metrics.txt": REPORT_PATH,
        "unsupervised_detection_results.csv": RESULTS_PATH,
        "unsupervised_confusion_matrix.png": CONFUSION_MATRIX_PATH,
        "anomaly_score_distribution.png": ANOMALY_SCORE_DISTRIBUTION_PATH,
    }

    for name, path in files_to_check.items():
        check_file_exists(path, name)
        check_file_not_empty(path, name)
        print(f"{name} : OK")

    print()
    print("Vérification du fichier de résultats...")
    check_results_file(RESULTS_PATH)
    print("Fichier de résultats : OK")

    print()
    print("Vérification du rapport...")
    check_report_content(REPORT_PATH)
    print("Rapport : OK")

    print()
    print("Tous les contrôles de la partie 3 sont validés.")
    print("Le modèle non supervisé Isolation Forest fonctionne correctement.")


if __name__ == "__main__":
    main()