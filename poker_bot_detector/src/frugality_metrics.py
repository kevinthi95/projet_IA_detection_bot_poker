"""
Rapport de frugalité et de complexité du projet Poker Bot Detector.

Ce fichier mesure le coût opérationnel du pipeline : volume de données,
nombre de variables, temps d'entraînement, taille des modèles et complexité
algorithmique indicative. L'objectif est de vérifier que le projet reste léger,
compréhensible et cohérent avec une logique d'IA frugale.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, TypeVar

import joblib
import pandas as pd

from config import (
    SIMULATED_DATA_PATH,
    SIMULATED_HANDS_PATH,
    FEATURE_COLUMNS,
    UNSUPERVISED_FEATURE_COLUMNS,
    SUPERVISED_MODEL_PATH,
    UNSUPERVISED_MODEL_PATH,
    FRUGALITY_REPORT_PATH,
    OUTPUTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
)
from train_supervised import train_supervised_model
from train_unsupervised import train_unsupervised_model

T = TypeVar("T")


def get_file_size_kb(path: Path) -> float:
    """Retourne la taille d'un fichier en kilo-octets."""
    if not path.exists() or not path.is_file():
        return 0.0
    return path.stat().st_size / 1024


def get_directory_size_kb(path: Path) -> float:
    """Retourne la taille totale d'un dossier en kilo-octets."""
    if not path.exists() or not path.is_dir():
        return 0.0
    total_bytes = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return total_bytes / 1024


def format_kb(size_kb: float) -> str:
    """Formate une taille en KB ou MB pour rendre le rapport lisible."""
    if size_kb >= 1024:
        return f"{size_kb / 1024:.2f} MB"
    return f"{size_kb:.2f} KB"


def measure_execution_time(function_to_measure: Callable[[], T]) -> tuple[float, T]:
    """Mesure le temps d'exécution d'une fonction et retourne aussi son résultat."""
    start_time = time.perf_counter()
    result = function_to_measure()
    end_time = time.perf_counter()
    return end_time - start_time, result


def load_required_csv(path: Path, description: str) -> pd.DataFrame:
    """Charge un CSV obligatoire avec une erreur claire si le fichier manque."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} introuvable : {path}. Lance d'abord python src/main.py "
            "ou le script de génération des données."
        )
    return pd.read_csv(path)


def count_output_files(path: Path) -> int:
    """Compte les fichiers générés dans un dossier."""
    if not path.exists():
        return 0
    return sum(1 for file in path.rglob("*") if file.is_file())


def get_model_details() -> dict[str, object]:
    """Récupère les paramètres principaux des modèles sauvegardés."""
    details: dict[str, object] = {
        "rf_estimators": "non disponible",
        "rf_max_depth": "non disponible",
        "if_estimators": "non disponible",
        "if_max_samples": "non disponible",
    }

    if SUPERVISED_MODEL_PATH.exists():
        supervised_model = joblib.load(SUPERVISED_MODEL_PATH)
        details["rf_estimators"] = getattr(supervised_model, "n_estimators", "non disponible")
        details["rf_max_depth"] = getattr(supervised_model, "max_depth", "non disponible")

    if UNSUPERVISED_MODEL_PATH.exists():
        unsupervised_pipeline = joblib.load(UNSUPERVISED_MODEL_PATH)
        isolation_forest = None
        if hasattr(unsupervised_pipeline, "named_steps"):
            isolation_forest = unsupervised_pipeline.named_steps.get("isolation_forest")
        else:
            isolation_forest = unsupervised_pipeline

        if isolation_forest is not None:
            details["if_estimators"] = getattr(isolation_forest, "n_estimators", "non disponible")
            details["if_max_samples"] = getattr(isolation_forest, "max_samples", "non disponible")

    return details


def estimate_frugality_label(
    total_model_size_kb: float,
    total_training_time: float,
    n_supervised_features: int,
    n_unsupervised_features: int,
) -> str:
    """Produit un verdict simple à partir de seuils pédagogiques."""
    if (
        total_model_size_kb <= 10 * 1024
        and total_training_time <= 10
        and n_supervised_features <= 50
        and n_unsupervised_features <= 15
    ):
        return "IA frugale"

    if total_model_size_kb <= 50 * 1024 and total_training_time <= 60:
        return "IA modérément frugale"

    return "IA peu frugale"


def generate_frugality_report() -> str:
    """
    Génère le rapport de frugalité complet.

    Le rapport entraîne à nouveau les deux modèles pour mesurer un temps réel
    d'exécution sur la machine utilisée, puis consolide les indicateurs dans
    outputs/reports/frugality_report.txt.
    """
    players_df = load_required_csv(SIMULATED_DATA_PATH, "Dataset joueurs")
    hands_df = load_required_csv(SIMULATED_HANDS_PATH, "Dataset mains")

    n_players = len(players_df)
    n_hands = len(hands_df)
    n_supervised_features = len(FEATURE_COLUMNS)
    n_unsupervised_features = len(UNSUPERVISED_FEATURE_COLUMNS)
    n_bots = int(players_df["is_bot"].sum()) if "is_bot" in players_df.columns else 0
    n_humans = n_players - n_bots
    bot_ratio = n_bots / n_players if n_players else 0.0

    print("Mesure du temps d'entraînement du modèle supervisé...")
    supervised_training_time, _ = measure_execution_time(train_supervised_model)

    print("Mesure du temps d'entraînement du modèle non supervisé...")
    unsupervised_training_time, _ = measure_execution_time(lambda: train_unsupervised_model(verbose=False))

    model_details = get_model_details()

    supervised_model_size_kb = get_file_size_kb(SUPERVISED_MODEL_PATH)
    unsupervised_model_size_kb = get_file_size_kb(UNSUPERVISED_MODEL_PATH)
    total_model_size_kb = supervised_model_size_kb + unsupervised_model_size_kb

    players_dataset_size_kb = get_file_size_kb(SIMULATED_DATA_PATH)
    hands_dataset_size_kb = get_file_size_kb(SIMULATED_HANDS_PATH)
    total_dataset_size_kb = players_dataset_size_kb + hands_dataset_size_kb

    reports_size_kb = get_directory_size_kb(REPORTS_DIR)
    figures_size_kb = get_directory_size_kb(FIGURES_DIR)
    outputs_size_kb = get_directory_size_kb(OUTPUTS_DIR)

    total_training_time = supervised_training_time + unsupervised_training_time
    output_files_count = count_output_files(OUTPUTS_DIR)

    # Estimations simples de complexité. Elles servent d'ordre de grandeur pédagogique,
    # pas de preuve mathématique exacte du coût interne de scikit-learn.
    rf_estimators = model_details["rf_estimators"]
    rf_max_depth = model_details["rf_max_depth"]
    if_estimators = model_details["if_estimators"]
    if_max_samples = model_details["if_max_samples"]

    train_rows = int(n_players * 0.8)
    isolation_sample = 256 if if_max_samples == "auto" or not isinstance(if_max_samples, int) else min(if_max_samples, n_players)
    isolation_log_sample = math.log2(max(isolation_sample, 2))

    rf_complexity = (
        "O(nombre_arbres × lignes_train × variables × profondeur_max)"
    )
    if_complexity = (
        "O(nombre_arbres × max_samples × log2(max_samples) × variables_non_supervisées)"
    )

    estimated_rf_operations = None
    if isinstance(rf_estimators, int) and isinstance(rf_max_depth, int):
        estimated_rf_operations = rf_estimators * train_rows * n_supervised_features * rf_max_depth

    estimated_if_operations = None
    if isinstance(if_estimators, int):
        estimated_if_operations = int(
            if_estimators * isolation_sample * isolation_log_sample * n_unsupervised_features
        )

    frugality_label = estimate_frugality_label(
        total_model_size_kb=total_model_size_kb,
        total_training_time=total_training_time,
        n_supervised_features=n_supervised_features,
        n_unsupervised_features=n_unsupervised_features,
    )

    supervised_features_text = "\n".join(f"- {feature}" for feature in FEATURE_COLUMNS)
    unsupervised_features_text = "\n".join(f"- {feature}" for feature in UNSUPERVISED_FEATURE_COLUMNS)

    estimated_rf_text = (
        f"Environ {estimated_rf_operations:,} opérations indicatives"
        if estimated_rf_operations is not None
        else "Estimation non disponible"
    ).replace(",", " ")

    estimated_if_text = (
        f"Environ {estimated_if_operations:,} opérations indicatives"
        if estimated_if_operations is not None
        else "Estimation non disponible"
    ).replace(",", " ")

    report = f"""
RAPPORT DE FRUGALITÉ ET DE COMPLEXITÉ DU SYSTÈME

1. Données utilisées

Nombre de joueurs analysés : {n_players}
Nombre d'humains : {n_humans}
Nombre de bots : {n_bots}
Ratio de bots : {bot_ratio:.2%}
Nombre de mains préflop simulées : {n_hands}
Nombre moyen de mains par joueur : {(n_hands / n_players) if n_players else 0:.2f}
Nombre de variables utilisées par le modèle supervisé : {n_supervised_features}
Nombre de variables utilisées par le modèle non supervisé : {n_unsupervised_features}

Taille du dataset joueurs : {format_kb(players_dataset_size_kb)}
Taille du dataset mains : {format_kb(hands_dataset_size_kb)}
Taille totale des datasets : {format_kb(total_dataset_size_kb)}

Variables utilisées par Random Forest :
{supervised_features_text}

Variables utilisées par Isolation Forest :
{unsupervised_features_text}

2. Modèle supervisé : Random Forest

Rôle : apprendre à distinguer humains et bots avec la colonne is_bot.
Nombre d'arbres : {rf_estimators}
Profondeur maximale : {rf_max_depth}
Lignes utilisées pour l'entraînement : environ {train_rows}
Temps d'entraînement mesuré : {supervised_training_time:.4f} secondes
Taille du modèle sauvegardé : {format_kb(supervised_model_size_kb)}
Complexité indicative : {rf_complexity}
Estimation pédagogique : {estimated_rf_text}

3. Modèle non supervisé : Isolation Forest

Rôle : détecter les comportements anormaux sans utiliser is_bot pendant l'entraînement.
Nombre d'arbres : {if_estimators}
Max samples : {if_max_samples}
Variables utilisées : {n_unsupervised_features}
Temps d'entraînement mesuré : {unsupervised_training_time:.4f} secondes
Taille du modèle sauvegardé : {format_kb(unsupervised_model_size_kb)}
Complexité indicative : {if_complexity}
Estimation pédagogique : {estimated_if_text}

4. Taille des livrables générés

Taille totale des modèles : {format_kb(total_model_size_kb)}
Taille totale des rapports : {format_kb(reports_size_kb)}
Taille totale des figures : {format_kb(figures_size_kb)}
Taille totale du dossier outputs : {format_kb(outputs_size_kb)}
Nombre de fichiers générés dans outputs : {output_files_count}

5. Synthèse frugalité

Temps total d'entraînement mesuré : {total_training_time:.4f} secondes
Verdict : {frugality_label}

Le projet respecte une logique d'IA frugale pour plusieurs raisons :
- les données sont tabulaires et agrégées, sans image, audio ou vidéo lourde ;
- les modèles sont classiques : Random Forest et Isolation Forest ;
- aucun modèle de deep learning n'est utilisé ;
- aucun entraînement GPU n'est nécessaire ;
- le modèle non supervisé utilise seulement les variables les plus pertinentes ;
- les fichiers générés restent légers et faciles à inspecter.

6. Limites de l'analyse

Ces mesures donnent un ordre de grandeur local : elles dépendent de la machine,
du système d'exploitation et des processus actifs au moment de l'exécution.
La complexité indiquée reste volontairement pédagogique : scikit-learn optimise
certaines opérations en interne, donc le nombre d'opérations estimé ne doit pas
être lu comme une mesure exacte.

Conclusion : le système est cohérent avec un projet étudiant d'IA frugale. Il
produit une détection exploitable sans architecture lourde, avec des modèles
rapides, interprétables et peu coûteux en stockage.
""".strip()

    FRUGALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FRUGALITY_REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"Rapport de frugalité sauvegardé dans : {FRUGALITY_REPORT_PATH}")
    return report


if __name__ == "__main__":
    generate_frugality_report()
