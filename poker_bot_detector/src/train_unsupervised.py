# joblib sert à sauvegarder et recharger des modèles Python.
import joblib

# pandas sert à lire, manipuler et sauvegarder des tableaux de données.
import pandas as pd

# matplotlib sert à créer et sauvegarder les graphiques.
import matplotlib.pyplot as plt

# IsolationForest est le modèle non supervisé utilisé pour détecter les anomalies.
from sklearn.ensemble import IsolationForest

# Pipeline permet d'enchaîner plusieurs étapes dans un seul modèle.
# Ici : StandardScaler puis IsolationForest.
from sklearn.pipeline import Pipeline

# StandardScaler permet de mettre toutes les variables sur une échelle comparable.
from sklearn.preprocessing import StandardScaler

# On importe les métriques utilisées pour évaluer le modèle après coup.
from sklearn.metrics import (
    accuracy_score,              # taux global de bonnes prédictions
    precision_score,             # parmi les joueurs prédits suspects, combien sont vraiment bots
    recall_score,                # parmi les vrais bots, combien ont été détectés
    f1_score,                    # compromis entre precision et recall
    classification_report,       # rapport détaillé precision / recall / f1-score
    confusion_matrix,            # tableau des bonnes et mauvaises classifications
    ConfusionMatrixDisplay,      # affichage graphique de la matrice de confusion
)

# On importe les chemins et paramètres centralisés dans config.py.
from config import (
    SIMULATED_DATA_PATH,                     # chemin vers simulated_players.csv
    UNSUPERVISED_FEATURE_COLUMNS,            # colonnes d'élite utilisées par le modèle non supervisé
    TARGET_COLUMN,                           # colonne cible : is_bot
    RANDOM_STATE,                            # graine aléatoire pour reproductibilité
    UNSUPERVISED_CONTAMINATION,              # proportion estimée d'anomalies
    UNSUPERVISED_MODEL_PATH,                 # chemin de sauvegarde du modèle non supervisé
    UNSUPERVISED_REPORT_PATH,                # chemin du rapport texte
    UNSUPERVISED_RESULTS_PATH,               # chemin du CSV de résultats
    UNSUPERVISED_CONFUSION_MATRIX_PATH,      # chemin de la matrice de confusion
    ANOMALY_SCORE_DISTRIBUTION_PATH,         # chemin du graphique des scores d'anomalie
)


def load_dataset() -> pd.DataFrame:
    """
    Charge le dataset agrégé par joueur.

    Ce dataset contient une ligne par joueur.
    Les colonnes viennent de l'agrégation des mains préflop simulées.
    """
    # On lit le fichier simulated_players.csv et on le retourne sous forme de DataFrame.
    return pd.read_csv(SIMULATED_DATA_PATH)


def build_isolation_forest_model() -> Pipeline:
    """
    Construit le modèle Isolation Forest.

    Le pipeline contient deux étapes :
    1. StandardScaler : normalise les variables ;
    2. IsolationForest : détecte les profils anormaux.

    Isolation Forest cherche à isoler les observations.
    Une observation facile à isoler est considérée comme plus suspecte.
    """
    # On crée un pipeline scikit-learn.
    model = Pipeline(
        steps=[
            # Première étape : standardisation des variables.
            (
                "scaler",          # nom de l'étape
                StandardScaler(),  # transforme les variables pour avoir moyenne 0 et écart-type 1
            ),

            # Deuxième étape : modèle de détection d'anomalies.
            (
                "isolation_forest",  # nom de l'étape
                IsolationForest(
                    n_estimators=100,                                         # nombre d'arbres utilisés
                    contamination=UNSUPERVISED_CONTAMINATION,                 # proportion attendue d'anomalies (5%)
                    max_samples=256,                                          # nombre max d'observations utilisées par arbre
                    random_state=RANDOM_STATE,                                # reproductibilité
                    n_jobs=-1,                                                # on utilise tous les cœurs pour aller plus vite
                ),
            ),
        ]
    )

    # On retourne le pipeline prêt à être entraîné.
    return model


def create_risk_level(score: float, medium_threshold: float, high_threshold: float) -> str:
    """
    Transforme un score de suspicion numérique en niveau de risque.
    Plus le score est élevé, plus le joueur est suspect.
    """
    # Si le score dépasse le seuil haut, le risque est élevé.
    if score >= high_threshold:
        return "high"

    # Si le score dépasse le seuil moyen, le risque est moyen.
    if score >= medium_threshold:
        return "medium"

    # Sinon, le risque est faible.
    return "low"


def build_results_dataframe(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """
    Crée le tableau final des résultats du modèle non supervisé.
    """
    # On récupère UNIQUEMENT les variables d'élite pour réduire le bruit.
    # On ne met PAS is_bot ici.
    X = df[UNSUPERVISED_FEATURE_COLUMNS]

    # Le modèle prédit si chaque joueur est normal ou anomalie.
    # Résultat : 1 = normal, -1 = anomalie
    isolation_predictions = model.predict(X)

    # On convertit la sortie Isolation Forest au format du projet.
    # Dans notre projet : 0 = non suspect, 1 = suspect
    predicted_suspect = (isolation_predictions == -1).astype(int)

    # decision_function donne un score de normalité.
    # Plus le score est élevé, plus le joueur est normal.
    # Plus le score est faible, plus le joueur est anormal.
    normality_scores = model.decision_function(X)

    # On inverse le signe pour obtenir un score de suspicion.
    # Maintenant : plus anomaly_score est élevé, plus le joueur est suspect.
    anomaly_scores = -normality_scores

    # On crée un DataFrame de résultats avec player_id et is_bot.
    results = df[["player_id", TARGET_COLUMN]].copy()

    # On ajoute la prédiction brute Isolation Forest : 1 ou -1.
    results["isolation_prediction"] = isolation_predictions

    # On ajoute la prédiction convertie : 0 ou 1.
    results["predicted_suspect"] = predicted_suspect

    # On ajoute le score de suspicion.
    results["anomaly_score"] = anomaly_scores

    # Seuil moyen : les 15 % les plus suspects seront medium ou high.
    medium_threshold = results["anomaly_score"].quantile(0.85)

    # Seuil haut : les 5 % les plus suspects seront high (correspond à notre ratio de bots).
    high_threshold = results["anomaly_score"].quantile(0.95)

    # On transforme chaque score en niveau de risque : low / medium / high.
    results["risk_level"] = results["anomaly_score"].apply(
        lambda score: create_risk_level(score, medium_threshold, high_threshold)
    )

    # On retourne le tableau final des résultats.
    return results


def save_model(model: Pipeline) -> None:
    """Sauvegarde le modèle non supervisé."""
    UNSUPERVISED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, UNSUPERVISED_MODEL_PATH)


def save_results(results: pd.DataFrame) -> None:
    """Sauvegarde les résultats de détection dans un fichier CSV."""
    UNSUPERVISED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(UNSUPERVISED_RESULTS_PATH, index=False)


def save_report(results: pd.DataFrame, verbose: bool = True) -> None:
    """Sauvegarde les métriques d'évaluation."""
    UNSUPERVISED_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    y_true = results[TARGET_COLUMN]
    y_pred = results["predicted_suspect"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    report = classification_report(
        y_true,                         
        y_pred,                         
        target_names=["Humain", "Bot/Suspect"],
        zero_division=0,
    )

    prediction_table = pd.crosstab(
        results["isolation_prediction"],  
        results[TARGET_COLUMN],           
        rownames=["isolation_prediction"],
        colnames=["is_bot"],
    )

    risk_table = results["risk_level"].value_counts()

    content = f"""
ÉVALUATION DU MODÈLE NON SUPERVISÉ - ISOLATION FOREST

Rappel important :
Le modèle n'utilise pas la colonne is_bot pendant l'entraînement.
Il s'est basé uniquement sur les {len(UNSUPERVISED_FEATURE_COLUMNS)} variables d'élite suivantes :
{', '.join(UNSUPERVISED_FEATURE_COLUMNS)}

Accuracy : {accuracy:.4f}
Precision : {precision:.4f}
Recall : {recall:.4f}
F1-score : {f1:.4f}

Rapport détaillé :

{report}

Répartition réelle humains/bots par prédiction Isolation Forest :

{prediction_table.to_string()}

Convention :
- isolation_prediction = 1  : profil considéré normal
- isolation_prediction = -1 : profil considéré anomalie/suspect

Répartition des niveaux de risque :

{risk_table.to_string()}
"""

    with open(UNSUPERVISED_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(content)

    if verbose:
        print(content)


def save_confusion_matrix(results: pd.DataFrame) -> None:
    """Sauvegarde la matrice de confusion du modèle non supervisé."""
    UNSUPERVISED_CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    y_true = results[TARGET_COLUMN]
    y_pred = results["predicted_suspect"]
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Humain", "Bot/Suspect"])
    display.plot()
    plt.title("Matrice de confusion - Isolation Forest")
    plt.savefig(UNSUPERVISED_CONFUSION_MATRIX_PATH, bbox_inches="tight")
    plt.close()


def save_anomaly_score_distribution(results: pd.DataFrame) -> None:
    """Sauvegarde la distribution des scores de suspicion."""
    ANOMALY_SCORE_DISTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    humans = results[results[TARGET_COLUMN] == 0]["anomaly_score"]
    bots = results[results[TARGET_COLUMN] == 1]["anomaly_score"]
    
    plt.figure(figsize=(8, 5))
    plt.hist(humans, bins=30, alpha=0.7, label="Humains")
    plt.hist(bots, bins=30, alpha=0.7, label="Bots")
    plt.title("Distribution des scores de suspicion - Isolation Forest")
    plt.xlabel("Score de suspicion (plus c'est élevé, plus c'est anormal)")
    plt.ylabel("Nombre de joueurs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ANOMALY_SCORE_DISTRIBUTION_PATH, bbox_inches="tight")
    plt.close()


def train_unsupervised_model(verbose: bool = True):
    """Pipeline complet du modèle non supervisé."""
    df = load_dataset()

    # On utilise uniquement les variables ciblées
    X = df[UNSUPERVISED_FEATURE_COLUMNS]

    model = build_isolation_forest_model()
    model.fit(X)

    save_model(model)
    results = build_results_dataframe(df, model)
    save_results(results)
    save_report(results, verbose=verbose)
    save_confusion_matrix(results)
    save_anomaly_score_distribution(results)

    return model, results


if __name__ == "__main__":
    model, results = train_unsupervised_model()
    print("Modèle non supervisé Isolation Forest entraîné avec succès.")
    print(f"Modèle sauvegardé dans : {UNSUPERVISED_MODEL_PATH}")
    print(f"Résultats sauvegardés dans : {UNSUPERVISED_RESULTS_PATH}")