"""Script de treinamento para o projeto Neuron."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, Any

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


EMOCAO_TO_SENTIMENTO = {
    "alegria": "positivo",
    "amor": "positivo",
    "surpresa": "positivo",
    "tristeza": "negativo",
    "raiva": "negativo",
    "medo": "negativo",
}


def baixar_recursos_nltk() -> None:
    """Baixa stopwords e stemmer do NLTK, garantindo que o pipeline funcione em máquinas limpas."""
    for recurso in ("stopwords", "rslp"):
        try:
            nltk.data.find(f"corpora/{recurso}")
        except LookupError:
            nltk.download(recurso)


def preprocessar_texto(texto: str, stopwords_custom=None) -> str:
    """
    Normaliza um texto em português para alimentar o modelo.

    - Converte para minúsculas.
    - Remove caracteres não alfabéticos.
    - Remove stopwords (português) e opcionais customizadas.
    - Aplica stemming via RSLP.
    """
    if not isinstance(texto, str):
        return ""

    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúâêôãõç\s]", " ", texto)

    tokens = texto.split()
    stop_words = set(stopwords.words("portuguese"))
    if stopwords_custom:
        stop_words.update({w.lower() for w in stopwords_custom})
    tokens = [tok for tok in tokens if tok not in stop_words]

    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(tok) for tok in tokens]
    return " ".join(tokens)


def carregar_dataset(caminho: Path) -> pd.DataFrame:
    """
    Lê o CSV de humor em PT-BR e gera colunas derivadas:
    - sentimento (mapa de emoção → positivo/negativo)
    - texto_limpo (pré-processado para o vetor TF-IDF)
    """
    df = pd.read_csv(caminho)
    obrigatorias = {"texto", "emocao"}
    if not obrigatorias.issubset(df.columns):
        raise ValueError("O CSV deve conter as colunas 'texto' e 'emocao'.")

    df = df.copy()
    df["sentimento"] = df["emocao"].map(EMOCAO_TO_SENTIMENTO)
    df["texto_limpo"] = df["texto"].apply(preprocessar_texto)
    return df


def treinar_modelos(df: pd.DataFrame, max_features: int, test_size: float, random_state: int) -> Dict[str, Any]:
    """
    Treina dois modelos supervisionados (emoção e sentimento) usando TF-IDF + Logistic Regression.

    Retorna um dicionário com vetorizador, modelos e métricas para persistência posterior.
    """
    # Um único split mantém os mesmos exemplos para emoção e sentimento.
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df["emocao"],
    )

    X_train = df.loc[train_idx, "texto_limpo"]
    X_test = df.loc[test_idx, "texto_limpo"]
    y_train_e = df.loc[train_idx, "emocao"]
    y_test_e = df.loc[test_idx, "emocao"]
    y_train_s = df.loc[train_idx, "sentimento"]
    y_test_s = df.loc[test_idx, "sentimento"]

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    modelo_emocao = LogisticRegression(max_iter=1000, multi_class="multinomial")
    modelo_emocao.fit(X_train_vec, y_train_e)

    modelo_sentimento = LogisticRegression(max_iter=1000)
    modelo_sentimento.fit(X_train_vec, y_train_s)

    metrics = {
        "emocao": gerar_metricas(modelo_emocao, X_test_vec, y_test_e),
        "sentimento": gerar_metricas(modelo_sentimento, X_test_vec, y_test_s),
    }

    artefatos = {
        "vectorizer": vectorizer,
        "modelo_emocao": modelo_emocao,
        "modelo_sentimento": modelo_sentimento,
        "metrics": metrics,
    }
    return artefatos


def gerar_metricas(modelo, X_test, y_true) -> Dict[str, Any]:
    """Gera acurácia e classification_report do sklearn como dicionário."""
    y_pred = modelo.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def salvar_modelos(artefatos, caminho: Path) -> None:
    """Persiste vetor, modelos e métricas em um único arquivo .pkl para servir na API."""
    with open(caminho, "wb") as f:
        pickle.dump(artefatos, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Treinar modelos de emoção e sentimento.")
    parser.add_argument("--csv", default="dados_humor_neuron_pt.csv", help="Caminho do dataset.")
    parser.add_argument("--saida", default="modelos_neuron_pt.pkl", help="Arquivo de saída .pkl.")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baixar_recursos_nltk()
    df = carregar_dataset(Path(args.csv))
    artefatos = treinar_modelos(df, args.max_features, args.test_size, args.random_state)
    salvar_modelos(artefatos, Path(args.saida))

    print(f"Modelos salvos em {args.saida}")
    print("Métricas obtidas:")
    print(json.dumps(artefatos["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
