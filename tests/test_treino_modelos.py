"""Smoke tests do pipeline de treino para emoção e sentimento."""

import pandas as pd

from treinar_modelos import (
    EMOCAO_TO_SENTIMENTO,
    baixar_recursos_nltk,
    preprocessar_texto,
    treinar_modelos,
)


def _dataset_sintetico():
    """Gera um dataset pequeno, mas balanceado entre as emoções."""
    base = [
        ("estou muito feliz com o time", "alegria"),
        ("sinto amor pela equipe", "amor"),
        ("estou com medo do prazo", "medo"),
        ("sinto raiva do atraso", "raiva"),
        ("fiquei triste com o resultado", "tristeza"),
        ("fiquei surpreso com a entrega", "surpresa"),
    ]
    df = pd.DataFrame(base * 8, columns=["texto", "emocao"])  # 48 linhas
    df["sentimento"] = df["emocao"].map(EMOCAO_TO_SENTIMENTO)
    df["texto_limpo"] = df["texto"].apply(preprocessar_texto)
    return df


def test_treinar_modelos_retorna_artefatos():
    baixar_recursos_nltk()
    df = _dataset_sintetico()
    artefatos = treinar_modelos(df, max_features=2000, test_size=0.2, random_state=42)

    assert "vectorizer" in artefatos
    assert "modelo_emocao" in artefatos
    assert "modelo_sentimento" in artefatos
    assert "metrics" in artefatos

    assert set(artefatos["metrics"].keys()) == {"emocao", "sentimento"}
    assert 0.0 <= artefatos["metrics"]["emocao"]["accuracy"] <= 1.0
