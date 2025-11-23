"""Testes básicos da API Flask exposta pelo projeto Neuron."""

import importlib
from typing import List, Tuple

import pandas as pd
import pytest
from treinar_modelos import (
    EMOCAO_TO_SENTIMENTO,
    baixar_recursos_nltk,
    preprocessar_texto,
    salvar_modelos,
    treinar_modelos,
)


def _dataset_sintetico() -> pd.DataFrame:
    base: List[Tuple[str, str]] = [
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


@pytest.fixture(scope="session")
def modelo_temporario(tmp_path_factory):
    """Gera um .pkl pequeno para os testes, evitando depender do arquivo real."""
    baixar_recursos_nltk()
    df = _dataset_sintetico()
    artefatos = treinar_modelos(df, max_features=2000, test_size=0.2, random_state=42)
    modelo_path = tmp_path_factory.mktemp("modelos") / "modelos_test.pkl"
    salvar_modelos(artefatos, modelo_path)
    return modelo_path


@pytest.fixture()
def flask_app(modelo_temporario, monkeypatch):
    """Configura a app Flask usando o modelo temporário gerado pelo teste."""
    monkeypatch.setenv("NEURON_MODEL_PATH", str(modelo_temporario))
    monkeypatch.setenv("NEURON_AUTO_TRAIN", "0")
    api_flask = importlib.reload(importlib.import_module("api_flask"))
    app = api_flask.criar_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture()
def client(flask_app):
    return flask_app.test_client()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_classificar_texto_unico(client):
    resp = client.post("/classificar", json={"texto": "hoje estou muito feliz"})
    assert resp.status_code == 200
    corpo = resp.get_json()
    assert corpo["emocao"]
    assert corpo["sentimento"]


def test_classificar_lista_de_textos(client):
    resp = client.post("/classificar", json={"textos": ["estou animado", "estou com medo do prazo"]})
    assert resp.status_code == 200
    corpo = resp.get_json()
    assert "resultados" in corpo
    assert len(corpo["resultados"]) == 2


def test_analisar_conversa(client):
    payload = {
        "mensagens": [
            {"timestamp": "2024-01-01T10:00:00", "texto": "estou feliz com a entrega"},
            {"timestamp": "2024-01-02T09:00:00", "texto": "estou preocupado com o prazo"},
        ]
    }
    resp = client.post("/conversas/analisar", json=payload)
    assert resp.status_code == 200
    corpo = resp.get_json()
    assert "mensagens" in corpo
    assert "resumo" in corpo
    assert len(corpo["mensagens"]) == 2
