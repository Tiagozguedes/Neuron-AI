"""API REST (Flask) para servir os modelos Neuron."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import nltk
from flask import Flask, jsonify, request
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

import pickle

_NON_LETTERS = re.compile(r"[^a-záéíóúâêôãõç\s]")


def _env_bool(var_name: str, default: bool = True) -> bool:
    valor = os.getenv(var_name)
    if valor is None:
        return default
    return valor.strip().lower() not in {"0", "false", "no"}


def garantir_modelo(caminho_modelos: Path) -> None:
    """Treina automaticamente os modelos se o arquivo .pkl ainda não existir."""
    if caminho_modelos.exists():
        return

    if not _env_bool("NEURON_AUTO_TRAIN", True):
        raise FileNotFoundError(
            f"Arquivo de modelos '{caminho_modelos}' não encontrado. "
            "Execute `python treinar_modelos.py` manualmente ou habilite o auto-treinamento."
        )

    dataset_path = Path(os.getenv("NEURON_DATASET_PATH", "dados_humor_neuron_pt.csv"))
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Arquivo de modelos '{caminho_modelos}' não encontrado e dataset '{dataset_path}' ausente. "
            "Disponibilize o CSV ou exporte NEURON_DATASET_PATH apontando para ele."
        )

    try:
        from treinar_modelos import (
            baixar_recursos_nltk as baixar_recursos_treino,
            carregar_dataset,
            salvar_modelos,
            treinar_modelos as treinar_pipeline,
        )
    except ImportError as exc:
        raise RuntimeError("Não foi possível importar treinar_modelos.py para o treino automático.") from exc

    print(f"[Neuron] Gerando modelos automaticamente a partir de {dataset_path}...")
    baixar_recursos_treino()
    df = carregar_dataset(dataset_path)
    artefatos = treinar_pipeline(df, max_features=5000, test_size=0.2, random_state=42)
    salvar_modelos(artefatos, caminho_modelos)
    print(f"[Neuron] Modelos treinados e salvos em {caminho_modelos}.")


def baixar_recursos_nltk() -> None:
    for recurso in ("stopwords", "rslp"):
        try:
            nltk.data.find(f"corpora/{recurso}")
        except LookupError:
            nltk.download(recurso)


def preprocessar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = _NON_LETTERS.sub(" ", texto)
    tokens = texto.split()
    stop_words = set(stopwords.words("portuguese"))
    tokens = [tok for tok in tokens if tok not in stop_words]
    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(tok) for tok in tokens]
    return " ".join(tokens)


class EmotionService:
    """Encapsula o carregamento dos modelos e as previsões."""

    def __init__(self, caminho_modelos: Path) -> None:
        baixar_recursos_nltk()
        with open(caminho_modelos, "rb") as f:
            artefatos = pickle.load(f)

        self.vectorizer = artefatos["vectorizer"]
        self.modelo_emocao = artefatos["modelo_emocao"]
        self.modelo_sentimento = artefatos["modelo_sentimento"]

    def _vectorizar(self, textos: Iterable[str]):
        corpus = [preprocessar_texto(t) for t in textos]
        return self.vectorizer.transform(corpus)

    def _probabilidades(self, modelo, matriz):
        probs = modelo.predict_proba(matriz)
        classes = modelo.classes_
        resultados = []
        for linha in probs:
            resultados.append({cls: float(valor) for cls, valor in zip(classes, linha)})
        return resultados

    def classificar(self, textos: List[str]):
        matriz = self._vectorizar(textos)
        emo = self.modelo_emocao.predict(matriz)
        emo_probs = self._probabilidades(self.modelo_emocao, matriz)

        sen = self.modelo_sentimento.predict(matriz)
        sen_probs = self._probabilidades(self.modelo_sentimento, matriz)

        saida = []
        for idx, texto in enumerate(textos):
            saida.append(
                {
                    "texto": texto,
                    "emocao": emo[idx],
                    "sentimento": sen[idx],
                    "emocao_scores": emo_probs[idx],
                    "sentimento_scores": sen_probs[idx],
                }
            )
        return saida

    def analisar_conversa(self, mensagens: List[Dict[str, str]]):
        textos = [msg.get("texto", "") for msg in mensagens]
        resultados = self.classificar(textos)

        mensagens_enriquecidas = []
        emocao_por_dia = defaultdict(Counter)
        sentimento_por_dia = defaultdict(Counter)
        emocao_total = Counter()
        sentimento_total = Counter()

        for msg, res in zip(mensagens, resultados):
            timestamp = msg.get("timestamp")
            data_iso = None
            if timestamp:
                try:
                    dt_obj = datetime.fromisoformat(timestamp)
                    data_iso = dt_obj.date().isoformat()
                except ValueError:
                    data_iso = None

            registro = {**res, "timestamp": timestamp, "data": data_iso}
            mensagens_enriquecidas.append(registro)

            emocao_total[res["emocao"]] += 1
            sentimento_total[res["sentimento"]] += 1
            if data_iso:
                emocao_por_dia[data_iso][res["emocao"]] += 1
                sentimento_por_dia[data_iso][res["sentimento"]] += 1

        return {
            "mensagens": mensagens_enriquecidas,
            "resumo": {
                "emocao_por_dia": {dia: dict(cont) for dia, cont in emocao_por_dia.items()},
                "sentimento_por_dia": {dia: dict(cont) for dia, cont in sentimento_por_dia.items()},
                "emocao_total": dict(emocao_total),
                "sentimento_total": dict(sentimento_total),
            },
        }


def criar_app() -> Flask:
    caminho = Path(os.getenv("NEURON_MODEL_PATH", "modelos_neuron_pt.pkl"))
    garantir_modelo(caminho)
    service = EmotionService(caminho)
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/classificar")
    def endpoint_classificar():
        payload = _obter_json()
        if payload is None:
            return jsonify({"error": "JSON inválido"}), 400

        if "texto" in payload:
            resultado = service.classificar([payload["texto"]])
            return jsonify(resultado[0])

        if "textos" in payload:
            textos = payload["textos"]
            if not isinstance(textos, list) or not all(isinstance(t, str) for t in textos):
                return jsonify({"error": "Campo 'textos' deve ser lista de strings."}), 400
            resultado = service.classificar(textos)
            return jsonify({"resultados": resultado})

        return jsonify({"error": "Envie 'texto' ou 'textos'."}), 400

    @app.post("/conversas/analisar")
    def endpoint_conversa():
        payload = _obter_json()
        if payload is None:
            return jsonify({"error": "JSON inválido"}), 400

        mensagens = payload.get("mensagens")
        if not isinstance(mensagens, list):
            return jsonify({"error": "Campo 'mensagens' deve ser uma lista."}), 400
        if not all(isinstance(m, dict) and "texto" in m for m in mensagens):
            return jsonify({"error": "Cada mensagem precisa conter ao menos 'texto'."}), 400

        resultado = service.analisar_conversa(mensagens)
        return jsonify(resultado)

    return app


def _obter_json():
    try:
        return request.get_json(force=True)
    except Exception:
        return None


app = criar_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
