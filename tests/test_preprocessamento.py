"""Testes de pré-processamento de texto em português."""

import pytest

from treinar_modelos import baixar_recursos_nltk, preprocessar_texto


@pytest.fixture(scope="session", autouse=True)
def _recursos_nltk():
    """Baixa recursos necessários apenas uma vez."""
    baixar_recursos_nltk()


def test_preprocessa_texto_regular():
    texto = "Estou muito FELIZ com este projeto!!!"
    resultado = preprocessar_texto(texto)
    assert resultado.islower()
    assert "feliz" in resultado
    assert "projet" in resultado  # stemming mantém raiz


def test_preprocessa_texto_vazio():
    assert preprocessar_texto("") == ""


def test_preprocessa_texto_com_caracteres_especiais():
    texto = "### *** ???"
    assert preprocessar_texto(texto) == ""
