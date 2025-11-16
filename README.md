# Neuron – Entrega de IA (FIAP)

Este repositório reúne todos os itens exigidos pelo desafio de **Artificial Intelligence & Chatbot**:

- `dados_humor_neuron_pt.csv`: base utilizada no notebook e no script de treino.
- `neuron_emocoes_pt.ipynb`: pipeline completo (EDA, pré-processamento, treino, avaliação e exemplo de uso).
- `treinar_modelos.py`: gera o arquivo `modelos_neuron_pt.pkl` (vetorizador + dois modelos).
- `api_flask.py`: expõe os modelos via API REST usando Flask.
- `requirements.txt`: dependências para executar o notebook, o treinamento e a API.
- `modelos_neuron_pt.pkl`: gerado localmente após rodar o script de treino.
- `integrantes.txt`: nomes/RM do time (crie e preencha antes da entrega final).

## Como reproduzir o pipeline fora do notebook

1. Instale as dependências (de preferência em um virtualenv):
   ```bash
   pip install -r requirements.txt
   ```
2. Treine os modelos usando o mesmo CSV presente no notebook:
   ```bash
   python treinar_modelos.py \
     --csv dados_humor_neuron_pt.csv \
     --saida modelos_neuron_pt.pkl
   ```
   O script baixa automaticamente os recursos do NLTK e imprime as métricas de acurácia/relatório.

## Como subir a API

1. Confirme que `modelos_neuron_pt.pkl` existe (passo anterior).
2. Exporte o caminho para o arquivo se quiser usar outro nome/local:
   ```bash
   export NEURON_MODEL_PATH=modelos_neuron_pt.pkl
   ```
3. Execute a API:
   ```bash
   flask --app api_flask run --reload
   # ou
   python api_flask.py
   ```

### Endpoints

- `GET /health` – verificação simples.
- `POST /classificar` – envie `{"texto": "mensagem"}` ou `{"textos": ["m1", "m2"]}` e receba emoção, sentimento e probabilidades.
- `POST /conversas/analisar` – envie `{"mensagens": [{"timestamp": "2024-06-01T10:00:00", "texto": "..."}, ...]}` para obter os rótulos individuais e um resumo por dia/total.

## Deploy no Render

Incluímos o arquivo `render.yaml`, que descreve um serviço web Python no Render pronto para compilar o modelo e servir a API via Gunicorn.

1. Faça o push deste repositório para o GitHub/GitLab/Bitbucket.
2. No painel do Render, clique em **New → Blueprint** e aponte para o repositório. O Render lê o `render.yaml` e propõe o serviço `neuron-api`.
3. O `buildCommand` instala as dependências e roda `python treinar_modelos.py ...`, gerando automaticamente `modelos_neuron_pt.pkl` dentro da imagem. Não é necessário subir o `.pkl` manualmente.
4. O `startCommand` executa `gunicorn --bind 0.0.0.0:$PORT api_flask:app`.
5. Variáveis de ambiente úteis:
   - `NEURON_MODEL_PATH`: caso queira apontar para outro caminho de modelo.
   - `PYTHON_VERSION`: já definido como 3.11.8 no blueprint, ajuste se preferir outra versão suportada.
6. Após o deploy ficar “Live”, teste `GET /health` no domínio do Render para validar.

Se desejar outro plano ou nome do serviço, basta editar o `render.yaml` antes do push.

## Checklist de entrega

1. `dados_humor_neuron_pt.csv`
2. `neuron_emocoes_pt.ipynb`
3. `modelos_neuron_pt.pkl` (gerado por vocês; não sobe pronto aqui)
4. `treinar_modelos.py`
5. `api_flask.py`
6. `integrantes.txt` com nomes/RM

Com isso, basta zipar o diretório e enviar no AVA FIAP.
# Neuron-AI
