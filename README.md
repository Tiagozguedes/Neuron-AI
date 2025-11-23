# Neuron AI – Motor emocional da plataforma Neuron

## 1. Visão geral do projeto
- Problema: burnout, ansiedade e estresse corporativo sem ferramentas éticas para monitorar bem-estar.
- Solução: IA emocional em português com privacy by design, anonimização e gamificação para adesão voluntária.
- Mercado: SaaS B2B com planos Essencial (dashboards + predição) e Premium (intervenções humanas do Time Neuron); acesso multinível para colaboradores, gestores e RH sem expor dados individuais.
- Papel deste repositório: entregar o core de IA (classificação de emoção e sentimento) consumido por outros serviços da Neuron via API REST.

## 2. Arquitetura da solução
1) Dataset `dados_humor_neuron_pt.csv` (textos + emoção).  
2) Treino em `treinar_modelos.py` (pré-processamento PT-BR, TF-IDF, Logistic Regression) → `modelos_neuron_pt.pkl` (vetorizador + 2 modelos + métricas).  
3) Notebook `neuron_emocoes_pt.ipynb` com EDA, treino e avaliação.  
4) API REST Flask (`api_flask.py`) carrega o `.pkl` e expõe endpoints `/health`, `/classificar` e `/conversas/analisar` para os demais módulos (app do colaborador, painel do gestor/RH, integrações).  
5) Deploy no Render com `render.yaml`.

## 3. Arquivos obrigatórios (entrega FIAP)
- `dados_humor_neuron_pt.csv`
- `neuron_emocoes_pt.ipynb`
- `modelos_neuron_pt.pkl`
- `treinar_modelos.py`
- `api_flask.py`
- `integrantes.txt`
- `README.md`, `requirements.txt`, `.gitignore`, `render.yaml`, `tests/`

## 4. Como executar localmente
1. Criar ambiente:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .\\.venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2. Treino (opcional se já tiver `modelos_neuron_pt.pkl`):
   ```bash
   python treinar_modelos.py \
     --csv dados_humor_neuron_pt.csv \
     --saida modelos_neuron_pt.pkl
   ```
   - Hiperparâmetros: `--max-features`, `--test-size`, `--random-state`.
3. Subir a API:
   ```bash
   flask --app api_flask run --reload
   # ou
   python api_flask.py
   ```
4. Variáveis úteis:
   - `NEURON_MODEL_PATH`: caminho do `.pkl` (padrão `modelos_neuron_pt.pkl`).
   - `NEURON_DATASET_PATH`: CSV usado para auto-treino (padrão `dados_humor_neuron_pt.csv`).
   - `NEURON_AUTO_TRAIN`: defina `0`/`false` para exigir modelo pronto e não treinar automaticamente.
   - `NEURON_SENTIMENT_OVERRIDE_THRESHOLD`: confiança mínima para manter o sentimento previsto em conflito com a emoção (padrão 0.65).

## 5. Modelos de IA
- Modelo de EMOÇÃO: classes `alegria`, `amor`, `surpresa`, `tristeza`, `raiva`, `medo`.
- Modelo de SENTIMENTO: classes `positivo` / `negativo`, derivadas das emoções.
- Pipeline: pré-processamento PT-BR (stopwords + stemming RSLP) → TF-IDF → Logistic Regression. Os dois modelos são treinados juntos e salvos no mesmo `.pkl`.
- Auto-treinamento da API: se `modelos_neuron_pt.pkl` não existir, a API treina a partir do CSV e salva o arquivo na raiz antes de subir.

## 6. Endpoints da API (Flask)
- `GET /health` → `{"status": "ok"}`.
- `POST /classificar`
  ```json
  {"texto": "Me sinto motivado e animado com o projeto"}
  ```
  Ou lista:
  ```json
  {"textos": ["estou animado", "estou com medo do prazo"]}
  ```
  Retorna emoção, sentimento e probabilidades por classe.
- `POST /conversas/analisar` (alias `/api/v1/analises-emocionais`)
  ```json
  {"mensagens": [{"timestamp": "2024-06-01T10:00:00", "texto": "estou feliz"}]}
  ```
  Retorna mensagens classificadas e resumo agregado por dia/total (útil para painéis anonimizados).

## 7. Testes automatizados
- Rodar todos:
  ```bash
  pytest
  ```
- Cobrem: pré-processamento, pipeline de treino (gera vetorizador, modelos e métricas) e smoke tests da API Flask (health, classificar texto único/lista e analisar conversas).

## 8. Deploy no Render
1. Faça push para GitHub/GitLab.
2. No Render: **New → Blueprint** e selecione o repositório.  
3. `render.yaml` define build (instala dependências + gera `.pkl` via `treinar_modelos.py`) e start (`gunicorn --bind 0.0.0.0:$PORT api_flask:app`).  
4. Teste `GET /health` no domínio gerado.

## 9. Privacidade, ética e uso corporativo
- Privacidade por design: consentimento explícito, opt-in/out e exclusão de dados pelo colaborador.
- Anonimização: relatórios apenas com grupos `k ≥ 3`; gestores/RH nunca veem dados individuais.
- Segurança: criptografia em repouso (AES-256) e em trânsito (TLS) no ambiente final; logs de consentimento auditáveis; retenção com anonimização após 12 meses.
- Integração corporativa: pode ser consumido por apps internos, painéis e notificações (Slack/Teams/Notion) sem expor dados pessoais.

## 10. Itens a não incluir no .zip de entrega
- Consulte `DEV_NOTES.md` para a lista detalhada (ex.: .venv, __pycache__, logs, arquivos temporários de IDE).  
- Mantenha os artefatos obrigatórios listados na seção 3.
