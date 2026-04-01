# Legal Classifier GenAI

An end-to-end legal AI application for:

- Legal document classification (Deep Learning)
- Document Q&A chatbot with live streaming responses
- Clause extraction and risk-focused summaries
- Workspace save/load style output and PDF report export

## Highlights

- Transformer classifier (Legal-BERT) with training pipeline
- Faster local chatbot mode with role templates:
  - `Law Student`
  - `Paralegal`
  - `Counsel`
- Multi-turn chatbot memory per active document
- Ask-with-citations workflow (supporting snippets + scores)
- Document intelligence cards:
  - Document summary
  - Clause extractor
  - Risk level + next action
- Exportable PDF report and workspace JSON snapshots

## Tech Stack

- Python
- Flask
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- HTML/CSS (server-rendered UI)

## Project Structure

```text
legal_classifier/
  app.py
  requirements.txt
  README.md
  data/
    raw/
      sample_legal_docs.csv
      legal_scotus.csv
  docs/
    screenshots/
      README.md
  src/
    legal_classifier/
      analysis.py
      config.py
      data.py
      download_dataset.py
      extract.py
      model.py
      predict.py
      qa.py
      train.py
      train_transformer.py
      workspace.py
  static/
    styles.css
  templates/
    index.html
```

## Quick Start

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run app:

```bash
python app.py
```

Open:

`http://127.0.0.1:5000`

## Model Training

Recommended dataset download:

```bash
python -m src.legal_classifier.download_dataset --dataset lex_glue --subset scotus --output data/raw/legal_scotus.csv
```

Train transformer classifier:

```bash
python -m src.legal_classifier.train_transformer --data_path data/raw/legal_scotus.csv --model_name nlpaueb/legal-bert-base-uncased --epochs 4 --batch_size 8
```

Baseline training (BiLSTM):

```bash
python -m src.legal_classifier.train
```

## Outputs and Artifacts

- `artifacts/transformer_model/`
- `artifacts/model_type.json`
- `artifacts/classification_report.json`
- `artifacts/training_history.json`
- `workspace_cases/case_*.json`
- exported report: `legal_report.pdf`

## Product Workflow

1. Upload/Paste legal text
2. Classify document type
3. Review summary + clause extraction + risk
4. Ask follow-up legal questions (streaming + citations)
5. Save workspace snapshot
6. Export final report as PDF

## Roadmap

- Add user authentication + case-level access control
- Add multi-document comparison and redline insights
- Add confidence calibration and escalation workflows
- Add cloud deployment with async queue workers

## Disclaimer

This tool is for educational and productivity use. It is **not legal advice**.
