import io
import json
from datetime import datetime

from flask import Flask, Response, render_template, request, stream_with_context, send_file

from src.legal_classifier.analysis import build_summary
from src.legal_classifier.extract import extract_text_from_upload
from src.legal_classifier.predict import Predictor
from src.legal_classifier.qa import answer_question, retrieve_evidence, stream_answer_question
from src.legal_classifier.workspace import save_workspace

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
predictor = None
load_error = None
model_type = "unknown"
last_document_text = ""
last_uploaded_filename = ""
last_prediction = {}
last_summary = {}
last_qa = {}
last_role = "counsel"

try:
    predictor = Predictor()
    model_type = predictor.model_type
except Exception as exc:
    load_error = str(exc)


def _parse_qa_sections(answer_text: str):
    if not answer_text:
        return None
    sections = {"issue": "", "analysis": "", "conclusion": ""}
    current = None
    for raw_line in str(answer_text).splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("issue:"):
            current = "issue"
            sections[current] = line.split(":", 1)[1].strip()
            continue
        if lower.startswith("analysis:"):
            current = "analysis"
            sections[current] = line.split(":", 1)[1].strip()
            continue
        if lower.startswith("conclusion:"):
            current = "conclusion"
            sections[current] = line.split(":", 1)[1].strip()
            continue
        if current and line:
            sections[current] = (sections[current] + "\n" + line).strip()
    if any(sections.values()):
        return sections
    return None


@app.route("/", methods=["GET", "POST"])
def index():
    global last_document_text, last_uploaded_filename, last_prediction, last_summary, last_qa, last_role

    result = None
    probabilities = []
    qa_result = None
    qa_sections = None
    question_text = ""
    role_template = last_role
    input_text = last_document_text
    uploaded_filename = last_uploaded_filename

    if request.method == "POST":
        action = request.form.get("action", "classify")

        if action == "classify":
            input_text = request.form.get("document_text", "").strip()
            file = request.files.get("document_file")

            if file and file.filename:
                uploaded_filename = file.filename
                try:
                    input_text = extract_text_from_upload(file.filename, file.read())
                except Exception as exc:
                    result = {"error": f"File extraction failed: {exc}"}

            if result and "error" in result:
                pass
            elif predictor is None:
                result = {"error": load_error or "Model not loaded."}
            elif not input_text:
                result = {"error": "Please enter text or upload a file (PDF, DOCX, TXT)."}
            else:
                last_document_text = input_text
                last_uploaded_filename = uploaded_filename
                label, confidence, predictions = predictor.predict(input_text)
                result = {"label": label, "confidence": round(confidence * 100, 2)}
                probabilities = [
                    {"label": p["label"], "confidence": round(p["confidence"] * 100, 2)}
                    for p in predictions
                ]
        elif action == "ask":
            question_text = request.form.get("question_text", "").strip()
            role_template = request.form.get("role_template", "counsel")
            last_role = role_template
            if not last_document_text:
                qa_result = {"error": "Classify or upload a document first, then ask questions."}
            else:
                qa_result = answer_question(
                    last_document_text,
                    question_text,
                    top_k=2,
                    role_template=role_template,
                )
                if not qa_result.get("error"):
                    qa_sections = _parse_qa_sections(qa_result.get("answer", ""))
                    last_qa = {
                        "question": question_text,
                        "answer": qa_result.get("answer", ""),
                        "evidence": qa_result.get("evidence", []),
                        "role_template": role_template,
                    }
            input_text = last_document_text
        elif action == "save_workspace":
            payload = {
                "saved_at": datetime.now().isoformat(),
                "uploaded_filename": last_uploaded_filename,
                "document_text": last_document_text,
                "prediction": last_prediction,
                "summary": last_summary,
                "qa": last_qa,
            }
            out = save_workspace(payload)
            qa_result = {"answer": f"Workspace saved to {out}"}

    if result and not result.get("error"):
        last_prediction = {"result": result, "probabilities": probabilities}
    if last_document_text:
        last_summary = build_summary(last_document_text, result["label"] if result and "label" in result else "")

    return render_template(
        "index.html",
        result=result,
        probabilities=probabilities,
        qa_result=qa_result,
        qa_sections=qa_sections,
        question_text=question_text,
        role_template=role_template,
        input_text=input_text,
        summary=last_summary,
        load_error=load_error,
        model_type=model_type,
        uploaded_filename=uploaded_filename,
    )


@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    global last_document_text, last_qa, last_role

    payload = request.get_json(silent=True) or {}
    question_text = str(payload.get("question_text", "")).strip()
    role_template = str(payload.get("role_template", last_role)).strip() or "counsel"
    last_role = role_template

    if not last_document_text:
        return ("Classify or upload a document first, then ask questions.", 400)
    if not question_text:
        return ("Please enter a question.", 400)
    evidence = retrieve_evidence(last_document_text, question_text, top_k=2)

    def generate():
        yield json.dumps({"type": "meta", "evidence": evidence}) + "\n"
        parts = []
        for chunk in stream_answer_question(
            last_document_text,
            question_text,
            top_k=2,
            role_template=role_template,
            precomputed_evidence=evidence,
        ):
            parts.append(chunk)
            yield json.dumps({"type": "token", "text": chunk}) + "\n"
        answer_text = "".join(parts).strip()
        last_qa = {
            "question": question_text,
            "answer": answer_text,
            "evidence": evidence,
            "role_template": role_template,
        }
        yield json.dumps({"type": "done"}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@app.route("/export_report", methods=["GET"])
def export_report():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        return ("reportlab not installed. Run: pip install reportlab", 500)

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    def line(txt: str, step: int = 16):
        nonlocal y
        for p in str(txt).split("\n"):
            c.drawString(40, y, p[:120])
            y -= step
            if y < 60:
                c.showPage()
                y = height - 50

    line("Legal Classifier Report")
    line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    line("")
    line(f"Uploaded file: {last_uploaded_filename or 'N/A'}")
    line("")
    if last_prediction:
        line("Prediction:")
        line(f"  Label: {last_prediction.get('result', {}).get('label', '')}")
        line(f"  Confidence: {last_prediction.get('result', {}).get('confidence', '')}%")
        line("")
    if last_summary:
        line("Summary:")
        line(f"  Document Type: {last_summary.get('document_type', '')}")
        line(f"  Risk Level: {last_summary.get('risk_level', '')}")
        line(f"  Next Action: {last_summary.get('next_action', '')}")
        line("  Key Issues: " + ", ".join(last_summary.get("key_issues", [])))
        line("")
        line("Clauses:")
        for clause in last_summary.get("clauses", []):
            state = "Found" if clause.get("found") else "Missing"
            line(f"  - {clause.get('name')}: {state}")
        line("")
    if last_qa:
        line("Latest Q&A:")
        line(f"  Question: {last_qa.get('question', '')}")
        line(f"  Role Template: {last_qa.get('role_template', '')}")
        line("  Answer:")
        line("  " + str(last_qa.get("answer", "")))

    c.save()
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="legal_report.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(debug=True)
