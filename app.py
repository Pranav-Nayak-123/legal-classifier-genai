from flask import Flask, render_template, request

from src.legal_classifier.extract import extract_text_from_upload
from src.legal_classifier.predict import Predictor
from src.legal_classifier.qa import answer_question

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
predictor = None
load_error = None
model_type = "unknown"
last_document_text = ""
last_uploaded_filename = ""

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
    global last_document_text, last_uploaded_filename

    result = None
    probabilities = []
    qa_result = None
    qa_sections = None
    question_text = ""
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
            if not last_document_text:
                qa_result = {"error": "Classify or upload a document first, then ask questions."}
            else:
                qa_result = answer_question(last_document_text, question_text)
                if not qa_result.get("error"):
                    qa_sections = _parse_qa_sections(qa_result.get("answer", ""))
            input_text = last_document_text

    return render_template(
        "index.html",
        result=result,
        probabilities=probabilities,
        qa_result=qa_result,
        qa_sections=qa_sections,
        question_text=question_text,
        input_text=input_text,
        load_error=load_error,
        model_type=model_type,
        uploaded_filename=uploaded_filename,
    )


if __name__ == "__main__":
    app.run(debug=True)
