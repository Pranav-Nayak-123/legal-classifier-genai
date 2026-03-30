import re
from typing import Dict, List


CLAUSE_PATTERNS = {
    "Termination": [r"\btermination\b", r"\bterminate\b", r"\bbreach\b"],
    "Confidentiality": [r"\bconfidential\b", r"\bnon[- ]disclosure\b", r"\bnda\b"],
    "Liability Cap": [r"\bliability\b", r"\blimitation of liability\b", r"\bcap\b"],
    "Indemnity": [r"\bindemnif(y|ication)\b", r"\bhold harmless\b"],
    "Arbitration / Dispute": [r"\barbitration\b", r"\bdispute resolution\b", r"\bjurisdiction\b"],
    "Payment Terms": [r"\bpayment\b", r"\bfees?\b", r"\binvoice\b", r"\bconsideration\b"],
}


def _find_snippets(text: str, pattern: str, max_hits: int = 1) -> List[str]:
    snippets = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 80)
        snippet = " ".join(text[start:end].split())
        snippets.append(snippet)
        if len(snippets) >= max_hits:
            break
    return snippets


def extract_clauses(text: str) -> List[Dict]:
    results: List[Dict] = []
    clean = str(text or "")
    for clause, patterns in CLAUSE_PATTERNS.items():
        found = False
        snippets: List[str] = []
        for p in patterns:
            s = _find_snippets(clean, p, max_hits=1)
            if s:
                found = True
                snippets.extend(s)
                break
        results.append(
            {
                "name": clause,
                "found": found,
                "snippet": snippets[0] if snippets else "",
            }
        )
    return results


def build_summary(text: str, predicted_label: str = "") -> Dict:
    clauses = extract_clauses(text)
    found = [c["name"] for c in clauses if c["found"]]
    missing = [c["name"] for c in clauses if not c["found"]]

    risk_points = 0
    if "Liability Cap" in missing:
        risk_points += 2
    if "Termination" in missing:
        risk_points += 2
    if "Arbitration / Dispute" in missing:
        risk_points += 1
    if "Confidentiality" in missing:
        risk_points += 1

    if risk_points >= 4:
        risk_level = "High"
    elif risk_points >= 2:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    next_action = (
        "Review missing high-impact clauses (Termination, Liability, Dispute) before sign-off."
        if risk_level in ("High", "Medium")
        else "Proceed to detailed legal review and negotiation notes."
    )

    return {
        "document_type": predicted_label or "Unclassified",
        "key_issues": found[:4] if found else ["General legal obligations"],
        "risk_level": risk_level,
        "next_action": next_action,
        "clauses": clauses,
    }

