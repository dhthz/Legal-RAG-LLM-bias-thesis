import json
import re
from typing import Dict, List


# Parses the trailing ```json {cited_case_ids, case_articles}``` block the
# prompt requests, validated against the retrieved case IDs. Falls back to
# substring matching (case title/ID in the response text) if the model
# doesn't follow the format.
def parse_structured_response(cases: List[Dict], response_text: str) -> Dict:
    valid_case_ids = {case['case_id'] for case in cases}

    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            cited = [cid for cid in parsed.get("cited_case_ids", []) if cid in valid_case_ids]
            case_articles = {
                cid: articles for cid, articles in parsed.get("case_articles", {}).items()
                if cid in valid_case_ids
            }
            result = {
                "cited_case_ids": cited,
                "case_articles": case_articles,
                "json_parse_ok": True,
            }
            if "predicted_articles" in parsed:
                predicted = parsed["predicted_articles"]
                result["predicted_articles"] = [str(a) for a in predicted] if isinstance(predicted, list) else None
            return result
        except (json.JSONDecodeError, AttributeError):
            pass

    cited = [case['case_id'] for case in cases
             if case['case_id'] in response_text or case['title'] in response_text]
    return {"cited_case_ids": cited, "case_articles": {}, "json_parse_ok": False}
