import json, traceback
from pathlib import Path
import requests
from gradio_client import Client, handle_file

OUT = Path("crossmodal_audit/api_smoke_outputs")
OUT.mkdir(parents=True, exist_ok=True)
IMAGE_URL = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/train/png/10197.png"
QUESTION = "What's the rightmost value dark brown graph?"
PROMPT = f"Question: {QUESTION}\nAnswer the question using only the evidence provided. Return only the short answer, without explanation."

img = OUT / "10197.png"
if not img.exists():
    r = requests.get(IMAGE_URL, timeout=60)
    r.raise_for_status()
    img.write_bytes(r.content)

SPACES = {
    "qwen": "developer0hye/Qwen2.5-VL-7B-Instruct",
    "internvl": "bert-ka/InternVL-8b-Demo",
}

def run_space(key, space):
    rec = {"space": space}
    try:
        client = Client(space, verbose=False)
        api = client.view_api(return_format="dict")
        rec["api"] = api
        # Named endpoint discovery.
        names = []
        if isinstance(api, dict):
            for container_key in ("named_endpoints", "unnamed_endpoints"):
                block = api.get(container_key, {})
                if isinstance(block, dict):
                    names.extend(block.keys())
        preferred = []
        if key == "qwen":
            preferred = ["/qwen_vl_inference", "/predict"]
        else:
            preferred = ["/internvl_inference", "/predict"]
        candidates = []
        for n in preferred + names:
            if n and n not in candidates:
                candidates.append(n)

        errors = {}
        for api_name in candidates:
            try:
                result = client.predict(
                    handle_file(str(img.resolve())),
                    PROMPT,
                    api_name=api_name,
                )
                rec.update({"status":"ok","api_name":api_name,"output":result})
                return rec
            except Exception as e:
                errors[api_name] = repr(e)
        rec.update({"status":"error","errors":errors})
        return rec
    except Exception as e:
        rec.update({"status":"error","error":repr(e),"trace":traceback.format_exc()})
        return rec

results = {k: run_space(k, v) for k, v in SPACES.items()}
(OUT / "api_smoke_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
print(json.dumps(results, indent=2, default=str))
