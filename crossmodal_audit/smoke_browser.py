import json, os, time, traceback
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright

OUT = Path("crossmodal_audit/smoke_outputs")
OUT.mkdir(parents=True, exist_ok=True)
IMAGE_URL = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/train/png/10197.png"
QUESTION = "What's the rightmost value dark brown graph?"
PROMPT = f"Question: {QUESTION}\nAnswer the question using only the evidence provided. Return only the short answer, without explanation."

img = OUT / "10197.png"
if not img.exists():
    r = requests.get(IMAGE_URL, timeout=60)
    r.raise_for_status()
    img.write_bytes(r.content)

results = {}

def save_debug(page, name):
    try:
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
    except Exception:
        pass
    try:
        (OUT / f"{name}.html").write_text(page.content(), encoding="utf-8")
    except Exception:
        pass

def smoke_qwen(browser):
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    try:
        page.goto("https://developer0hye-qwen2-5-vl-7b-instruct.hf.space/", wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(10000)
        save_debug(page, "qwen_loaded")
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files(str(img.resolve()))
        tb = page.get_by_label("Question")
        if tb.count() == 0:
            tb = page.locator("textarea").first
        tb.fill(PROMPT)
        btn = page.get_by_role("button", name="Submit")
        if btn.count() == 0:
            btn = page.locator("button").filter(has_text="Submit").first
        btn.click()
        page.wait_for_timeout(2000)
        # Gradio output textbox is usually a textarea/input with label Model Output.
        out = page.get_by_label("Model Output")
        if out.count() == 0:
            out = page.locator("textarea").last
        value = ""
        for _ in range(90):
            try:
                value = out.input_value().strip()
            except Exception:
                try:
                    value = out.inner_text().strip()
                except Exception:
                    value = ""
            if value:
                break
            page.wait_for_timeout(2000)
        save_debug(page, "qwen_done")
        return {"status":"ok" if value else "empty", "output":value, "url":page.url}
    except Exception as e:
        save_debug(page, "qwen_error")
        return {"status":"error","error":repr(e),"trace":traceback.format_exc(),"url":page.url}
    finally:
        page.close()

def smoke_internvl(browser):
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    urls = [
        "https://internvl.opengvlab.com/",
        "https://opengvlab-internvl.hf.space/",
        "https://developer0hye-internvl3-8b.hf.space/",
    ]
    last = None
    for url in urls:
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=120000)
            page.wait_for_timeout(10000)
            save_debug(page, "internvl_loaded")
            body = page.locator("body").inner_text(timeout=10000)
            # Prefer an exact InternVL3-8B selector if present.
            if "InternVL3-8B" in body:
                try:
                    page.get_by_text("InternVL3-8B", exact=True).first.click(timeout=5000)
                    page.wait_for_timeout(1500)
                except Exception:
                    pass
            fi = page.locator('input[type="file"]').first
            if fi.count() == 0:
                raise RuntimeError("No image file input found")
            fi.set_input_files(str(img.resolve()))
            # Multimodal textbox may render as textarea.
            ta = page.locator("textarea").first
            if ta.count() == 0:
                ta = page.locator('input[type="text"]').first
            if ta.count() == 0:
                raise RuntimeError("No prompt textbox found")
            ta.fill(PROMPT)
            # Try Send/Submit.
            btn = page.get_by_role("button", name="Send")
            if btn.count() == 0:
                btn = page.get_by_role("button", name="Submit")
            if btn.count() == 0:
                btn = page.locator("button").filter(has_text="Send").first
            if btn.count() == 0:
                raise RuntimeError("No send button found")
            btn.click()
            page.wait_for_timeout(3000)
            value = ""
            # Chatbot outputs generally appear as markdown blocks; poll visible text.
            for _ in range(90):
                txt = page.locator("body").inner_text()
                # keep tail for diagnosis; success if prompt is followed by substantially new text
                if len(txt) > len(body) + 20:
                    value = txt[-2000:].strip()
                    break
                page.wait_for_timeout(2000)
            save_debug(page, "internvl_done")
            return {"status":"ok" if value else "empty","output_tail":value,"url":page.url}
        except Exception as e:
            last = {"status":"error","error":repr(e),"trace":traceback.format_exc(),"url":page.url}
            save_debug(page, "internvl_error")
            continue
    page.close()
    return last or {"status":"error","error":"no URL attempted"}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    results["qwen"] = smoke_qwen(browser)
    results["internvl"] = smoke_internvl(browser)
    browser.close()

(OUT / "smoke_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
