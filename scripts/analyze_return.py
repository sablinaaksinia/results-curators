import csv
import json
import urllib.request
import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "docs" / "Возвраты.csv"
PROMPT_PATH = BASE_DIR / "prompts" / "sentiment-prompt.md"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llm-tools-32k:latest"

DELIMITER = ";"
NAME_COL = "Student"
REASON_COL = "Prichina"
RESULT_COL = "action"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def ask_ollama(prompt_text: str) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt_text,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "").strip()


def build_prompt(template: str, name: str, reason: str) -> str:
    return template.replace("{Student}", name).replace("{Prichina}", reason)


def process():
    prompt_template = load_prompt()

    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=DELIMITER)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if RESULT_COL not in fieldnames:
        fieldnames = list(fieldnames) + [RESULT_COL]

    total = len(rows)
    for i, row in enumerate(rows, 1):
        name = row.get(NAME_COL, "").strip()
        reason = row.get(REASON_COL, "").strip()

        if not reason:
            row[RESULT_COL] = ""
            print(f"[{i}/{total}] {name} — причина пустая, пропускаем")
            continue

        prompt_text = build_prompt(prompt_template, name, reason)
        print(f"[{i}/{total}] {name} — отправляем в модель...", end=" ", flush=True)

        try:
            answer = ask_ollama(prompt_text)
            row[RESULT_COL] = answer
            print(answer)
        except Exception as e:
            row[RESULT_COL] = "ошибка"
            print(f"ОШИБКА: {e}", file=sys.stderr)

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=DELIMITER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nГотово. Обработано строк: {total}. Файл сохранён: {CSV_PATH}")


if __name__ == "__main__":
    process()
