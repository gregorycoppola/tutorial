import re
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

BIB_FILE = Path(__file__).parent / "refs.bib"
OUT_FILE = Path(__file__).parent / "citation-check.md"
CACHE_FILE = Path(__file__).parent / "citation-cache.json"

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_URL = "http://export.arxiv.org/api/query"

TRUSTED_KEYS = {
    "pearl1988probabilistic",
    "good1950probability",
    "vaswani2017attention",
    "elhage2021mathematical",
    "elhage2022superposition",
    "olsson2022induction",
    "scarselli2009gnn",
    "gilmer2017mpnn",
    "yoon2019gnn",
    "joshi2020gnn",
    "geva2021ffn",
    "meng2022rome",
    "wang2022ioi",
    "yedidia2003bp",
    "perez2021turing",
}


def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def parse_bib(bib_text):
    entries = []
    pattern = re.compile(r"@(\w+)\{([^,]+),(.+?)(?=\n@|\Z)", re.DOTALL)
    field_pattern = re.compile(r"(\w+)\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
    for m in pattern.finditer(bib_text):
        kind = m.group(1).lower()
        key = m.group(2).strip()
        body = m.group(3)
        fields = {}
        for fm in field_pattern.finditer(body):
            fields[fm.group(1).lower()] = fm.group(2).strip()
        entries.append({"key": key, "kind": kind, "fields": fields})
    return entries


def clean(s):
    s = re.sub(r"\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    return s.strip()


def search_arxiv(arxiv_id):
    params = urllib.parse.urlencode({"id_list": arxiv_id.strip()})
    url = f"{ARXIV_URL}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "citation-checker/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
        if "<entry>" in body:
            title_m = re.search(r"<title>(.*?)</title>", body, re.DOTALL)
            if title_m and "Error" not in title_m.group(1):
                return "verified", title_m.group(1).strip()
        return "not_found", ""
    except Exception as e:
        return f"error:{e}", ""


def search_semantic_scholar(title):
    params = urllib.parse.urlencode({
        "query": clean(title),
        "fields": "title,externalIds",
        "limit": 3,
    })
    url = f"{SEMANTIC_SCHOLAR_URL}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "citation-checker/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        for paper in data.get("data", []):
            if similar(paper.get("title", "").lower(), clean(title).lower()):
                return "verified", paper.get("externalIds", {}).get("ArXiv", "")
        return "uncertain", ""
    except urllib.error.HTTPError as e:
        return f"error:HTTP{e.code}", ""
    except Exception as e:
        return f"error:{e}", ""


def similar(a, b):
    stop = {"a", "an", "the", "of", "in", "on", "for", "and", "to", "is", "are", "with"}
    aw = set(re.findall(r"\w+", a)) - stop
    bw = set(re.findall(r"\w+", b)) - stop
    if not aw or not bw:
        return False
    return len(aw & bw) / max(len(aw), len(bw)) >= 0.5


def extract_arxiv_id(fields):
    for f in ["journal", "note", "eprint", "url"]:
        m = re.search(r"(\d{4}\.\d{4,5})", fields.get(f, ""))
        if m:
            return m.group(1)
    return None


def check_entry(entry, cache):
    key = entry["key"]
    fields = entry["fields"]
    title = fields.get("title", "")

    if key in TRUSTED_KEYS or entry["kind"] == "book":
        return key, "trusted", "", clean(title), False

    if key in cache:
        r = cache[key]
        return key, r["status"], r["arxiv_id"], r["title"], False

    arxiv_id = extract_arxiv_id(fields)
    if arxiv_id:
        status, found = search_arxiv(arxiv_id)
        time.sleep(1.0)
        return key, status, arxiv_id, found or clean(title), True

    if title:
        status, arxiv_found = search_semantic_scholar(title)
        time.sleep(6.0)
        return key, status, arxiv_found, clean(title), True

    return key, "no_title", "", "", False


def write_report(results):
    icons = {"verified": "✓", "trusted": "✓", "uncertain": "⚠",
             "not_found": "✗", "manual": "📖"}
    lines = [
        "# Citation verification report\n\n",
        "| Key | Status | arXiv | Title |\n",
        "|-----|--------|-------|-------|\n",
    ]
    for key, status, arxiv_id, title in results:
        icon = icons.get(status, "?")
        lines.append(f"| `{key}` | {icon} {status} | {arxiv_id} | {title[:80]} |\n")

    needs_review = [r for r in results if r[1] not in ("verified", "trusted")]
    if needs_review:
        lines.append("\n## Needs manual review\n\n")
        for key, status, arxiv_id, title in needs_review:
            lines.append(f"- `{key}` ({status}): {title}\n")
    else:
        lines.append("\nAll citations verified or trusted.\n")

    OUT_FILE.write_text("".join(lines))


def main():
    bib_text = BIB_FILE.read_text()
    entries = parse_bib(bib_text)
    cache = load_cache()
    print(f"Found {len(entries)} entries. Cache has {len(cache)} saved results.\n")

    results = []
    for entry in entries:
        key = entry["key"]
        print(f"  checking {key}...", end=" ", flush=True)
        key, status, arxiv_id, title, did_fetch = check_entry(entry, cache)
        print(status + (" (cached)" if not did_fetch and key not in TRUSTED_KEYS else ""))
        results.append((key, status, arxiv_id, title))

        if did_fetch and status not in ("no_title",):
            cache[key] = {"status": status, "arxiv_id": arxiv_id, "title": title}
            save_cache(cache)

    write_report(results)
    verified = sum(1 for r in results if r[1] in ("verified", "trusted"))
    print(f"\n{verified}/{len(results)} verified or trusted.")
    print(f"Report written to citation-check.md")


if __name__ == "__main__":
    main()