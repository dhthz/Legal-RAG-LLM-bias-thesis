import argparse
import base64
import html
import os
import re
import shutil
import subprocess
import time
from pathlib import Path, PureWindowsPath

# report.md -> one self-contained HTML page (charts embedded) -> PDF, printed by a Chromium-based browser
# (Chrome, Chromium or Edge) in headless mode. Without a browser the HTML page is kept, to print from any browser.
BROWSERS = ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome", "msedge", "microsoft-edge",
            "microsoft-edge-stable"]
BROWSER_PATHS = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                 "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                 "/Applications/Chromium.app/Contents/MacOS/Chromium",
                 r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                 r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                 r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",  # WSL: the Windows browser
                 "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe",
                 "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"]
STYLE = """
  @page { size: A4; margin: 16mm 15mm; }
  body { font-family: "Segoe UI", Calibri, Arial, sans-serif; color: #1c1d20; font-size: 10.5pt; line-height: 1.45; margin: 0; }
  h1 { font-family: Georgia, serif; font-size: 20pt; margin: 0 0 3mm; border-bottom: 2px solid #7a3b2e; padding-bottom: 2mm; }
  h2 { font-size: 13pt; color: #7a3b2e; margin: 6mm 0 2mm; }
  h1 + h1 { border: 0; }
  p { margin: 0 0 2.5mm; }
  table { border-collapse: collapse; width: 100%; margin: 1mm 0 3mm; font-size: 9.5pt; font-variant-numeric: tabular-nums; break-inside: avoid; }
  th { background: #f3f1ec; text-align: left; padding: 1.4mm 2mm; }
  td { border-top: 1px solid #e3e0d8; padding: 1.2mm 2mm; }
  img { max-width: 100%; max-height: 120mm; display: block; margin: 2mm auto 4mm; break-inside: avoid; }
"""


def inline(text):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text, quote=False))


def image(alt, path, base):
    full = path if os.path.isabs(path) else os.path.join(base, path)
    if not os.path.exists(full):
        return f"<p>[chart missing: {html.escape(path)}]</p>"
    with open(full, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], [c for c in cells[1:] if not all(set(x) <= set("-: ") for x in c)]
    out = ["<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>"]
    out += ["<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body]
    return "".join(out) + "</table>"


# The report's own markdown: headings, paragraphs, bold, tables and images; nothing else is needed
def to_html(markdown, base):
    blocks, lines, title = [], markdown.splitlines(), "Bias audit report"
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            blocks.append(table(lines[i:j]))
            i = j
            continue
        m = re.match(r"!\[(.*?)\]\((.+?)\)$", line.strip())
        if m:
            blocks.append(image(m.group(1), m.group(2), base))
        elif line.startswith("## "):
            blocks.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            blocks.append(f"<h1>{inline(line[2:])}</h1>")
            title = title if blocks[:-1] else line[2:]
        elif line.strip():
            blocks.append(f"<p>{inline(line)}</p>")
        i += 1
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f"<style>{STYLE}</style></head><body>\n" + "\n".join(blocks) + "\n</body></html>\n")


def find_browser():
    for name in BROWSERS:
        if shutil.which(name):
            return shutil.which(name)
    return next((p for p in BROWSER_PATHS if os.path.exists(p)), None)


# A Windows browser run from WSL needs Windows paths (\\wsl.localhost\... for files inside Linux)
def browser_path(path, browser):
    path = os.path.abspath(path)
    if browser.startswith("/mnt/") and browser.endswith(".exe"):
        return subprocess.run(["wslpath", "-w", path], capture_output=True, text=True, check=True).stdout.strip()
    return path


def print_pdf(browser, html_path, pdf_path):
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    page, out = browser_path(html_path, browser), browser_path(pdf_path, browser)
    uri = PureWindowsPath(page).as_uri() if "\\" in page else Path(page).as_uri()
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", uri],
                   capture_output=True, timeout=120)
    for _ in range(30):
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            return True
        time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description="Export the audit report (with its charts) as one PDF.")
    parser.add_argument("--report", default="report.md", help="report.md from report.py (charts linked with --figures)")
    parser.add_argument("--out", default="report.pdf")
    parser.add_argument("--keep-html", action="store_true", help="Keep the HTML page next to the PDF")
    args = parser.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        page = to_html(f.read(), os.path.dirname(os.path.abspath(args.report)))
    html_path = os.path.splitext(args.out)[0] + ".html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(page)
    browser = find_browser()
    if not browser:
        print(f"No Chrome, Chromium or Edge found, so no PDF: open {html_path} in a browser and print it to PDF.")
        return
    if not print_pdf(browser, html_path, args.out):
        print(f"The browser did not write the PDF; {html_path} is kept, print it to PDF from a browser.")
        return
    if not args.keep_html:
        os.remove(html_path)
    print(f"Saved: {args.out}" + (f" and {html_path}" if args.keep_html else ""))


if __name__ == "__main__":
    main()
