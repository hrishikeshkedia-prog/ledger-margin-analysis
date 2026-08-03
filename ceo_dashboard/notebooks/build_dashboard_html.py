"""
Builds the Stage 7 standalone presentation file:
ceo_dashboard/notebooks/ceo_dashboard.html

A single self-contained HTML file (charts embedded as base64 PNGs) that
opens offline in any browser -- no Python, no Colab, no live kernel. Every
number and chart in it comes from `src/export_html.py`, which itself only
calls existing core.py / fashion.py / alerts.py / forecast.py /
dashboard.py functions -- this script just orchestrates the write.

Rendering the embedded PNGs uses kaleido, which needs a real Chrome/
Chromium binary. If none is found on PATH, point BROWSER_PATH at one, e.g.:
    BROWSER_PATH=/opt/pw-browsers/chromium-1194/chrome-linux/chrome \\
        python notebooks/build_dashboard_html.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CEO_DASHBOARD_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _CEO_DASHBOARD_ROOT)

from src.export_html import write_dashboard_html  # noqa: E402

OUTPUT_PATH = os.path.join(_HERE, "ceo_dashboard.html")

if __name__ == "__main__":
    path = write_dashboard_html(OUTPUT_PATH, seed=42)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Wrote {path} ({size_mb:.2f} MB)")
