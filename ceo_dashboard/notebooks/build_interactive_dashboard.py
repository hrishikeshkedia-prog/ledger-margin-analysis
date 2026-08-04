"""
Builds the Stage 8 interactive dashboard:
ceo_dashboard/notebooks/interactive_dashboard.html

Bakes the bundled synthetic fashion CSVs (data/synthetic/orders.csv,
marketing_spend.csv, inventory_snapshots.csv) into
interactive_dashboard_template.html as the page's default dataset, so it
opens looking identical to the static Stage 7 export -- but every chart in
it is computed live in the browser from that embedded CSV text, and a user
can replace it with their own CSV entirely client-side.

This does NOT touch the Stage 1-7 Python pipeline (data_generator.py,
core.py, fashion.py, dashboard.py, export_html.py, ceo_dashboard.ipynb,
ceo_dashboard.html) -- it is an additional, alongside deliverable that
reimplements the presentation layer's math in JavaScript for interactivity.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CEO_DASHBOARD_ROOT = os.path.dirname(_HERE)

TEMPLATE_PATH = os.path.join(_HERE, "interactive_dashboard_template.html")
OUTPUT_PATH = os.path.join(_HERE, "interactive_dashboard.html")
DATA_DIR = os.path.join(_CEO_DASHBOARD_ROOT, "data", "synthetic")


def main():
    with open(os.path.join(DATA_DIR, "orders.csv"), encoding="utf-8") as f:
        orders_csv = f.read()
    with open(os.path.join(DATA_DIR, "marketing_spend.csv"), encoding="utf-8") as f:
        marketing_csv = f.read()
    with open(os.path.join(DATA_DIR, "inventory_snapshots.csv"), encoding="utf-8") as f:
        inventory_csv = f.read()

    default_data = {"orders": orders_csv, "marketing": marketing_csv, "inventory": inventory_csv}

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()

    # json.dumps produces a string that is simultaneously valid JSON (for the
    # <script type="application/json"> the page JSON.parse()s) and safe to
    # drop into HTML as-is, since </script>-breaking sequences can't appear
    # inside CSV text from this project's own generator.
    html = template.replace("__DEFAULT_DATA_JSON__", json.dumps(default_data))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"Wrote {OUTPUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
