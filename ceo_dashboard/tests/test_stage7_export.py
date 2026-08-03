"""Stage 7 tests: standalone HTML presentation export.

Every number in the exported page must trace back to the same core.py /
fashion.py / alerts.py / forecast.py / dashboard.py functions used
throughout the notebook -- this module only lays results out as HTML.
"""

import os
import shutil

import pytest

from ceo_dashboard.src import export_html


# `fig.to_image()` (kaleido) needs a real Chrome/Chromium binary, which this
# sandboxed environment provides at a fixed path via BROWSER_PATH rather than
# a system-installed `chrome`/`chromium` on PATH. Set it if missing so the
# suite works out of the box here; skip gracefully anywhere no browser is
# discoverable at all, since image rendering is an environment concern, not
# a code-correctness one.
_FALLBACK_BROWSER = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
if not os.environ.get("BROWSER_PATH") and os.path.exists(_FALLBACK_BROWSER):
    os.environ["BROWSER_PATH"] = _FALLBACK_BROWSER

_HAS_BROWSER = bool(os.environ.get("BROWSER_PATH")) or shutil.which("chromium") or shutil.which("google-chrome")

pytestmark = pytest.mark.skipif(not _HAS_BROWSER, reason="no Chrome/Chromium available for kaleido PNG rendering")


@pytest.fixture(scope="module")
def export():
    return export_html.compute_export_data(seed=42)


# ---------------------------------------------------------------------------
# 1. compute_export_data traces to the known, previously-verified numbers
# ---------------------------------------------------------------------------

def test_findings_match_known_verified_numbers(export):
    f = export["findings"]
    # Same figures the user quoted from prior stages' verification output.
    assert f["n_loss_sku"] == 28
    assert f["n_total_sku"] == 150
    assert f["n_loss_channel"] == 1
    assert f["n_at_risk"] == 12
    assert f["n_cac_driven"] == 22
    assert 0.25 < f["returns_pct_of_cm"] < 0.35  # "~29%" quoted; verify it's in the right ballpark, not hand-tuned to match
    assert 0.15 < f["wape"] < 0.25  # "~19%"
    assert 0.70 < f["stockout_recall"] < 0.85  # "79%"
    assert 0.20 < f["stockout_precision"] < 0.35  # "27%"
    assert f["influencer_cac_excess"] > 0  # Influencer really does overspend peer channels' CAC


def test_seven_figures_present(export):
    expected = {"scorecard", "revenue_margin_trend", "returns_bridge", "margin_by_sku",
                "channel_view", "cohort_heatmap", "inventory_view"}
    assert set(export["figures"]) == expected


def test_alerts_and_inventory_tables_are_the_same_ones_stage5_stage6_produce(export):
    # entity_type/severity values must be exactly what alerts.build_alerts_table defines
    assert set(export["alerts_table"]["severity"].unique()) <= {"Loss-Making", "At-Risk"}
    assert set(export["inventory_action"]["inventory_flag"].unique()) <= {"Stockout Risk", "Overstock Risk", "Healthy"}


# ---------------------------------------------------------------------------
# 2. HTML assembly helpers
# ---------------------------------------------------------------------------

def test_money_cr_formatting():
    assert export_html._money_cr(1.83e7) == "₹1.83 Cr"
    assert export_html._money_cr(0) == "₹0.00 Cr"


def test_alerts_table_html_renders_one_row_per_alert(export):
    html = export_html._alerts_table_html(export["alerts_table"])
    # one <tr> for the header row, plus one per alert
    assert html.count("<tr>") == len(export["alerts_table"]) + 1
    assert "badge-critical" in html or "badge-warning" in html


# ---------------------------------------------------------------------------
# 3. Full page build -- self-contained, offline, no black-box numbers
# ---------------------------------------------------------------------------

def test_build_html_is_self_contained(export):
    html = export_html.build_html(export)
    assert "<!doctype html>" in html.lower()
    # every image must be an embedded base64 PNG, never a network reference
    assert "data:image/png;base64," in html
    assert "http://" not in html and "https://" not in html
    assert 'src="//' not in html
    # the synthetic-data disclaimer and honest-limitations footer are required content
    assert "SYNTHETIC" in html
    assert "Honest Limitations" in html
    assert "precision" in html.lower() and "recall" in html.lower()


def test_write_dashboard_html_writes_a_file(tmp_path):
    out = export_html.write_dashboard_html(str(tmp_path / "dashboard.html"), seed=42)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert len(content) > 500_000  # seven embedded PNG charts make this a large, single file
    assert "https://" not in content and "http://" not in content
