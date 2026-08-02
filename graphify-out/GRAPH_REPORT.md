# Graph Report - .  (2026-08-02)

## Corpus Check
- Corpus is ~18,268 words - fits in a single context window. You may not need a graph.

## Summary
- 278 nodes · 635 edges · 14 communities (13 shown, 1 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.75)
- Token cost: 133,279 input · 0 output

## Community Hubs (Navigation)
- Pipeline Entry & Schema
- Data Cleaning Pipeline
- Margin Allocation Strategies
- Streamlit Dashboard Views
- Synthetic Data Generation
- Routing & Dead-Head Detection
- Standalone Tool & Docs
- Capability Detection & Tests
- Margin Dashboard Rendering (JS)
- Routing Tool JS Logic
- Confounder Analysis Checks
- Graph Tooltip UI

## God Nodes (most connected - your core abstractions)
1. `validate()` - 20 edges
2. `detect_empty_legs()` - 17 edges
3. `clean()` - 14 edges
4. `clean_trips()` - 13 edges
5. `src/routing.py` - 13 edges
6. `normalise_dates()` - 12 edges
7. `detect_capabilities()` - 12 edges
8. `compute_margin()` - 12 edges
9. `run_routing()` - 12 edges
10. `summarize_vehicle_km()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `detect_empty_legs()` --shares_data_with--> `data/reference/distances.csv (placeholder matrix)`  [EXTRACTED]
  src/routing.py → data/reference/README.md
- `utilisationSummary()` --implements--> `utilisation_summary()`  [EXTRACTED]
  dashboard/standalone.html → src/routing.py
- `priceEmptyKm()` --implements--> `price_empty_km()`  [EXTRACTED]
  dashboard/standalone.html → src/routing.py
- `STRATEGY_EXPLANATIONS` --references--> `normalise_dates()`  [EXTRACTED]
  dashboard/index.html → src/clean.py
- `clean()` --shares_data_with--> `data/processed/decision_log.csv`  [EXTRACTED]
  src/clean.py → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Margin allocation strategy pattern (margin.STRATEGIES)** — src_margin_direct_cost_only, src_margin_proportional_by_revenue, src_margin_per_unit_by_quantity, src_margin_compute_margin [EXTRACTED 1.00]
- **Routing/empty-running analysis pipeline (src/routing.py)** — src_routing_detect_empty_legs, src_routing_summarize_vehicle_km, src_routing_utilisation, src_routing_utilisation_summary, src_routing_price_empty_km [EXTRACTED 1.00]
- **JavaScript port of the routing pipeline in dashboard/standalone.html** — dashboard_standalone_detectemptylegs, dashboard_standalone_summarizevehiclekm, dashboard_standalone_utilisation, dashboard_standalone_utilisationsummary, dashboard_standalone_priceemptykm [EXTRACTED 1.00]

## Communities (14 total, 1 thin omitted)

### Community 0 - "Pipeline Entry & Schema"
Cohesion: 0.10
Nodes (27): Path, main(), _parse_args(), Namespace, Pipeline entry point for both the cost/margin and routing pipelines.…, Inspect auto_path's columns with src/detect.py, print what it found and what it…, run(), run_auto() (+19 more)

### Community 1 - "Data Cleaning Pipeline"
Cohesion: 0.13
Nodes (33): cleanTrips(), normaliseId(), parseFlexibleDate(), data/processed/decision_log.csv, DecisionLog, clean(), clean_trips(), decision_log_to_frame() (+25 more)

### Community 2 - "Margin Allocation Strategies"
Cohesion: 0.15
Nodes (29): Allocator, DATA (precomputed strategy snapshot), Series, _allocate_overhead(), compute_margin(), direct_cost_only(), _finalize(), src/margin.py (+21 more)

### Community 3 - "Streamlit Dashboard Views"
Cohesion: 0.15
Nodes (28): cache_data, load_cost_data(), load_distances(), load_ledger(), load_overhead(), load_trips(), main(), DataFrame (+20 more)

### Community 4 - "Synthetic Data Generation"
Cohesion: 0.17
Nodes (28): date, fixture, Random, _build_routes(), generate_ledger(), generate_overhead(), generate_trips(), main() (+20 more)

### Community 5 - "Routing & Dead-Head Detection"
Cohesion: 0.18
Nodes (27): detect_empty_legs(), _distance_map(), price_empty_km(), DataFrame, Empty-running (dead-head) detection and vehicle utilisation. Core idea: a…, Per vehicle, per trip: idle days between this trip's date and the next trip's…, Per vehicle: total and average idle days, and how many gaps that's computed…, Add an empty_cost column (rupees) to a copy of empty_df, converting empty_km… (+19 more)

### Community 6 - "Standalone Tool & Docs"
Cohesion: 0.13
Nodes (23): dashboard/app.py, COLUMN_ALIASES (JS), detectCapabilities(), mapAliases(), dashboard/standalone.html (Freight Empty-Running Tool), data/reference/distances.csv (placeholder matrix), Make targets (synth, synth-trips, clean-data, clean-trips, test, dashboard), notebooks/01_explore.ipynb (+15 more)

### Community 7 - "Capability Detection & Tests"
Cohesion: 0.21
Nodes (13): detect_capabilities(), _map_aliases(), DataFrame, Schema detection: decide which pipeline(s) a given file's columns support -…, Returns (mapping, unmapped): mapping is original header -> canonical name for…, Inspect df's columns and decide which analysis pipeline(s) are runnable.…, test_column_aliases_are_mapped_to_canonical_names(), test_detector_never_fabricates_a_missing_column() (+5 more)

### Community 8 - "Margin Dashboard Rendering (JS)"
Cohesion: 0.18
Nodes (10): dashboard/index.html (interactive margin dashboard), renderBarList(), renderConcentrationTable(), renderCustomerSection(), renderKpis(), renderMonthTable(), renderRouteSection(), renderStrategyDependent() (+2 more)

### Community 9 - "Routing Tool JS Logic"
Cohesion: 0.26
Nodes (10): buildDistanceMap(), detectEmptyLegs(), distanceKey(), groupByVehicleSortedByDate(), loadDistanceRecords(), priceEmptyKm(), summarizeVehicleKm(), tryComputeAndRender() (+2 more)

### Community 10 - "Confounder Analysis Checks"
Cohesion: 0.26
Nodes (11): accrual_vs_cash_view(), mix_shift_decomposition(), src/confounder.py, period_over_period_excluding(), DataFrame, Helpers for trying to kill a finding, not confirm one. Each function here takes…, Split the change in total margin between period_a and period_b into a…, Compare a period's totals booked by txn_date (accrual - when the sale happened)… (+3 more)

## Knowledge Gaps
- **5 isolated node(s):** `LEDGER_SCHEMA`, `OVERHEAD_SCHEMA`, `TRIPS_SCHEMA`, `src/synth.py`, `data/processed/decision_log.csv`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `validate()` connect `Pipeline Entry & Schema` to `Data Cleaning Pipeline`, `Streamlit Dashboard Views`, `Routing & Dead-Head Detection`, `Standalone Tool & Docs`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Why does `normalise_dates()` connect `Data Cleaning Pipeline` to `Margin Dashboard Rendering (JS)`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `STRATEGY_EXPLANATIONS` connect `Margin Dashboard Rendering (JS)` to `Data Cleaning Pipeline`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **What connects `LEDGER_SCHEMA`, `OVERHEAD_SCHEMA`, `TRIPS_SCHEMA` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Pipeline Entry & Schema` be split into smaller, more focused modules?**
  _Cohesion score 0.1006006006006006 - nodes in this community are weakly interconnected._
- **Should `Data Cleaning Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._
- **Should `Standalone Tool & Docs` be split into smaller, more focused modules?**
  _Cohesion score 0.12681159420289856 - nodes in this community are weakly interconnected._