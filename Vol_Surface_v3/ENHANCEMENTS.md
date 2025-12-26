Volatility Surface v3 — Functional Enhancement Blueprint

Scope
- Raise the platform to professional grade across 13F parsing, Insider (Form 4) signals, and Options Volatility analytics, with commodity‑oriented extensions.

13F Institutional Holdings (Correctness + Analytics)
- Source selection: Use SEC modern endpoints; prefer authoritative INFORMATION TABLE documents; support 13F‑HR/A (amendments) with de‑duplication; exclude 13F‑NT.
- XML robustness: Namespaced XPath parsing with lxml recover; fallback to regex only if necessary; filing‑level QA (AUM vs sum of values; report period checks).
- Symbology & corporate actions: CUSIP→FIGI→Ticker mapping; point‑in‑time corporate action handling; currency flags.
- Holdings time series: Position deltas normalized by prior size and AUM; classify NEW/INCREASE/DECREASE/EXIT; filter noise.
- Consensus, crowding, conviction: Weight by fund quality; penalize crowded names; conviction = concentration + holding age + value‑weighted deltas.
- Fund quality: Shadow‑fund backtests with 45‑day reporting lag; sector‑style normalization.

Insider (Form 4) — History & Signals
- Parser: Clean transaction codes (P/S/A/D/M), derivative vs non‑derivative, direct vs indirect, 10b5‑1; deduplicate amendments; cluster by time.
- Abnormal returns: 1m/3m abnormal returns vs factors; exclude mechanical events (vesting/gifts) from alpha pool.
- Cohort grading: Insider “grade” by stock and role (CEO/CFO/Chair); size and cluster density scaling.
- Integration: Combine with 13F consensus; divergence detection (insiders vs institutions); 10‑year point‑in‑time backtests and leaderboards.

Volatility Surface — Institutional Quality
- Data hygiene: NBBO filtering; size‑weighted “micro‑mid”; dividend/borrow inputs; forward from surface.
- Surface construction: Joint SVI calibration across expiries with static‑arbitrage constraints (convex in strike; monotone in time); forward variance and risk‑neutral density extraction; QC metrics.
- Model calibration: Heston/Merton fits validated via QuantLib; parameter stability tracking.
- Strategy pricing: Calendars/diagonals/butterflies/risk‑reversals priced under the calibrated surface; transaction costs and liquidity screens.
- Regime routing: Short‑vol carry in calm regimes; long‑vol convexity in stress; vega/tenor laddering and CVaR caps.

Commodity‑Focused Extensions
- Futures curve model: Store full curves; map options to correct futures contracts; Black‑76 pricing for futures options; SABR alternative for smiles.
- Carry & convenience yield: Interest/storage/insurance, convenience and seasonal carry modeling; explicit roll yield metrics.
- Data drivers: EIA/WASDE calendars; COT positioning; LME/COMEX stocks; PMI; DXY; freight; weather/ENSO; crop progress; refinery maintenance.
- Curve+smile dynamics: Regime clustering on curve shape; vol‑of‑vol by tenor; jump intensity around scheduled reports; smile kink detection.
- Multi‑asset spreads: Crack/crush/spark spread option approximations; diagonal/calendar RV across tenor.
- Strategies: Seasonality‑aware calendars/diagonals; pre‑event long‑vol; post‑event mean‑reversion shorts; skew trades (RR/flies).
- Risk & execution: Exchange limit/lock modeling; delta hedging with correct futures contracts; margin modeling; tenor‑bucket CVaR.
- Backtesting realism: Historical options on futures with settlement/holiday/limit handling; point‑in‑time chain availability and realistic slippage.

UX & Unification
- “Smart‑Money to Options” flow: Select STRONG names (13F+Form4), analyze surface/seasonality, and suggest options overlay (calendar/diagonal/flies) matched to the term/skew regime.
- Dashboards: Fund performance profiles; stock dossiers (flows/insiders/vol surface); commodity dashboards (curve, seasonality, alerts).
- Artifact lineage: Every signal/report references source filings/transactions and surface versions for auditability.

Phased Roadmap
- 0–2 weeks: 13F correctness (XPath + amendments + FIGI), holdings time‑series, consensus/crowding/conviction, QA tests.
- 2–4 weeks: Insider engine (parser, abnormal returns, grading, combined signals), 10‑year backtests, leaderboards.
- 4–8 weeks: Arbitrage‑free SVI‑Joint, Heston/Merton validation, strategy pricing, regime router, historical options backtests.
- Ongoing: Commodity drivers, Black‑76/SABR smiles, COT/event gating, dashboards, and portfolio risk overlays.

