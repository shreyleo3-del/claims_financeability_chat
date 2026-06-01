# HealthCRED Claim Intelligence

**HealthCRED Claim Intelligence** is a Streamlit chat application that helps healthcare finance teams evaluate accounts-receivable (AR) claim financeability. It combines a pre-scored Howard University Hospital Q4 2025 portfolio (48,597 claims, ~$530M gross billed) with Claude AI to answer claim-level and portfolio-level questions using a transparent, rules-based scoring model.

Built for due-diligence workflows (e.g. VMG Healthcare Consultants evaluating a ~$122M financing facility).

**Vidyasoh Healthcare Tech Service Pvt Ltd — HealthCRED Claim Intelligence v1.0**

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Data files](#data-files)
- [Scoring model overview](#scoring-model-overview)
- [Prerequisites](#prerequisites)
- [Local setup](#local-setup)
- [Configuration](#configuration)
- [Using the app](#using-the-app)
- [Deploy a public shareable link](#deploy-a-public-shareable-link)
- [Project structure](#project-structure)
- [Regenerating data (notebook)](#regenerating-data-notebook)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [License and attribution](#license-and-attribution)

---

## Features

### Portfolio dashboard (landing page)

On first load (before any chat messages), the app displays:

- **Portfolio metrics:** total claims, gross billed, eligible pool billed (Finance Eligible + Eligible with Haircut), hard-stop count and billed amount
- **Decision band cards:** Finance Eligible, Eligible with Haircut, Manual Review, Fix Before Finance, Not Eligible, Hard Stop — with claim counts and billed totals

The dashboard hides during an active conversation so the chat area stays clean while Claude is analyzing.

### AI chat assistant

- Natural-language questions via **`st.chat_input()`** (no file upload required)
- Powered by **Claude Sonnet** (`claude-sonnet-4-20250514`)
- Conversation history persisted in session state
- Sidebar **example questions** for common analyses
- **Clear conversation** button to reset chat and restore the dashboard

### Two query modes

| Mode | Trigger | Context sent to Claude |
|------|---------|------------------------|
| **Claim lookup** | UUID in message (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`) | Full claim row JSON + matched payer performance + live portfolio summary |
| **Portfolio / what-if** | Any other question | Live portfolio summary (no single claim row required) |

### Enriched portfolio context (every request)

Each Claude turn receives a freshly built JSON summary including:

- Decision breakdown, hard-stop reasons, eligibility totals
- **`payer_performance`:** top 15 payer–filing rows by billed (from 12-month lookback table, 84 rows)
- **`procedure_summary`:** top 15 CPT/HCPCS by billed + D2 coding-issue rollups
- Precomputed analytics (bundling counts, Fix Before Finance, HS-04 recoupment, indicative advance at 68%, etc.)

### Underwriter lens (procedure / coding)

For procedure and coding questions, Claude is instructed to think like a **payer underwriter**: assess whether `procedure_code_list` combinations are likely to pass cleanly or trigger bundling, medical necessity, unbundling, or remit-mismatch denials, and state clean-pass likelihood (high / medium / low).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                        │
├─────────────────────────────────────────────────────────────────┤
│  Startup                                                         │
│    claim_explanation_spine.csv  →  st.session_state.df          │
│    payer_performance.csv        →  st.session_state.payer_df   │
├─────────────────────────────────────────────────────────────────┤
│  Each chat turn                                                  │
│    build_portfolio_summary(df, payer_df)                        │
│    build_procedure_summary() [cached]                          │
│    [optional] lookup_claim(UUID) + lookup_payer_performance()  │
│    build_user_message() → Anthropic Messages API               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Claude (system prompt =
                    full scoring rules +
                    portfolio context guidance)
```

**Upstream analytics:** Claim scores and CSV exports are produced in [`ClaimSpine&Analysis.ipynb`](ClaimSpine&Analysis.ipynb) using DuckDB over EDI 835 remittance and claim CSVs. The Streamlit app reads the exported CSVs only (no DuckDB at runtime).

---

## Data files

### Required by the app (commit for Streamlit Cloud)

| File | Session state | Description |
|------|---------------|-------------|
| [`outputs/claim_explanation_spine.csv`](outputs/claim_explanation_spine.csv) | `st.session_state.df` | **48,597 rows × 88 columns.** One row per claim with scores, decisions, hard-stop flags, ATB/remit fields, `procedure_code_list`, `revenue_code_list`, etc. (~22 MB) |
| [`outputs/payer_performance.csv`](outputs/payer_performance.csv) | `st.session_state.payer_df` | **84 rows × 21 columns.** Payer–filing behavioral metrics (12-month lookback, min 10 claims per group). (~10 KB) |

**Join key (claim → payer performance):**  
`subscriber_payer_name` + `ClaimFilingTypeCode` ↔ `payer_name` + `filing_type_code`  
(~80% of spine claims match a payer-performance row.)

### Supporting outputs (not loaded by Streamlit)

| File | Purpose |
|------|---------|
| `outputs/scored_claims.csv` | Full scored export from notebook |
| `outputs/eligible_claims.csv` | Finance Eligible + Haircut subset |
| `outputs/decision_summary.csv` | Decision band aggregates |
| `outputs/hard_stop_breakdown.csv` | Hard-stop reason counts |

---

## Scoring model overview

The complete rules are embedded in the app’s system prompt. Summary:

### Portfolio context

- **Hospital:** Howard University Hospital  
- **Period:** Q4 2025 AR  
- **Scale:** 48,597 claims, ~$530.1M gross billed  
- **Billing entity NPI:** 1487740957  

### Hard stops (any = Hard Stop, 0% advance, no scoring)

| Code | Rule |
|------|------|
| HS-01 | `ClaimStatusCode = 4` (denied) |
| HS-02 | `ClaimStatusCode = 22` (reversal) |
| HS-03 | `frequency_type` in 7, 8 (void/replacement) |
| HS-04 | `has_wo_recoupment = 1` (active recoupment) |
| HS-05 | `handling_code = H` (payment held) |
| HS-06 | `atb_has_negative_balance = 1` |
| HS-07 | `ClaimStatusCode` in 19, 20 (forwarded) |
| HS-08 | `coverage_expiration_date < StatementFromDate` |
| HS-10 | `payment_method_code = NON` and not denied/reversed |

### Payer tracks

| Track | Types | Max advance |
|-------|--------|-------------|
| A | Commercial (CI, BL, 12, 13, 16, AM) | 85% |
| B | Medicare / Champus (MA, MB, HM, CH) | 75% |
| C | Medicaid (MC, TV) | 70% |
| D | Manual (WC, VA, ZZ, null) | No scoring (Manual Review) |

### Score dimensions (max 100 before penalties)

| Dimension | Max | Focus |
|-----------|-----|--------|
| D1 | 20 | Coverage and authorization |
| D2 | 25 | Completeness and coding (deductions from 25) |
| D3 | 20 | Payer behavior (denial, recovery, days-to-pay) |
| D4 | 20 | Provider / NPI risk |
| D5 | 15 | AR and ATB status (deductions from 15) |
| Match penalty | −29 | Remit/ATB/NPI/tax ID integrity |

### Decision bands (by track)

Examples for Track A (Commercial): ≥85 Finance Eligible (80–85% advance tiers), 78–84 Haircut 75%, 70–77 Haircut 65%, 50–69 Fix Before Finance, &lt;50 Not Eligible. Tracks B and C use different thresholds (see system prompt in `app.py`).

---

## Prerequisites

- **Python 3.11** (see [`runtime.txt`](runtime.txt))
- **Anthropic API key** with access to `claude-sonnet-4-20250514`
- CSV data files in `outputs/` (see above)
- ~500 MB disk for virtualenv + dependencies

---

## Local setup

### 1. Clone and enter the project

```bash
git clone https://github.com/shreyleo3-del/claims_financeability_chat.git
cd "DuckDB Claim Financeability"
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Pinned packages:

| Package | Version |
|---------|---------|
| streamlit | 1.32.0 |
| anthropic | 0.25.0 |
| httpx | 0.27.2 |
| pandas | 2.2.0 |
| numpy | 1.26.4 |

> **Note:** `httpx==0.27.2` is required. Newer httpx versions break `anthropic==0.25.0` with a `proxies` initialization error.

### 4. Configure secrets

Create [`.streamlit/secrets.toml`](.streamlit/secrets.toml) (this file is gitignored):

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

**Never commit this file or paste your key in chat.**

### 5. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

---

## Configuration

### Streamlit theme

[`.streamlit/config.toml`](.streamlit/config.toml) sets a dark theme aligned with the UI:

- Background: `#0a0a0f`
- Accent: `#1F3864`
- Text: `#f0f0f5`

### App constants (`app.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `CSV_PATH` | `outputs/claim_explanation_spine.csv` | Claim spine |
| `PAYER_CSV_PATH` | `outputs/payer_performance.csv` | Payer performance |
| `MODEL` | `claude-sonnet-4-20250514` | Claude model ID |
| `MAX_TOKENS` | `2000` | Max response tokens |

### API key

- **Local:** `.streamlit/secrets.toml`
- **Streamlit Cloud:** App **Settings → Secrets** (same TOML format)
- There is **no API key field in the UI** by design.

---

## Using the app

### Landing page

1. Review portfolio metrics and decision bands.
2. Use the sidebar **Example Questions** or type in the chat box.

### Example questions (sidebar)

- Explain claim `e3f62dc6-4893-4656-aa55-559300793246`
- How many claims are Fix Before Finance and why?
- Which payer is causing the most hard stops?
- Show top 5 Not Eligible claims by billed amount
- What if recoupment was resolved for all HS-04 claims?
- Why do Medicaid claims score lower than Commercial?
- How many claims have bundling issues?
- What is average D2 score inpatient vs outpatient?
- What is the total indicative advance at 68% blended rate?

### Claim lookup

Include a claim UUID anywhere in your message. The app loads the full spine row and, when possible, the matching payer-performance row.

### During chat

- The dashboard is hidden while you have messages in the conversation.
- Click **Clear conversation** in the sidebar to reset and show the dashboard again.

---

## Deploy a public shareable link

### 1. Push to GitHub

Include:

- `app.py`, `requirements.txt`, `runtime.txt`
- `.streamlit/config.toml`
- `outputs/claim_explanation_spine.csv`
- `outputs/payer_performance.csv`
- `.gitignore`

Exclude:

- `.streamlit/secrets.toml`
- `venv/`, `.venv/`

```bash
git add app.py requirements.txt runtime.txt .streamlit/config.toml .gitignore outputs/claim_explanation_spine.csv outputs/payer_performance.csv
git commit -m "HealthCRED Claim Intelligence"
git push -u origin main
```

### 2. Deploy on Streamlit Community Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **Create app** → select your repository.
3. **Main file path:** `app.py`
4. **Python version:** 3.11 (from `runtime.txt`).

### 3. Add secrets on Streamlit Cloud

**Settings → Secrets:**

```toml
ANTHROPIC_API_KEY = "sk-ant-your-real-key"
```

Reboot the app after saving.

### 4. Share the URL

Your public link will look like:

`https://your-app-name.streamlit.app`

Anyone with the link can use the app. API usage is billed to your Anthropic account.

---

## Project structure

```
DuckDB Claim Financeability/
├── app.py                          # Streamlit application (main entry point)
├── requirements.txt                # Python dependencies
├── runtime.txt                     # python-3.11 (Streamlit Cloud)
├── README.md                       # This file
├── ClaimSpine&Analysis.ipynb       # DuckDB scoring pipeline (source of CSVs)
├── .streamlit/
│   ├── config.toml                 # Dark theme (safe to commit)
│   └── secrets.toml                # API key — LOCAL ONLY, gitignored
├── .gitignore
├── outputs/
│   ├── claim_explanation_spine.csv # Required by app
│   ├── payer_performance.csv       # Required by app
│   ├── scored_claims.csv
│   ├── eligible_claims.csv
│   ├── decision_summary.csv
│   └── hard_stop_breakdown.csv
├── 02_raw_csv/                     # Raw EDI/ATB inputs (notebook)
└── venv/ or .venv/                 # Local virtualenv (gitignored)
```

---

## Regenerating data (notebook)

To rebuild scores and CSVs from raw data:

1. Open [`ClaimSpine&Analysis.ipynb`](ClaimSpine&Analysis.ipynb).
2. Ensure raw CSVs are under `02_raw_csv/edi/` and `02_raw_csv/atb/`.
3. Run all cells to rebuild `claim_spine`, scoring, `payer_performance`, and exports under `outputs/`.
4. Restart the Streamlit app to pick up new files (clear cache: **Streamlit menu → Clear cache** if needed).

The app does not run DuckDB; it only reads the exported CSVs.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Client.init() got an unexpected keyword argument 'proxies'` | Reinstall deps: `pip install -r requirements.txt` (needs `httpx==0.27.2`) |
| Chat says API key not configured | Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` or Streamlit Cloud Secrets |
| `Data file not found` | Run from project root; ensure `outputs/*.csv` exist |
| Dashboard numbers ghosting during “Analyzing…” | Update to latest `app.py`; dashboard hides during chat |
| Text hard to read on dark background | Ensure `.streamlit/config.toml` is present; hard-refresh browser |
| Slow first load | Spine CSV is ~22 MB; `@st.cache_data` caches after first load |
| Claim UUID not found | Verify UUID exists in `claim_explanation_spine.csv` (`ClaimId` column) |
| Payer performance missing on claim | Payer–filing combo may be outside 84-row lookback table; spine fields still available |

---

## Security

- **Do not** commit `.streamlit/secrets.toml` or share API keys in issues, chat, or screenshots.
- Rotate keys if exposed.
- Streamlit Cloud free tier apps are **public by URL** unless you use paid access controls.
- Claim data may contain PHI-adjacent fields; deploy only to environments compliant with your data-handling policies.

---

## License and attribution

**HealthCRED Claim Intelligence v1.0**  
**Vidyasoh Healthcare Tech Service Pvt Ltd**

Portfolio analytics derived from Howard University Hospital Q4 2025 AR data. Scoring logic and exports produced via DuckDB analysis notebook. AI responses powered by Anthropic Claude.

For questions or enhancements, open an issue in the repository or contact the project maintainer.
