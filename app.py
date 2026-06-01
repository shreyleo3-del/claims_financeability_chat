# HealthCRED Claim Intelligence
#
# Deployment (Streamlit Cloud):
#   Push all files except .streamlit/secrets.toml to GitHub.
#   Connect repo, set main file to app.py, add secret ANTHROPIC_API_KEY in Settings, deploy.
#   Local dev uses outputs/claim_explanation_spine.csv and outputs/payer_performance.csv.

import json
import re
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import streamlit as st

CSV_PATH = "outputs/claim_explanation_spine.csv"
PAYER_CSV_PATH = "outputs/payer_performance.csv"
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 2000
UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

ELIGIBLE_DECISIONS = ("Finance Eligible", "Eligible with Haircut")

DECISION_COLORS = {
    "Finance Eligible": "#22c55e",
    "Eligible with Haircut": "#eab308",
    "Manual Review": "#9ca3af",
    "Fix Before Finance": "#f97316",
    "Not Eligible": "#ef4444",
    "Hard Stop": "#7f1d1d",
}

DECISION_ORDER = [
    "Finance Eligible",
    "Eligible with Haircut",
    "Manual Review",
    "Fix Before Finance",
    "Not Eligible",
    "Hard Stop",
]

EXAMPLE_QUESTIONS = [
    "Explain claim e3f62dc6-4893-4656-aa55-559300793246",
    "How many claims are Fix Before Finance and why?",
    "Which payer is causing the most hard stops?",
    "Show top 5 Not Eligible claims by billed amount",
    "What if recoupment was resolved for all HS-04 claims?",
    "Why do Medicaid claims score lower than Commercial?",
    "How many claims have bundling issues?",
    "What is average D2 score inpatient vs outpatient?",
    "What is the total indicative advance at 68% blended rate?",
]

SYSTEM_PROMPT = """You are HealthCRED Claim Intelligence, an expert assistant for Michele at VMG Healthcare Consultants evaluating a $122M financing facility against the Howard University Hospital Q4 2025 AR portfolio.

CONTEXT: Howard University Hospital Q4 2025 AR portfolio. 48,597 claims. $530.1M gross billed. Single billing entity NPI 1487740957.

HARD STOPS (any fires = Hard Stop, 0% advance, no scoring):
HS-01: ClaimStatusCode=4 (denied)
HS-02: ClaimStatusCode=22 (reversal)
HS-03: frequency_type in 7,8 (void/replacement)
HS-04: has_wo_recoupment=1 (active recoupment)
HS-05: handling_code=H (payment held)
HS-06: atb_has_negative_balance=1 (negative ATB)
HS-07: ClaimStatusCode in 19,20 (forwarded)
HS-08: coverage_expiration_date < StatementFromDate
HS-10: payment_method_code=NON and not denied/reversed

PAYER TRACKS:
A=Commercial (CI,BL,12,13,16,AM) max 85%
B=Medicare/Champus (MA,MB,HM,CH) max 75%
C=Medicaid (MC,TV) max 70%
D=Manual (WC,VA,ZZ,null) no scoring

D1 COVERAGE AND AUTH (max 20):
Base: Commercial 20/17/16, Medicare 14/12/12, Medicaid 12/10/10, Unknown 10/8/6 (both auth / claim only / none)
Modifiers: tertiary_payer -5, secondary_payer -2, resubmission -2, status_code_2 -1, cob_paid>0 +1
If no_remittance_match=1, D1=0

D2 COMPLETENESS AND CODING (max 25, deduction from 25):
-25: PrincipalDiagnosisCode is null
-17: inpatient AND DRGCode is null
-10: adj_reason_29_count>0 (late filing)
-8: svc_adj_reason_181_count>0 (invalid procedure)
-8/-6/-4/-2: dropped_procedure_ratio >0.50/>0.25/>0.10/>0
-5: malformed_icd_count>0
-7: svc_adj_reason_50_count/service_line_count>0.25 (systemic medical necessity)
-4: svc_adj_reason_50_count>0 and ratio<=0.25 (isolated)
-7/-4/-2: non_covered_ratio >0.30/>0.10/>0
-6/-4/-2: DRG mismatch with remit_drg_weight <1.0/1.0-2.0/>2.0
-6: adj_bundling_count>0 (claim bundling RC97)
-4: svc_adj_reason_97_count>0 (service bundling)
-6/-4/-2: proc_code_coverage <0.30/<0.50/<0.80 (outpatient only)
-4: adj_reason_96_count>0 (non-covered RC96)
-3: original_proc_changed_count>0
-3: adj_distinct_reason_codes>5
-3/-2: missing_dx_pointer_ratio >0.50/>0.25
-2: has_hospital_acquired_condition=1
-2: modifier_present_count=0 AND service_line_count>5

D3 PAYER BEHAVIOR (max 20):
Denial score (max 7): 0%=7, <=5%=6, <=15%=4, <=30%=2, >30%=1. Fallback: A=5, B=4, C=3, D=2
Recovery score (max 7): >=0.70=7, >=0.55=6, >=0.40=5, >=0.25=3, >=0.10=1, <0.10=0. Fallback: A=4, B=3, C=3, D=2
DTP score (max 6): <=15d=6, <=30d=5, <=60d=3, <=90d=2, >90d=0, null=3
Modifiers: has_l6_interest -1, adj_reason_253_count>0 -1, has_fb_forwarding -1
PI adj ratio: >0.15=-3, >0.05=-2, >0=-1
DRG+DTP: remit_drg_weight>5 AND days_to_pay>60=-2, weight>3 AND dtp>45=-1

D4 PROVIDER RISK (max 20):
Base NPI: all match=20, billing=attending payee differs=12, billing=payee attending differs=14, all differ=8, attending null=16, billing null=4
Modifiers: distinct_provider_npi_count>3=-3, operating_npi differs from billing=-1, provider_avg_drg_weight<0.5=-2

D5 AR AND ATB STATUS (max 15, deduction from 15):
Aging: >1095d=-12, >730d=-7, >365d=-5, >180d=-3, >90d=-2, <=90d=0
no_atb_match AND claim_age>30=-5
atb_payment_ratio<0.05 AND days>180=-3
atb_adjustment_ratio>0.80=-3
financial_class_changed=-2
atb_payer is null AND no_atb_match=0=-3
atb_zero_charges=-5
remit_drg_weight>5 AND days>365=-2
has_fb_forwarding=-2
has_cs_adjustment=-1
adjudication_cycle_count>3=-6, =3=-4, =2=-2
ClaimOriginalReferenceNumber not null=-3
DischargeStatusCode=20=-2
remit_has_ma02_appeal=-2

MATCH INTEGRITY PENALTY (max -29):
no_remittance_match=-10
no_atb_match AND claim_age>30=-5
abs(remit_charges-billed)/billed>0.20=-3
frequency_mismatch=-3
payee_npi != billing_npi=-3
payee_tax_id != billing_tax_id=-5

DECISION BANDS:
Track A: >=85=Finance Eligible (95+=85%, 90-94=83%, 85-89=80%), 78-84=Haircut 75%, 70-77=Haircut 65%, 50-69=Fix Before Finance, <50=Not Eligible
Track B: >=75=Finance Eligible (90+=75%, 82-89=72%, 75-81=70%), 65-74=Haircut 60%, 55-64=Haircut 55%, 40-54=Fix Before Finance, <40=Not Eligible
Track C: >=72=Finance Eligible (88+=70%, 80-87=67%, 72-79=65%), 62-71=Haircut 55%, 52-61=Haircut 50%, 38-51=Fix Before Finance, <38=Not Eligible

PORTFOLIO CONTEXT (injected each request):
- payer_performance: pre-computed D3 behavioral metrics per payer-filing (12-month lookback, 84 rows; top 15 by billed in summary). Prefer this for denial rate, recovery, DTP, recoupment, forwarding, and payer scores — do not re-aggregate from raw claims.
- procedure_summary: top 15 CPT/HCPCS by billed from procedure_code_list plus portfolio D2 coding-issue rollups. For procedure questions use this plus claim-level procedure_code_list, revenue_code_list, proc_code_coverage, and D2 rules.

UNDERWRITER LENS (procedure / coding questions and claim lookups with procedure_code_list):
Think like a payer underwriter adjudicating whether the claim will cleanly pass. For each claim or code mix, assess:
- Code combination coherence: do CPT/HCPCS codes on the same claim belong together clinically and by setting (inpatient DRG vs outpatient lab/imaging/E&M mix)?
- Bundling and unbundling risk: patterns that trigger RC97/service bundling (adj_bundling_count, svc_adj_reason_97_count), duplicate or overlapping E&M, panel + component codes, high service_line_count with modifier gaps.
- Diagnosis linkage: PrincipalDiagnosisCode and care_type vs procedure set; medical necessity flags (svc_adj_reason_50_count, non_covered_ratio).
- Remit alignment: proc_code_coverage, dropped_procedure_ratio, original_proc_changed_count — signals billed vs paid procedure mismatch.
- Clean-pass vs audit risk: state likelihood (high/medium/low) of clean adjudication and what would break (denial, downcode, bundling, medical necessity).
Use portfolio procedure_summary for volume context; use claim-level fields for the specific determination.

BEHAVIOUR: Be precise. Cite actual field values. For claim lookups trace every dimension. For portfolio questions use the live summary. For what-if questions recalculate and show the delta. Audience is Michele at VMG Healthcare Consultants evaluating a $122M financing facility.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

.stApp, .stApp * {
    color: #f0f0f5;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #f0f0f5 !important;
}

.stApp {
    background-color: #0a0a0f;
    color: #f0f0f5;
}

.main .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    color: #f0f0f5 !important;
}

.main p, .main span, .main label, .main li,
.main [data-testid="stMarkdownContainer"] p,
.main [data-testid="stMarkdownContainer"] li,
.main [data-testid="stMarkdownContainer"] span {
    color: #f0f0f5 !important;
}

[data-testid="stSidebar"] {
    background-color: #0f1018;
    border-right: 1px solid #1F3864;
    color: #f0f0f5 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #e8e8ef !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'IBM Plex Mono', monospace;
    color: #f0f0f5 !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #f0f0f5 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
.stCaption, small {
    color: #b8c0d4 !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #12131c 0%, #1a1f2e 100%);
    border: 1px solid #1F3864;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #b8c0d4 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: #ffffff !important;
}

[data-testid="stMetricDelta"] {
    color: #b8c0d4 !important;
}

[data-testid="stChatMessage"] {
    background-color: #12131c;
    border: 1px solid #1F3864;
    border-radius: 8px;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] strong {
    color: #f0f0f5 !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #12131c !important;
    color: #ffffff !important;
    border: 1px solid #1F3864 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #8b93a7 !important;
}

div[data-testid="stSpinner"] {
    color: #f0f0f5 !important;
}

div[data-testid="stButton"] button {
    background-color: #1F3864;
    color: #ffffff !important;
    border: 1px solid #2d4a7a;
    font-size: 0.8rem;
    text-align: left;
    white-space: normal;
    height: auto;
    min-height: 2.5rem;
    padding: 0.5rem 0.75rem;
}

div[data-testid="stButton"] button:hover {
    background-color: #2d4a7a;
    border-color: #3d5a8a;
    color: #ffffff !important;
}
</style>
"""


def format_currency(value: float) -> str:
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:,.0f}K"
    return f"${value:,.2f}"


def row_to_jsonable(row: pd.Series) -> dict:
    out = {}
    for k, v in row.items():
        if pd.isna(v):
            out[k] = None
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        else:
            out[k] = v
    return out


@st.cache_data(show_spinner="Loading claim portfolio…")
def load_claim_data() -> pd.DataFrame:
    path = Path(CSV_PATH)
    if not path.exists():
        st.error(f"Data file not found: {CSV_PATH}")
        st.stop()
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner="Loading payer performance…")
def load_payer_performance() -> pd.DataFrame:
    path = Path(PAYER_CSV_PATH)
    if not path.exists():
        st.error(f"Data file not found: {PAYER_CSV_PATH}")
        st.stop()
    return pd.read_csv(path, low_memory=False)


@st.cache_data
def build_procedure_summary() -> dict:
    df = load_claim_data()
    proc_df = df[df["procedure_code_list"].notna()].copy()
    proc_df["procedure_code"] = proc_df["procedure_code_list"].str.split("|")
    exploded = proc_df.explode("procedure_code")
    exploded["procedure_code"] = exploded["procedure_code"].str.strip()
    exploded = exploded[exploded["procedure_code"].astype(bool)]

    if len(exploded) > 0:
        code_grp = (
            exploded.groupby("procedure_code", dropna=False)
            .agg(claim_count=("ClaimId", "count"), total_billed=("billed_amount", "sum"))
            .reset_index()
            .sort_values("total_billed", ascending=False)
            .head(15)
        )
        top_15_by_billed = [
            {
                "procedure_code": row["procedure_code"],
                "claim_count": int(row["claim_count"]),
                "total_billed": float(row["total_billed"]),
                "total_billed_formatted": format_currency(float(row["total_billed"])),
            }
            for _, row in code_grp.iterrows()
        ]
    else:
        top_15_by_billed = []

    bundling_mask = (df["adj_bundling_count"].fillna(0) > 0) | (
        df["svc_adj_reason_97_count"].fillna(0) > 0
    )
    low_coverage_mask = df["proc_code_coverage"].fillna(1) < 0.5
    dropped_proc_mask = df["dropped_procedure_ratio"].fillna(0) > 0.1
    proc_changed_mask = df["original_proc_changed_count"].fillna(0) > 0

    return {
        "top_15_by_billed": top_15_by_billed,
        "claims_with_procedure_codes": int(proc_df.shape[0]),
        "underwriter_lens": (
            "Evaluate procedure_code_list (and revenue_code_list where present) as a "
            "payer underwriter: assess whether code combinations on the claim are likely "
            "to pass cleanly or trigger bundling, medical necessity, unbundling, or "
            "remit mismatch denials. Cross-check proc_code_coverage, dropped_procedure_ratio, "
            "bundling counts, and PrincipalDiagnosisCode. State clean-pass likelihood "
            "(high/medium/low) and specific risk drivers."
        ),
        "d2_coding_issue_rollups": {
            "bundling_claims": int(bundling_mask.sum()),
            "bundling_billed": float(df.loc[bundling_mask, "billed_amount"].sum()),
            "low_proc_code_coverage_claims": int(low_coverage_mask.sum()),
            "dropped_procedure_ratio_gt_10pct": int(dropped_proc_mask.sum()),
            "original_proc_changed_claims": int(proc_changed_mask.sum()),
        },
    }


def build_payer_performance_summary(payer_df: pd.DataFrame) -> dict:
    top = payer_df.sort_values("total_billed", ascending=False).head(15)
    rows = []
    for _, row in top.iterrows():
        entry = row_to_jsonable(row)
        entry["total_billed_formatted"] = format_currency(float(row["total_billed"]))
        rows.append(entry)
    return {
        "source": "12-month lookback per ClaimSpine notebook",
        "total_payer_filing_rows": int(len(payer_df)),
        "top_15_by_billed": rows,
    }


def build_portfolio_summary(df: pd.DataFrame, payer_df: pd.DataFrame) -> dict:
    total_claims = len(df)
    total_billed = float(df["billed_amount"].sum())

    eligible_mask = df["financeability_decision"].isin(ELIGIBLE_DECISIONS)
    eligible_pool_billed = float(df.loc[eligible_mask, "billed_amount"].sum())

    hard_stop_mask = (df["hard_stop"] == 1) | (
        df["financeability_decision"] == "Hard Stop"
    )
    hard_stop_count = int(hard_stop_mask.sum())
    hard_stop_billed = float(df.loc[hard_stop_mask, "billed_amount"].sum())

    decision_grp = (
        df.groupby("financeability_decision", dropna=False)
        .agg(claim_count=("ClaimId", "count"), total_billed=("billed_amount", "sum"))
        .reset_index()
        .sort_values("total_billed", ascending=False)
    )
    decision_breakdown = [
        {
            "decision": row["financeability_decision"],
            "claim_count": int(row["claim_count"]),
            "total_billed": float(row["total_billed"]),
            "total_billed_formatted": format_currency(float(row["total_billed"])),
        }
        for _, row in decision_grp.iterrows()
    ]

    hs_df = df.loc[hard_stop_mask & df["hard_stop_reason"].notna()]
    if len(hs_df) > 0:
        hs_grp = (
            hs_df.groupby("hard_stop_reason", dropna=False)
            .agg(claim_count=("ClaimId", "count"), total_billed=("billed_amount", "sum"))
            .reset_index()
            .sort_values("total_billed", ascending=False)
        )
        hard_stop_reasons = [
            {
                "reason": row["hard_stop_reason"],
                "claim_count": int(row["claim_count"]),
                "total_billed": float(row["total_billed"]),
                "total_billed_formatted": format_currency(float(row["total_billed"])),
            }
            for _, row in hs_grp.iterrows()
        ]
    else:
        hard_stop_reasons = []

    bundling_mask = (df["adj_bundling_count"].fillna(0) > 0) | (
        df["svc_adj_reason_97_count"].fillna(0) > 0
    )
    fix_before = df[df["financeability_decision"] == "Fix Before Finance"]
    not_eligible = df[df["financeability_decision"] == "Not Eligible"]

    hs04_mask = df["hard_stop_reason"].astype(str).str.contains("HS-04", na=False)
    hs04 = df.loc[hs04_mask]

    d2_by_care = (
        df[df["financeability_decision"] != "Hard Stop"]
        .groupby("care_type", dropna=False)["score_d2"]
        .mean()
        .round(2)
        .to_dict()
    )

    payer_hard_stops = (
        df.loc[hard_stop_mask]
        .groupby("subscriber_payer_name", dropna=False)
        .agg(claim_count=("ClaimId", "count"), total_billed=("billed_amount", "sum"))
        .reset_index()
        .sort_values("claim_count", ascending=False)
    )
    top_hard_stop_payers = [
        {
            "payer": row["subscriber_payer_name"] if pd.notna(row["subscriber_payer_name"]) else "(Unknown)",
            "claim_count": int(row["claim_count"]),
            "total_billed": float(row["total_billed"]),
        }
        for _, row in payer_hard_stops.head(10).iterrows()
    ]

    top5_not_eligible = (
        not_eligible.nlargest(5, "billed_amount")[
            ["ClaimId", "subscriber_payer_name", "billed_amount", "financeability_score"]
        ]
        .to_dict(orient="records")
    )

    analytics = {
        "fix_before_finance_count": int(len(fix_before)),
        "fix_before_finance_billed": float(fix_before["billed_amount"].sum()),
        "bundling_issue_count": int(bundling_mask.sum()),
        "bundling_issue_billed": float(df.loc[bundling_mask, "billed_amount"].sum()),
        "avg_score_d2_by_care_type": {str(k): v for k, v in d2_by_care.items()},
        "hs04_recoupment_count": int(len(hs04)),
        "hs04_recoupment_billed": float(hs04["billed_amount"].sum()),
        "indicative_advance_at_68pct": float(eligible_pool_billed * 0.68),
        "top_hard_stop_payers": top_hard_stop_payers,
        "top_5_not_eligible_by_billed": top5_not_eligible,
        "avg_score_by_payer_track": (
            df[df["financeability_decision"] != "Hard Stop"]
            .groupby("payer_track")["financeability_score"]
            .mean()
            .round(2)
            .to_dict()
        ),
        "medicaid_vs_commercial_avg_score": {
            "track_A_commercial": float(
                df.loc[
                    (df["payer_track"] == "A")
                    & (df["financeability_decision"] != "Hard Stop"),
                    "financeability_score",
                ].mean()
            )
            if len(df[df["payer_track"] == "A"]) > 0
            else None,
            "track_C_medicaid": float(
                df.loc[
                    (df["payer_track"] == "C")
                    & (df["financeability_decision"] != "Hard Stop"),
                    "financeability_score",
                ].mean()
            )
            if len(df[df["payer_track"] == "C"]) > 0
            else None,
        },
    }

    return {
        "totals": {
            "total_claims": total_claims,
            "total_billed": total_billed,
            "total_billed_formatted": format_currency(total_billed),
            "eligible_pool_billed": eligible_pool_billed,
            "eligible_pool_billed_formatted": format_currency(eligible_pool_billed),
            "hard_stop_count": hard_stop_count,
            "hard_stop_billed": hard_stop_billed,
            "hard_stop_billed_formatted": format_currency(hard_stop_billed),
        },
        "decision_breakdown": decision_breakdown,
        "hard_stop_reasons": hard_stop_reasons,
        "payer_performance": build_payer_performance_summary(payer_df),
        "procedure_summary": build_procedure_summary(),
        "analytics": analytics,
    }


def lookup_claim(df: pd.DataFrame, uuid: str) -> dict | None:
    matches = df[df["ClaimId"].str.lower() == uuid.lower()]
    if matches.empty:
        return None
    return row_to_jsonable(matches.iloc[0])


def lookup_payer_performance(
    payer_df: pd.DataFrame, claim_record: dict
) -> dict | None:
    payer_name = claim_record.get("subscriber_payer_name")
    filing_code = claim_record.get("ClaimFilingTypeCode")
    if payer_name is None or filing_code is None:
        return None
    matches = payer_df[
        (payer_df["payer_name"] == payer_name)
        & (payer_df["filing_type_code"] == filing_code)
    ]
    if matches.empty:
        return None
    row = row_to_jsonable(matches.iloc[0])
    row["total_billed_formatted"] = format_currency(float(matches.iloc[0]["total_billed"]))
    return row


def build_user_message(
    question: str,
    summary: dict,
    claim_record: dict | None = None,
    claim_not_found: str | None = None,
    payer_performance_record: dict | None = None,
    payer_performance_note: str | None = None,
) -> str:
    parts = [
        "LIVE PORTFOLIO SUMMARY (rebuilt this request):",
        json.dumps(summary, indent=2, default=str),
    ]
    if claim_record is not None:
        parts.append("CLAIM RECORD (full row):")
        parts.append(json.dumps(claim_record, indent=2, default=str))
    elif claim_not_found:
        parts.append(f"CLAIM LOOKUP: {claim_not_found}")
    if payer_performance_record is not None:
        parts.append("PAYER PERFORMANCE (matched filing):")
        parts.append(json.dumps(payer_performance_record, indent=2, default=str))
    elif payer_performance_note:
        parts.append(f"PAYER PERFORMANCE: {payer_performance_note}")
    parts.append(f"USER QUESTION:\n{question}")
    return "\n\n".join(parts)


def get_assistant_reply(messages: list, enriched_user_content: str) -> str:
    api_messages = []
    for i, m in enumerate(messages):
        if i == len(messages) - 1 and m["role"] == "user":
            api_messages.append({"role": "user", "content": enriched_user_content})
        else:
            api_messages.append({"role": m["role"], "content": m["content"]})

    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        return (
            "API key not configured. Add ANTHROPIC_API_KEY to "
            ".streamlit/secrets.toml (local) or Streamlit Cloud Secrets."
        )

    if not api_key or api_key == "your-key-here":
        return (
            "Please set a valid ANTHROPIC_API_KEY in "
            ".streamlit/secrets.toml or Streamlit Cloud Secrets."
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=api_messages,
        )
        return response.content[0].text
    except Exception as e:
        return f"I encountered an error contacting Claude: {e}"


def show_dashboard() -> bool:
    """Dashboard only on landing; hidden once the user starts chatting."""
    return len(st.session_state.messages) == 0


def render_stats_row(summary: dict) -> None:
    t = summary["totals"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Claims", f"{t['total_claims']:,}")
    with c2:
        st.metric("Gross Billed", t["total_billed_formatted"])
    with c3:
        st.metric(
            "Eligible Pool",
            t["eligible_pool_billed_formatted"],
            delta="Finance Eligible + Haircut",
            delta_color="off",
        )
    with c4:
        st.metric(
            "Hard Stops",
            f"{t['hard_stop_count']:,}",
            delta=f"{t['hard_stop_billed_formatted']} billed",
            delta_color="off",
        )


def render_decision_cards(summary: dict) -> None:
    breakdown = {d["decision"]: d for d in summary["decision_breakdown"]}
    visible = [d for d in DECISION_ORDER if d in breakdown]
    if not visible:
        return

    st.markdown("##### Decision Bands")
    cols = st.columns(len(visible))
    for col, decision in zip(cols, visible):
        d = breakdown[decision]
        color = DECISION_COLORS.get(decision, "#6b7280")
        with col:
            st.markdown(
                f'<p style="color:{color};font-weight:600;font-size:0.8rem;'
                f'margin-bottom:0.25rem;">{decision}</p>',
                unsafe_allow_html=True,
            )
            st.metric(
                "Claims",
                f"{d['claim_count']:,}",
                delta=d["total_billed_formatted"],
                delta_color="off",
            )


def render_sidebar() -> None:
    st.sidebar.markdown("### Example Questions")
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        if st.sidebar.button(question, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_prompt = question
            st.rerun()

    if st.session_state.messages:
        st.sidebar.markdown("---")
        if st.sidebar.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("_generating", None)
            st.session_state.pop("pending_prompt", None)
            st.rerun()

    st.sidebar.caption(
        "Vidyasoh Healthcare Tech Service Pvt Ltd — "
        "HealthCRED Claim Intelligence v1.0"
    )


def generate_assistant_reply(
    df: pd.DataFrame, payer_df: pd.DataFrame, prompt: str
) -> str:
    summary = build_portfolio_summary(df, payer_df)
    uuids = UUID_PATTERN.findall(prompt)

    claim_record = None
    claim_not_found = None
    payer_performance_record = None
    payer_performance_note = None
    if uuids:
        claim_record = lookup_claim(df, uuids[0])
        if claim_record is None:
            claim_not_found = f"No claim found for ClaimId {uuids[0]}"
        else:
            payer_performance_record = lookup_payer_performance(payer_df, claim_record)
            if payer_performance_record is None:
                payer_performance_note = (
                    f"No payer_performance row for "
                    f"{claim_record.get('subscriber_payer_name')} / "
                    f"{claim_record.get('ClaimFilingTypeCode')} "
                    f"(outside 84-row 12-month lookback table)."
                )

    enriched = build_user_message(
        prompt,
        summary,
        claim_record=claim_record,
        claim_not_found=claim_not_found,
        payer_performance_record=payer_performance_record,
        payer_performance_note=payer_performance_note,
    )
    return get_assistant_reply(st.session_state.messages, enriched)


def main() -> None:
    st.set_page_config(
        page_title="HealthCRED Claim Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "df" not in st.session_state:
        st.session_state.df = load_claim_data()
    if "payer_df" not in st.session_state:
        st.session_state.payer_df = load_payer_performance()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    df = st.session_state.df
    payer_df = st.session_state.payer_df
    summary = build_portfolio_summary(df, payer_df)

    render_sidebar()

    st.title("HealthCRED Claim Intelligence")
    st.caption("Howard University Hospital Q4 2025 AR — Claim Financeability")

    if show_dashboard():
        render_stats_row(summary)
        render_decision_cards(summary)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about a claim or the portfolio…")
    if prompt is None and "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    if (
        st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
        and not st.session_state.get("_generating")
    ):
        st.session_state._generating = True
        last_prompt = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                reply = generate_assistant_reply(df, payer_df, last_prompt)
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state._generating = False
        st.rerun()


if __name__ == "__main__":
    main()
