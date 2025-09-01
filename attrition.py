# attrition_dashboard.py
import os, json, time, math, datetime
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# -------------------- App Setup --------------------
load_dotenv()
st.set_page_config(page_title="Attrition Prediction", page_icon="📉", layout="wide")

def _init_state():
    st.session_state.setdefault("source_df", None)      # original uploaded data
    st.session_state.setdefault("result_df", None)      # normalized predictions
    st.session_state.setdefault("all_results", None)    # raw JSON rows from o3-mini
    st.session_state.setdefault("selected_emp", None)
    st.session_state.setdefault("file_name", None)

_init_state()

# -------------------- Azure OpenAI (o3-mini) --------------------
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3-mini")  # your deployed name for o3-mini
REASONING_EFFORT = os.getenv("O3_REASONING_EFFORT", "medium")              # low | medium | high
MAX_OUTPUT_TOKENS = int(os.getenv("O3_MAX_OUTPUT_TOKENS", "50000"))         # tune as needed
BATCH_SIZE = 25                                                             # fixed batch size; change if you want

# ==== Client factory ====
def get_client() -> AzureOpenAI:
    # Keep your current behavior (env with fallback)
    api_key = os.getenv("AZURE_OPENAI_API_KEY","1DkyvFGwRKbcWKFYxfGAot2s8Qc9UPM8NmsbJR2OJDWJTBs084usJQQJ99ALACHYHv6XJ3w3AAABACOGvGzF")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT","https://summarize-gen.openai.azure.com/")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    if not api_key or not endpoint:
        st.error("Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT.")
        st.stop()
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

# -------------------- Scoring Prompt & Config --------------------
SYSTEM_PROMPT = """
You are an HR Attrition Risk Scorer.

Objective
- Given one or more employee records, compute a rule-based attrition risk score (0–100),
  classify Risk (Low/Medium/High) and Criticality (Low/Medium/High), and produce an RC meter.

Output Rules
- Respond ONLY with valid JSON matching this shape:
  { "results":[
      {"employeeId":"", "totalScore":0.0, "riskLevel":"Low|Medium|High",
       "criticality":"Low|Medium|High", "rcMeter":"✅ Low|🟨 Medium|🟧 Medium-High|🟥 High",
       "componentScores":{
         "UnitAttritionRate":0.0,"Competency":0.0,"PerfRating3Y":0.0,"BillabilityStatus":0.0,
         "Location":0.0,"OnsiteOffshore":0.0,"SonataTenureMonths":0.0,"TotalExperienceYears":0.0,
         "ETGorLateral":0.0,"Education":0.0,"DateOfLastPromotion":0.0,"PromotedEver":0.0,
         "PromotedInLast3Years":0.0,"RetentionLast6Months":0.0,"RotationTenureMonths":0.0,
         "NewSkillsAdded6Months":0.0,"EngagementSurveyRisk":0.0
       },
       "factorExplanations": {"<FactorName>":"one-line reason"},
       "rationale":"1–2 concise sentences"
      }
  ]}

Weights sum to 100:
- UnitAttritionRate:5, Competency:10, PerfRating3Y:5, BillabilityStatus:5, Location:2,
  OnsiteOffshore:5, SonataTenureMonths:5, TotalExperienceYears:2, ETGorLateral:5, Education:1,
  DateOfLastPromotion:5, PromotedEver:5, PromotedInLast3Years:5, RetentionLast6Months:15,
  RotationTenureMonths:5, NewSkillsAdded6Months:10, EngagementSurveyRisk:10
- Band/Grade/Unit weight 0 (used only for criticality).

Risk by totalScore: Low < 33, Medium 33–65.99, High ≥ 66.
RC Meter:
- (Low, Low|Medium) => "✅ Low"
- (Low, High)|(Medium, Low|Medium) => "🟨 Medium"
- (Medium, High)|(High, Low) => "🟧 Medium-High"
- (High, Medium|High) => "🟥 High"
"""

CONFIG = {
    "perfScaleMax": 5,
    "highRiskBillability": ["Bench","Buffer","Shadow","Unassigned"],
    "highRiskLocations": ["Bangalore","Hyderabad"],
    "onsiteIsRiskier": False,
    "tenureLow": 12, "tenureHigh": 36,
    "expLow": 2, "expHigh": 6,
    "promRecentMonths": 36,
    "retentionIncreasesRisk": True,
    "rotationLong": 18,
    "skillsReduceRisk": True, "skillsHighCount": 2,
    "engagementMax": 10,
    "criticalBands": ["B8","B9"],
    "highCritComps": ["Cloud","AI","Security"],
    "defaultCriticality": "Medium",
}

def build_user_message(rows: List[Dict[str, Any]]) -> str:
    block = {
        "Configuration": {
            "PerformanceScaleMax": CONFIG["perfScaleMax"],
            "HighRiskBillability": CONFIG["highRiskBillability"],
            "HighRiskLocations": CONFIG["highRiskLocations"],
            "OnsiteIsRiskier": CONFIG["onsiteIsRiskier"],
            "TenureThresholdsMonths": {"low": CONFIG["tenureLow"], "high": CONFIG["tenureHigh"]},
            "ExperienceThresholdsYears": {"low": CONFIG["expLow"], "high": CONFIG["expHigh"]},
            "PromotionRecentIfMonths": CONFIG["promRecentMonths"],
            "RetentionIncreasesRisk": CONFIG["retentionIncreasesRisk"],
            "RotationLongIfMonths": CONFIG["rotationLong"],
            "MoreNewSkillsReduceRisk": CONFIG["skillsReduceRisk"],
            "NewSkillsHighCount": CONFIG["skillsHighCount"],
            "EngagementMax": CONFIG["engagementMax"],
            "CriticalBands": CONFIG["criticalBands"],
            "HighCriticalCompetencies": CONFIG["highCritComps"],
            "DefaultCriticality": CONFIG["defaultCriticality"]
        },
        "Employees": rows
    }
    return "Score attrition risk for these employee records.\n\n" + json.dumps(block, ensure_ascii=False)

def call_llm_o3_json(client: AzureOpenAI, system_msg: str, user_msg: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user",   "content": user_msg}],
        response_format={"type": "json_object"},
        reasoning_effort=REASONING_EFFORT,
        max_completion_tokens=100000,
    )
    return json.loads(resp.choices[0].message.content)

# -------------------- Data hygiene --------------------
def to_basic_type(v: Any):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return float(v) if isinstance(v, np.floating) else int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, datetime.date, datetime.datetime)):
        try:
            return pd.to_datetime(v).date().isoformat()
        except Exception:
            return str(v)
    return v

def sanitize_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    clean = []
    for rec in df.to_dict(orient="records"):
        clean.append({k: to_basic_type(v) for k, v in rec.items()})
    return clean

def chunk(lst: List[Any], k: int) -> List[List[Any]]:
    return [lst[i:i+k] for i in range(0, len(lst), k)]

# -------------------- Styles --------------------
CSS = """
<style>
.hero {background:#E9F1FF;border-radius:18px;padding:28px 32px;margin-bottom:22px}
.hero h1 {margin:0;font-size:40px;letter-spacing:.3px}
.hero p {margin:6px 0 0;color:#4b5563}

.kpi-row {margin-top:6px; margin-bottom:18px}
.kpi {background:#fff;border:1px solid #eee;border-radius:16px;padding:18px 20px;height:120px}
.kpi h3 {font-size:14px;color:#6b7280;margin:0}
.kpi .v {font-size:34px;font-weight:700;margin-top:12px}
.kpi .sub {display:none} /* remove tiny captions under KPI per requirement */

.tabbar-spacer {height:12px}
.stTabs [data-baseweb="tab-list"] {gap: 28px;} /* breathing room between tabs */
.stTabs [data-baseweb="tab"] {padding:10px 16px; font-weight:600;}

.badge {display:inline-block;padding:10px 18px;border-radius:28px;background:#f1f5f9;font-weight:700}

.fcard {border:1px solid #eee;border-radius:12px;padding:10px;margin-bottom:10px}

.insight-bullets li{margin:8px 0}
.section {border-radius:16px;padding:18px 22px;margin:14px 0}
.section.immediate {background:#ECFDF5;border:1px solid #C7F1E6}
.section.medium {background:#EEF5FF;border:1px solid #D5E6FF}
.section.long {background:#F7F2FF;border:1px solid #E8DBFF}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def render_hero():
    st.markdown(
        """
        <div class="hero">
          <h1>Sonata Attrition Prediction Model</h1>
          <p>Upload employee data. Score attrition with a reasoning model and act on insights.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def kpi_row(result_df: pd.DataFrame):
    total = int(result_df.shape[0]) if result_df is not None else 0
    overall = float(result_df["totalScore"].mean()) if (result_df is not None and not result_df.empty) else 0.0
    high_ct = int((result_df["riskLevel"] == "High").sum()) if result_df is not None else 0
    med_ct  = int((result_df["riskLevel"] == "Medium").sum()) if result_df is not None else 0
    low_ct  = int((result_df["riskLevel"] == "Low").sum()) if result_df is not None else 0

    st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi"><h3>Overall Avg Risk Score</h3>'
                    f'<div class="v" style="color:#ef4444">{overall:.1f}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi"><h3>Total Employees</h3>'
                    f'<div class="v">{total:,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi"><h3>High Risk</h3>'
                    f'<div class="v" style="color:#f59e0b">{high_ct:,}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="kpi"><h3>Medium / Low</h3>'
                    f'<div class="v">{med_ct:,} / {low_ct:,}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Sidebar (Batch only) --------------------
with st.sidebar:
    st.header("Batch Upload & Predict")
    upload = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], key="uploader")
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

# -------------------- Run Prediction --------------------
client = get_client()

if predict_clicked:
    if upload is None:
        st.warning("Please upload a file first.")
    else:
        # read source
        if upload.name.lower().endswith(".xlsx"):
            src_df = pd.read_excel(upload)
        else:
            src_df = pd.read_csv(upload)

        if "EmployeeID" not in src_df.columns:
            st.error("The file must include 'EmployeeID'.")
            st.stop()

        rows = sanitize_rows(src_df)
        batches = chunk(rows, BATCH_SIZE)
        all_results, errors = [], []

        with st.spinner(f"Scoring {len(rows)} rows in {len(batches)} batch(es)…"):
            for i, b in enumerate(batches, start=1):
                try:
                    msg = build_user_message(b)
                    data = call_llm_o3_json(client, SYSTEM_PROMPT, msg)
                    res = data.get("results", [])
                    if not isinstance(res, list):
                        raise ValueError("Invalid 'results' from model.")
                    all_results.extend(res)
                except Exception as e:
                    errors.append(f"Batch {i}: {e}")
                time.sleep(0.05)

        if errors:
            st.warning("Some batches failed:")
            for e in errors:
                st.code(e)

        if not all_results:
            st.error("No results returned from the model.")
            st.stop()

        # Normalize JSON results
        result_df = pd.json_normalize(all_results, max_level=1)
        prefer = ["employeeId","totalScore","riskLevel","criticality","rcMeter","rationale"]
        rest   = [c for c in result_df.columns if c not in prefer]
        result_df = result_df[prefer + rest]

        # Merge a few meta cols for analysis charts if present
        meta_cols = [m for m in ["Location","Competency","Unit","Band","Grade"] if m in src_df.columns]
        if meta_cols:
            meta = src_df[["EmployeeID"] + meta_cols].copy()
            meta = meta.rename(columns={"EmployeeID":"employeeId"})
            for m in meta_cols:
                meta.rename(columns={m: f"meta.{m.lower()}"}, inplace=True)
            result_df = result_df.merge(meta, on="employeeId", how="left")

        # Store
        st.session_state.source_df   = src_df
        st.session_state.result_df   = result_df
        st.session_state.all_results = all_results
        st.session_state.file_name   = upload.name
        st.session_state.selected_emp = result_df["employeeId"].iloc[0] if len(result_df) else None

# -------------------- Dashboard --------------------
render_hero()

pred_df = st.session_state.result_df
if pred_df is None or pred_df.empty:
    st.info("Upload a file in the sidebar and click **Predict with o3-mini** to see the dashboard.")
    st.stop()

# KPI cards (static, always on top)
kpi_row(pred_df)

# Space, then tabs (placed below KPIs)
st.markdown('<div class="tabbar-spacer"></div>', unsafe_allow_html=True)
tabs = st.tabs(["Overview", "Data Analysis", "Individual Prediction", "Insights & Actions"])

# ---- Overview ----
with tabs[0]:
    st.subheader("Overview")
    # Only downloader here (no long table)
    buf = BytesIO()
    pred_df.to_csv(buf, index=False)
    fname = (st.session_state.file_name or "attrition").replace(".xlsx","").replace(".csv","")
    st.download_button("⬇️ Download Predictions (CSV)",
                       data=buf.getvalue(),
                       file_name=f"{fname}_predictions.csv",
                       mime="text/csv",
                       type="primary")

# ---- Data Analysis ----
with tabs[1]:
    st.subheader("Data Analysis")
    st.caption("Explore average attrition risk by key dimensions.")
    # Location chart
    if "meta.location" in pred_df.columns:
        df_loc = pred_df.groupby("meta.location")["totalScore"].mean().reset_index()
        df_loc.rename(columns={"meta.location":"Location","totalScore":"AvgRisk"}, inplace=True)
        fig = px.bar(df_loc, x="Location", y="AvgRisk", title="Attrition Rate by Location")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    # Competency chart
    if "meta.competency" in pred_df.columns:
        df_c = pred_df.groupby("meta.competency")["totalScore"].mean().reset_index()
        df_c.rename(columns={"meta.competency":"Competency","totalScore":"AvgRisk"}, inplace=True)
        fig2 = px.bar(df_c, x="Competency", y="AvgRisk", title="Attrition Rate by Technology / Competency")
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, use_container_width=True)

# ---- Individual Prediction ----
with tabs[2]:
    st.subheader("Individual Prediction")
    emp_ids = pred_df["employeeId"].tolist()
    default_idx = 0 if st.session_state.selected_emp is None else emp_ids.index(st.session_state.selected_emp)
    emp = st.selectbox("Select Employee", emp_ids, index=default_idx)
    st.session_state.selected_emp = emp
    row = pred_df[pred_df["employeeId"] == emp].iloc[0].to_dict()

    c1,c2,c3,c4 = st.columns([1.1,1.1,1.1,1.4])
    with c1:
        st.markdown("**Total Score**")
        st.progress(min(100, int(row["totalScore"])))
        st.markdown(f"<div style='font-size:14px;margin-top:4px;'>{row['totalScore']:.2f}/100</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("**Risk**"); st.markdown(f"<div class='badge'>{row['riskLevel']}</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("**Criticality**"); st.markdown(f"<div class='badge'>{row['criticality']}</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("**RC Meter**"); st.markdown(f"<div class='badge'>{row['rcMeter']}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Explanation:** {row.get('rationale','')}")

    # componentScores.* -> top bar
    comp_cols = [c for c in pred_df.columns if c.startswith("componentScores.")]
    if comp_cols:
        pairs=[]
        for c in comp_cols:
            f=c.split(".",1)[1]; pairs.append((f, float(row.get(c,0.0))))
        top_df = pd.DataFrame(sorted(pairs, key=lambda x: x[1], reverse=True), columns=["Factor","Points"])
        fig3 = px.bar(top_df.head(10), x="Points", y="Factor", orientation="h",
                      title=f"Top Contributing Factors — {emp}")
        fig3.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10), font=dict(size=12))
        st.plotly_chart(fig3, use_container_width=True)

    # factorExplanations.* -> cards
    fact_cols = [c for c in pred_df.columns if c.startswith("factorExplanations.")]
    if fact_cols:
        st.markdown("### Key Risk Factors")
        colA,colB,colC = st.columns(3)
        for i, c in enumerate(sorted(fact_cols)):
            factor = c.split(".",1)[1]
            text   = str(row.get(c,"")).strip()
            if not text: continue
            box = {0:colA,1:colB,2:colC}[i%3]
            with box:
                st.markdown(f"<div class='fcard'><b>{factor}</b><br>"
                            f"<span style='font-size:12px'>{text}</span></div>",
                            unsafe_allow_html=True)

# ---- Insights & Actions (o3-mini) ----
with tabs[3]:
    st.subheader("Insights & Actions")

    # Aggregate summary for LLM
    agg = {
        "overall_avg_score": float(pred_df["totalScore"].mean()),
        "high_risk_pct": float((pred_df["riskLevel"]=="High").mean()*100),
        "medium_risk_pct": float((pred_df["riskLevel"]=="Medium").mean()*100),
        "low_risk_pct": float((pred_df["riskLevel"]=="Low").mean()*100),
    }
    comp_cols = [c for c in pred_df.columns if c.startswith("componentScores.")]
    topf = (pred_df[comp_cols].mean().sort_values(ascending=False).head(8)
            if comp_cols else pd.Series(dtype=float))
    topf = [{"factor": k.split(".",1)[1], "avgPoints": float(v)} for k,v in topf.items()]

    sys = "You are an HR analytics consultant. Return ONLY valid HTML snippets (no markdown)."
    usr = f"""
Dataset summary JSON:
{json.dumps({"summary": agg, "topFactors": topf}, ensure_ascii=False)}

Produce two HTML sections styled like:
1) <ul class="insight-bullets"> with 4–6 concise bullet insights. Each li starts with a colored dot emoji span: 
   <span style='color:#ef4444'>●</span>, <span style='color:#f59e0b'>●</span>, <span style='color:#eab308'>●</span>, <span style='color:#3b82f6'>●</span>.
   Each bullet ≤ 18 words, board-ready.
2) Three <div class="section ..."> blocks titled 'Immediate Actions', 'Medium-term Strategies', 'Long-term Initiatives'
   with 3–4 short bullet items each. Use classes: immediate | medium | long.

Do not wrap with <html> or <body>. No headings besides those in the sections.
"""
    with st.spinner("Generating insights with o3-mini…"):
        html = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
            response_format={"type":"text"},
            reasoning_effort=REASONING_EFFORT,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        ).choices[0].message.content.strip()

    # Render two blocks (bullets + action sections)
    st.markdown("### Key Insights")
    st.markdown(html, unsafe_allow_html=True)
