import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAE Customer Intelligence",
    page_icon="🇦🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --brand-red:    #E8341C;
    --brand-orange: #F97316;
    --brand-amber:  #FBBF24;
    --dark-bg:      #0D0D0D;
    --card-bg:      #161616;
    --card-border:  #2A2A2A;
    --text-primary: #F5F5F5;
    --text-muted:   #888888;
    --gradient:     linear-gradient(135deg, #E8341C 0%, #F97316 50%, #FBBF24 100%);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--dark-bg) !important;
    color: var(--text-primary) !important;
}
.stApp { background-color: var(--dark-bg) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid var(--card-border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Hero banner ── */
.hero {
    background: var(--gradient);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🇦🇪";
    position: absolute;
    right: 48px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 80px;
    opacity: 0.25;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    color: #fff !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: -1px;
}
.hero p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
    font-weight: 300;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--card-border);
    margin-left: 8px;
}

/* ── KPI cards ── */
.kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient);
}
.kpi-card:hover { border-color: var(--brand-orange); }
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 10px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 6px;
}

/* ── Info / upload card ── */
.upload-zone {
    background: var(--card-bg);
    border: 2px dashed var(--card-border);
    border-radius: 16px;
    padding: 48px;
    text-align: center;
}

/* ── Plotly chart card ── */
.chart-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
}

/* ── Metric delta ── */
[data-testid="stMetricDelta"] { color: var(--brand-orange) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Buttons ── */
.stButton>button {
    background: var(--gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}
.stButton>button:hover { opacity: 0.88 !important; }

/* ── Tab styling ── */
[data-baseweb="tab-list"] { background: var(--card-bg) !important; border-radius: 10px !important; padding: 4px !important; }
[data-baseweb="tab"] { color: var(--text-muted) !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; }
[aria-selected="true"] { background: var(--brand-red) !important; color: white !important; border-radius: 8px !important; }

/* ── Selectbox / slider ── */
[data-baseweb="select"] > div { background: var(--card-bg) !important; border-color: var(--card-border) !important; }
.stSlider [data-baseweb="slider"] { color: var(--brand-orange) !important; }

/* ── Divider ── */
hr { border-color: var(--card-border) !important; }

/* ── Prediction result badge ── */
.pred-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
}
.pred-yes { background: rgba(232,52,28,0.2); color: #F97316; border: 1px solid var(--brand-orange); }
.pred-no  { background: rgba(100,100,100,0.2); color: #888; border: 1px solid #444; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--dark-bg); }
::-webkit-scrollbar-thumb { background: var(--card-border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#F5F5F5"),
    margin=dict(l=0, r=0, t=36, b=0),
    colorway=["#E8341C", "#F97316", "#FBBF24", "#FB923C", "#FCA5A5"],
    xaxis=dict(gridcolor="#2A2A2A", showline=False),
    yaxis=dict(gridcolor="#2A2A2A", showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
)
ORANGE_SEQ = px.colors.sequential.Oranges

# ── Helpers ──────────────────────────────────────────────────────────────────
def kpi(label, value, sub=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""

def section(icon, title):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)

def card_chart(fig, height=380):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>UAE Customer Intelligence</h1>
    <p>ML-powered segmentation · purchase prediction · spend forecasting</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px'>
        <span style='font-family:Syne;font-size:1.1rem;font-weight:700;
                     background:linear-gradient(135deg,#E8341C,#FBBF24);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        ⚙ Controls
        </span>
    </div>""", unsafe_allow_html=True)

    n_clusters = st.slider("Number of Segments", 2, 8, 4)
    test_size  = st.slider("Test Split %", 10, 40, 20)
    n_trees    = st.selectbox("Random Forest Trees", [50, 100, 200, 500], index=1)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem;color:#666;line-height:1.6'>
    <b style='color:#F97316'>Required columns:</b><br>
    • <code>Will_Buy</code> — classification target<br>
    • <code>Max_Spend</code> — regression target<br>
    • Any other numeric / categorical features
    </div>""", unsafe_allow_html=True)

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂  Upload your customer dataset (CSV)", type=["csv"])

if not uploaded_file:
    st.markdown("""
    <div class="upload-zone">
        <div style='font-size:3rem;margin-bottom:12px'>📊</div>
        <div style='font-family:Syne;font-size:1.2rem;font-weight:700;color:#F5F5F5;margin-bottom:8px'>
            Drop your CSV to begin
        </div>
        <div style='color:#666;font-size:0.9rem'>
            Needs <code>Will_Buy</code> and <code>Max_Spend</code> columns
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load & encode ─────────────────────────────────────────────────────────────
df = pd.read_csv(uploaded_file)

df_encoded = df.copy()
le_dict = {}
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab_eda, tab_diag, tab_pred, tab_presc, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview",
    "🔍 EDA",
    "🩺 Diagnostic",
    "📈 Predictive",
    "🧭 Prescriptive",
    "🎯 Classification",
    "👥 Segmentation",
    "💰 Spend Forecast",
    "🔮 Predict New",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    section("📋", "Dataset Overview")

    total     = len(df)
    buyers    = int(df["Will_Buy"].sum()) if "Will_Buy" in df.columns else "—"
    buy_rate  = f"{buyers/total*100:.1f}%" if isinstance(buyers, int) else "—"
    avg_spend = f"AED {df['Max_Spend'].mean():,.0f}" if "Max_Spend" in df.columns else "—"
    n_feat    = df.shape[1]

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi("Total Customers", f"{total:,}", "in dataset"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Likely Buyers", f"{buyers:,}" if isinstance(buyers,int) else buyers, f"{buy_rate} conversion rate"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Avg Max Spend", avg_spend, "per customer"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Features", f"{n_feat}", "columns detected"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        section("📊", "Feature Distributions")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        sel = st.selectbox("Select feature", num_cols, label_visibility="collapsed")
        fig = px.histogram(df, x=sel, nbins=30,
                           color_discrete_sequence=["#F97316"])
        fig.update_traces(marker_line_width=0)
        card_chart(fig, 320)

    with col_r:
        section("🥧", "Buy Intent Split")
        if "Will_Buy" in df.columns:
            vc = df["Will_Buy"].value_counts().reset_index()
            vc.columns = ["Will_Buy", "Count"]
            vc["Label"] = vc["Will_Buy"].map({1: "Will Buy", 0: "Won't Buy"})
            fig2 = px.pie(vc, names="Label", values="Count",
                          color_discrete_sequence=["#E8341C", "#2A2A2A"],
                          hole=0.55)
            fig2.update_traces(textfont_color="white", pull=[0.04, 0])
            card_chart(fig2, 320)

    section("🗂", "Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB EDA — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    section("🔍", "Exploratory Data Analysis")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # ── Data Quality KPIs ──
    total_cells  = df.shape[0] * df.shape[1]
    missing_vals = df.isnull().sum().sum()
    dup_rows     = df.duplicated().sum()
    completeness = (1 - missing_vals / total_cells) * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi("Rows", f"{df.shape[0]:,}", "records"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Columns", f"{df.shape[1]}", f"{len(num_cols)} numeric · {len(cat_cols)} categorical"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Missing Values", f"{missing_vals:,}", f"{missing_vals/total_cells*100:.1f}% of cells"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Completeness", f"{completeness:.1f}%", f"{dup_rows} duplicate rows"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Missing Value Heatmap ──
    missing_per_col = df.isnull().sum()
    missing_per_col = missing_per_col[missing_per_col > 0]

    if len(missing_per_col) > 0:
        section("🕳", "Missing Values by Column")
        miss_df = missing_per_col.reset_index()
        miss_df.columns = ["Column", "Missing"]
        miss_df["Pct"] = (miss_df["Missing"] / len(df) * 100).round(2)
        fig_miss = px.bar(miss_df, x="Column", y="Pct",
                          color="Pct",
                          color_continuous_scale=["#F97316", "#E8341C"],
                          labels={"Pct": "Missing (%)"})
        fig_miss.update_layout(coloraxis_showscale=False)
        card_chart(fig_miss, 280)
    else:
        st.markdown("""
        <div style='background:#161616;border:1px solid #2A2A2A;border-radius:12px;
                    padding:20px 24px;color:#F97316;font-weight:600;text-align:center;
                    margin-bottom:24px;'>
            ✅ No missing values detected — dataset is complete!
        </div>""", unsafe_allow_html=True)

    # ── Descriptive Statistics ──
    section("📐", "Descriptive Statistics")
    desc = df[num_cols].describe().T.round(3)
    desc.index.name = "Feature"
    st.dataframe(desc.reset_index(), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Distribution Explorer ──
    section("📊", "Distribution Explorer")
    col_l, col_r = st.columns(2)

    with col_l:
        sel_num = st.selectbox("Numeric feature", num_cols, key="eda_num")
        fig_hist = px.histogram(df, x=sel_num, nbins=35,
                                marginal="box",
                                color_discrete_sequence=["#F97316"])
        fig_hist.update_traces(marker_line_width=0)
        card_chart(fig_hist, 360)

    with col_r:
        if cat_cols:
            sel_cat = st.selectbox("Categorical feature", cat_cols, key="eda_cat")
            vc = df[sel_cat].value_counts().head(15).reset_index()
            vc.columns = [sel_cat, "Count"]
            fig_cat = px.bar(vc, x=sel_cat, y="Count",
                             color="Count",
                             color_continuous_scale=["#2A2A2A", "#E8341C", "#FBBF24"])
            fig_cat.update_layout(coloraxis_showscale=False)
            card_chart(fig_cat, 360)
        else:
            st.info("No categorical columns found.")

    # ── Correlation Heatmap ──
    section("🌡", "Correlation Heatmap")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().round(2)
        fig_corr = px.imshow(
            corr,
            color_continuous_scale=["#0D0D0D", "#E8341C", "#FBBF24"],
            zmin=-1, zmax=1,
            text_auto=True,
            aspect="auto")
        fig_corr.update_traces(textfont_size=11)
        card_chart(fig_corr, max(350, len(num_cols) * 45))

    # ── Scatter Matrix ──
    section("🔗", "Pairwise Scatter Explorer")
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        x_feat = st.selectbox("X axis", num_cols, index=0, key="scat_x")
    with col_r2:
        y_feat = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="scat_y")

    color_feat = None
    if "Will_Buy" in df.columns:
        color_feat = df["Will_Buy"].astype(str)

    fig_scat = px.scatter(
        df, x=x_feat, y=y_feat,
        color=color_feat,
        color_discrete_sequence=["#E8341C", "#FBBF24"],
        opacity=0.7,
        labels={"color": "Will Buy"})
    # Manual trendline via numpy
    x_vals = pd.to_numeric(df[x_feat], errors="coerce").dropna()
    y_vals = pd.to_numeric(df[y_feat], errors="coerce").loc[x_vals.index]
    if len(x_vals) > 1:
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig_scat.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines", name="Trend",
            line=dict(color="#F97316", width=2, dash="dash")))
    fig_scat.update_traces(marker=dict(size=7))
    card_chart(fig_scat, 400)

    # ── Boxplots by Target ──
    if "Will_Buy" in df.columns and len(num_cols) > 0:
        section("📦", "Feature Distribution by Purchase Intent")
        sel_box = st.selectbox("Feature to compare", num_cols, key="eda_box")
        df_box = df.copy()
        df_box["Will_Buy_Label"] = df_box["Will_Buy"].map({1: "Will Buy ✓", 0: "Won't Buy ✗"})
        fig_box = px.box(
            df_box, x="Will_Buy_Label", y=sel_box,
            color="Will_Buy_Label",
            color_discrete_sequence=["#E8341C", "#2A2A2A"],
            points="outliers")
        fig_box.update_layout(showlegend=False)
        card_chart(fig_box, 380)

    # ── Categorical breakdown vs target ──
    if "Will_Buy" in df.columns and cat_cols:
        section("🏷", "Categorical Breakdown vs Purchase Intent")
        sel_cat2 = st.selectbox("Categorical feature", cat_cols, key="eda_cat2")
        grp = df.groupby([sel_cat2, "Will_Buy"]).size().reset_index(name="Count")
        grp["Will_Buy_Label"] = grp["Will_Buy"].map({1: "Will Buy", 0: "Won't Buy"})
        fig_grp = px.bar(
            grp, x=sel_cat2, y="Count",
            color="Will_Buy_Label",
            barmode="group",
            color_discrete_sequence=["#E8341C", "#2A2A2A"],
            labels={"Will_Buy_Label": ""})
        card_chart(fig_grp, 380)

    # ── Outlier Detection ──
    section("⚠️", "Outlier Detection (IQR Method)")
    outlier_summary = []
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        outlier_summary.append({"Feature": col, "Outliers": int(n_out),
                                 "Outlier %": round(n_out/len(df)*100, 2),
                                 "Q1": round(Q1,2), "Q3": round(Q3,2), "IQR": round(IQR,2)})
    out_df = pd.DataFrame(outlier_summary).sort_values("Outliers", ascending=False)
    col_l3, col_r3 = st.columns([2, 3])
    with col_l3:
        st.dataframe(out_df, use_container_width=True)
    with col_r3:
        fig_out = px.bar(out_df[out_df["Outliers"]>0], x="Feature", y="Outlier %",
                         color="Outlier %",
                         color_continuous_scale=["#F97316","#E8341C"],
                         labels={"Outlier %": "Outlier %"})
        fig_out.update_layout(coloraxis_showscale=False)
        card_chart(fig_out, 320)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    section("🎯", "Purchase Prediction Model")

    if "Will_Buy" not in df_encoded.columns:
        st.warning("Column `Will_Buy` not found.")
    else:
        X = df_encoded.drop(columns=["Will_Buy"])
        if "Max_Spend" in X.columns:
            X = X.drop(columns=["Max_Spend"])
        y = df_encoded["Will_Buy"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42)

        clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(kpi("Accuracy",  f"{acc*100:.1f}%", "overall correct"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("Precision", f"{prec*100:.1f}%", "of predicted buyers"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("Recall",    f"{rec*100:.1f}%",  "buyers captured"), unsafe_allow_html=True)
        with c4: st.markdown(kpi("F1 Score",  f"{f1:.3f}", "harmonic mean"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        with col_l:
            section("📈", "ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}",
                line=dict(color="#F97316", width=3),
                fill="tozeroy", fillcolor="rgba(249,115,22,0.15)"))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines", name="Random",
                line=dict(color="#444", width=1.5, dash="dash")))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate")
            card_chart(fig_roc, 380)

        with col_r:
            section("🏆", "Feature Importance")
            fi = pd.DataFrame({
                "Feature": X.columns,
                "Importance": clf.feature_importances_
            }).sort_values("Importance", ascending=True).tail(12)
            fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                            color="Importance",
                            color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"])
            fig_fi.update_layout(coloraxis_showscale=False, yaxis_title="")
            card_chart(fig_fi, 380)

        # Probability distribution
        section("🔔", "Predicted Probability Distribution")
        fig_prob = px.histogram(
            x=y_prob, nbins=40, color_discrete_sequence=["#E8341C"],
            labels={"x": "Purchase Probability"})
        fig_prob.update_traces(marker_line_width=0)
        card_chart(fig_prob, 280)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    section("👥", "Customer Segmentation")

    X_clust = df_encoded.drop(columns=[c for c in ["Will_Buy","Max_Spend"] if c in df_encoded.columns])

    kmeans   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_clust)
    df_seg   = df.copy()
    df_seg["Segment"] = [f"Segment {i+1}" for i in clusters]

    pca   = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X_clust)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        section("🗺", "PCA Cluster Map")
        fig_pca = px.scatter(
            x=comps[:, 0], y=comps[:, 1],
            color=df_seg["Segment"],
            color_discrete_sequence=["#E8341C","#F97316","#FBBF24","#FB923C",
                                     "#FCA5A5","#FDE68A","#FED7AA","#FDBA74"],
            labels={"x": "PCA Component 1", "y": "PCA Component 2"},
            opacity=0.8)
        fig_pca.update_traces(marker=dict(size=8))
        card_chart(fig_pca, 400)

    with col_r:
        section("📊", "Segment Sizes")
        seg_counts = df_seg["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_bar = px.bar(
            seg_counts, x="Segment", y="Count",
            color="Segment",
            color_discrete_sequence=["#E8341C","#F97316","#FBBF24","#FB923C",
                                     "#FCA5A5","#FDE68A","#FED7AA","#FDBA74"])
        fig_bar.update_layout(showlegend=False)
        card_chart(fig_bar, 400)

    # Segment profile table
    section("📋", "Segment Profiles")
    profile = df_seg.groupby("Segment").mean(numeric_only=True).round(2).reset_index()
    st.dataframe(profile, use_container_width=True)

    if "Max_Spend" in df_seg.columns:
        section("💰", "Avg Spend by Segment")
        spend_seg = df_seg.groupby("Segment")["Max_Spend"].mean().reset_index()
        fig_sp = px.bar(
            spend_seg, x="Segment", y="Max_Spend",
            color="Max_Spend",
            color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"],
            labels={"Max_Spend": "Avg Max Spend (AED)"})
        fig_sp.update_layout(coloraxis_showscale=False)
        card_chart(fig_sp, 300)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SPEND FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    section("💰", "Spending Forecast Model")

    if "Max_Spend" not in df_encoded.columns:
        st.warning("Column `Max_Spend` not found.")
    else:
        y_reg = df_encoded["Max_Spend"]
        X_reg = df_encoded.drop(columns=[c for c in ["Max_Spend","Will_Buy"] if c in df_encoded.columns])

        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            X_reg, y_reg, test_size=test_size/100, random_state=42)

        reg = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        reg.fit(Xr_tr, yr_tr)
        preds = reg.predict(Xr_te)

        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(yr_te, preds)
        r2  = r2_score(yr_te, preds)
        rmse = np.sqrt(((preds - yr_te)**2).mean())

        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(kpi("R² Score", f"{r2:.3f}", "variance explained"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("MAE", f"AED {mae:,.0f}", "mean abs error"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("RMSE", f"AED {rmse:,.0f}", "root mean sq error"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        with col_l:
            section("🎯", "Actual vs Predicted")
            fig_av = go.Figure()
            fig_av.add_trace(go.Scatter(
                x=yr_te.values, y=preds, mode="markers",
                marker=dict(color="#F97316", size=7, opacity=0.7),
                name="Predictions"))
            mn, mx = float(yr_te.min()), float(yr_te.max())
            fig_av.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx], mode="lines",
                line=dict(color="#FBBF24", width=2, dash="dash"),
                name="Perfect fit"))
            fig_av.update_layout(
                xaxis_title="Actual Spend (AED)",
                yaxis_title="Predicted Spend (AED)")
            card_chart(fig_av, 380)

        with col_r:
            section("📉", "Residual Distribution")
            residuals = preds - yr_te.values
            fig_res = px.histogram(x=residuals, nbins=35,
                                   color_discrete_sequence=["#E8341C"],
                                   labels={"x": "Residual (AED)"})
            fig_res.add_vline(x=0, line_color="#FBBF24", line_dash="dash")
            card_chart(fig_res, 380)

        section("🏆", "Feature Importance — Spend")
        fi_r = pd.DataFrame({
            "Feature": X_reg.columns,
            "Importance": reg.feature_importances_
        }).sort_values("Importance", ascending=True).tail(12)
        fig_fir = px.bar(fi_r, x="Importance", y="Feature", orientation="h",
                         color="Importance",
                         color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"])
        fig_fir.update_layout(coloraxis_showscale=False, yaxis_title="")
        card_chart(fig_fir, 340)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT NEW
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    section("🔮", "Predict New Customers")

    st.markdown("""
    <div style='background:var(--card-bg);border:1px solid var(--card-border);
                border-radius:12px;padding:20px 24px;margin-bottom:24px;
                font-size:0.88rem;color:#888;'>
    Upload a CSV with the same columns as your training data
    (excluding <code>Will_Buy</code> and <code>Max_Spend</code>).
    </div>""", unsafe_allow_html=True)

    new_file = st.file_uploader("📂  Upload new customer data", key="new_pred", type=["csv"])

    if new_file and "Will_Buy" in df_encoded.columns:
        new_df = pd.read_csv(new_file)

        # Encode categoricals
        for col in new_df.columns:
            if col in le_dict:
                known = set(le_dict[col].classes_)
                new_df[col] = new_df[col].astype(str).apply(
                    lambda x: x if x in known else le_dict[col].classes_[0])
                new_df[col] = le_dict[col].transform(new_df[col].astype(str))

        # Align features
        X_new = new_df.reindex(columns=clf.feature_names_in_, fill_value=0)

        buy_pred  = clf.predict(X_new)
        buy_proba = clf.predict_proba(X_new)[:, 1]

        results = new_df.copy()
        results["Will_Buy_Prediction"] = buy_pred
        results["Buy_Probability"]     = (buy_proba * 100).round(1)

        # KPIs
        n_new     = len(results)
        n_buyers  = int(buy_pred.sum())
        avg_prob  = buy_proba.mean() * 100

        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(kpi("Customers Scored", f"{n_new:,}", "records"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("Predicted Buyers", f"{n_buyers:,}", f"{n_buyers/n_new*100:.1f}%"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("Avg Buy Prob", f"{avg_prob:.1f}%", "confidence"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        section("📊", "Buy Probability Distribution")
        fig_np = px.histogram(
            x=buy_proba*100, nbins=30,
            color_discrete_sequence=["#F97316"],
            labels={"x": "Buy Probability (%)"})
        fig_np.add_vline(x=50, line_color="#FBBF24", line_dash="dash",
                         annotation_text="50% threshold")
        card_chart(fig_np, 280)

        section("📋", "Prediction Results")
        st.dataframe(results, use_container_width=True)

        csv_out = results.to_csv(index=False).encode()
        st.download_button(
            "⬇  Download Predictions CSV",
            csv_out,
            "predictions.csv",
            "text/csv")
    elif new_file:
        st.warning("Train the classification model first (upload your main dataset above).")

# ══════════════════════════════════════════════════════════════════════════════
# TAB DIAGNOSTIC — WHY IS IT HAPPENING?
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    section("🩺", "Diagnostic Analysis — Why Is It Happening?")
    st.markdown("""
    <div style='background:var(--card-bg);border-left:4px solid #E8341C;border-radius:8px;
                padding:16px 20px;margin-bottom:28px;color:#aaa;font-size:0.9rem;line-height:1.7'>
    <b style='color:#F5F5F5'>What is Diagnostic Analysis?</b><br>
    Diagnostic analysis goes beyond describing what happened — it investigates <em>why</em> it happened.
    Here we drill into the root causes of purchase intent and spending behaviour across customer
    segments, channels, and demographics to uncover the key drivers behind conversion and revenue.
    </div>""", unsafe_allow_html=True)

    num_cols_d = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_d = df.select_dtypes(include="object").columns.tolist()

    if "Will_Buy" in df.columns and "Max_Spend" in df.columns:
        buyers_df    = df[df["Will_Buy"] == 1]
        non_buyers   = df[df["Will_Buy"] == 0]
        avg_spend_b  = buyers_df["Max_Spend"].mean()
        avg_spend_nb = non_buyers["Max_Spend"].mean()
        spend_lift   = ((avg_spend_b - avg_spend_nb) / avg_spend_nb * 100)
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(kpi("Buyer Avg Spend",     f"AED {avg_spend_b:,.0f}",  "who will buy"),       unsafe_allow_html=True)
        with c2: st.markdown(kpi("Non-Buyer Avg Spend", f"AED {avg_spend_nb:,.0f}", "who won't buy"),      unsafe_allow_html=True)
        with c3: st.markdown(kpi("Spend Lift",          f"+{spend_lift:.1f}%",      "buyers vs non-buyers"), unsafe_allow_html=True)
        with c4: st.markdown(kpi("Buy Rate",            f"{len(buyers_df)/len(df)*100:.1f}%", "overall"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    section("📊", "Purchase Rate by Key Segment")
    if "Will_Buy" in df.columns and cat_cols_d:
        diag_cat = st.selectbox("Select segment to diagnose", cat_cols_d, key="diag_cat")
        grp = df.groupby(diag_cat)["Will_Buy"].agg(["mean","count"]).reset_index()
        grp.columns = [diag_cat,"Buy_Rate","Count"]
        grp["Buy_Rate_Pct"] = (grp["Buy_Rate"]*100).round(1)
        grp = grp.sort_values("Buy_Rate_Pct", ascending=False)
        col_l, col_r = st.columns(2)
        with col_l:
            fig_dr = px.bar(grp, x=diag_cat, y="Buy_Rate_Pct",
                            color="Buy_Rate_Pct",
                            color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"],
                            labels={"Buy_Rate_Pct":"Buy Rate (%)"})
            fig_dr.update_layout(coloraxis_showscale=False)
            card_chart(fig_dr, 360)
        with col_r:
            fig_cnt = px.bar(grp, x=diag_cat, y="Count",
                             color="Count",
                             color_continuous_scale=["#2A2A2A","#F97316","#FBBF24"],
                             labels={"Count":"Number of Customers"})
            fig_cnt.update_layout(coloraxis_showscale=False)
            card_chart(fig_cnt, 360)

    if "Preferred_Channel" in df.columns and "Max_Spend" in df.columns:
        section("🛒", "Average Spend by Shopping Channel")
        ch = df.groupby("Preferred_Channel")["Max_Spend"].mean().reset_index()
        ch.columns = ["Channel","Avg_Spend"]
        fig_ch = px.bar(ch, x="Channel", y="Avg_Spend",
                        color="Avg_Spend",
                        color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"],
                        labels={"Avg_Spend":"Avg Max Spend (AED)"})
        fig_ch.update_layout(coloraxis_showscale=False)
        card_chart(fig_ch, 300)

    if "App_Sessions_Per_Week" in df.columns and "Will_Buy" in df.columns:
        section("📱", "Digital Engagement vs Purchase Intent")
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            df_d = df.copy()
            df_d["Engagement_Band"] = pd.cut(df_d["App_Sessions_Per_Week"],
                bins=[0,5,10,20,30], labels=["Low (0-5)","Medium (6-10)","High (11-20)","Very High (21+)"])
            eng = df_d.groupby("Engagement_Band", observed=True)["Will_Buy"].mean().reset_index()
            eng["Buy_Rate"] = (eng["Will_Buy"]*100).round(1)
            fig_eng = px.bar(eng, x="Engagement_Band", y="Buy_Rate",
                             color="Buy_Rate",
                             color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"],
                             labels={"Buy_Rate":"Buy Rate (%)","Engagement_Band":"App Sessions/Week"})
            fig_eng.update_layout(coloraxis_showscale=False)
            card_chart(fig_eng, 320)
        with col_r2:
            if "Brand_Loyalty_Score" in df.columns:
                df_d["Loyalty_Band"] = pd.cut(df_d["Brand_Loyalty_Score"],
                    bins=[0,3,6,8,10], labels=["Low (1-3)","Medium (4-6)","High (7-8)","Very High (9-10)"])
                loy = df_d.groupby("Loyalty_Band", observed=True)["Will_Buy"].mean().reset_index()
                loy["Buy_Rate"] = (loy["Will_Buy"]*100).round(1)
                fig_loy = px.bar(loy, x="Loyalty_Band", y="Buy_Rate",
                                 color="Buy_Rate",
                                 color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"],
                                 labels={"Buy_Rate":"Buy Rate (%)","Loyalty_Band":"Brand Loyalty Score"})
                fig_loy.update_layout(coloraxis_showscale=False)
                card_chart(fig_loy, 320)

    if all(c in df.columns for c in ["Shops_During_Ramadan","Shops_During_DSF","Will_Buy"]):
        section("🌙", "Cultural Event Impact on Purchase Intent")
        col_l3, col_r3 = st.columns(2)
        with col_l3:
            ram = df.groupby("Shops_During_Ramadan")["Will_Buy"].mean().reset_index()
            ram["Label"] = ram["Shops_During_Ramadan"].map({1:"Shops During Ramadan",0:"Does Not"})
            ram["Buy_Rate"] = (ram["Will_Buy"]*100).round(1)
            fig_ram = px.bar(ram, x="Label", y="Buy_Rate",
                             color="Buy_Rate",
                             color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"],
                             labels={"Buy_Rate":"Buy Rate (%)"})
            fig_ram.update_layout(coloraxis_showscale=False, title="Ramadan Shoppers")
            card_chart(fig_ram, 300)
        with col_r3:
            dsf = df.groupby("Shops_During_DSF")["Will_Buy"].mean().reset_index()
            dsf["Label"] = dsf["Shops_During_DSF"].map({1:"Shops During DSF",0:"Does Not"})
            dsf["Buy_Rate"] = (dsf["Will_Buy"]*100).round(1)
            fig_dsf = px.bar(dsf, x="Label", y="Buy_Rate",
                             color="Buy_Rate",
                             color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"],
                             labels={"Buy_Rate":"Buy Rate (%)"})
            fig_dsf.update_layout(coloraxis_showscale=False, title="DSF Shoppers")
            card_chart(fig_dsf, 300)

    if "Monthly_Income_AED" in df.columns and "Max_Spend" in df.columns:
        section("💰", "Income vs Spending Capacity")
        color_diag = df["Will_Buy"].astype(str) if "Will_Buy" in df.columns else None
        fig_inc = px.scatter(df, x="Monthly_Income_AED", y="Max_Spend",
                             color=color_diag,
                             color_discrete_sequence=["#E8341C","#FBBF24"],
                             opacity=0.5,
                             labels={"Monthly_Income_AED":"Monthly Income (AED)",
                                     "Max_Spend":"Max Spend (AED)","color":"Will Buy"})
        fig_inc.update_traces(marker=dict(size=5))
        card_chart(fig_inc, 400)


# ══════════════════════════════════════════════════════════════════════════════
# TAB PREDICTIVE — WHAT WILL HAPPEN?
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    section("📈", "Predictive Analysis — What Will Happen?")
    st.markdown("""
    <div style='background:var(--card-bg);border-left:4px solid #F97316;border-radius:8px;
                padding:16px 20px;margin-bottom:28px;color:#aaa;font-size:0.9rem;line-height:1.7'>
    <b style='color:#F5F5F5'>What is Predictive Analysis?</b><br>
    Predictive analysis uses historical data and machine learning to forecast future outcomes.
    Here we predict <em>which customers are likely to buy</em>, <em>how much they will spend</em>,
    and identify the features that most strongly signal future purchasing behaviour.
    </div>""", unsafe_allow_html=True)

    if "Will_Buy" not in df_encoded.columns:
        st.warning("Column Will_Buy not found.")
    else:
        from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix

        X_p = df_encoded.drop(columns=[c for c in ["Will_Buy","Max_Spend"] if c in df_encoded.columns])
        y_p = df_encoded["Will_Buy"]
        Xp_tr,Xp_te,yp_tr,yp_te = train_test_split(X_p, y_p, test_size=0.2, random_state=42)
        clf_p = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_p.fit(Xp_tr, yp_tr)
        yp_pred  = clf_p.predict(Xp_te)
        yp_proba = clf_p.predict_proba(Xp_te)[:,1]

        acc_p  = accuracy_score(yp_te, yp_pred)
        prec_p = precision_score(yp_te, yp_pred, zero_division=0)
        rec_p  = recall_score(yp_te, yp_pred, zero_division=0)
        f1_p   = f1_score(yp_te, yp_pred, zero_division=0)

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(kpi("Model Accuracy", f"{acc_p*100:.1f}%",  "correctly predicted"),    unsafe_allow_html=True)
        with c2: st.markdown(kpi("Precision",       f"{prec_p*100:.1f}%","predicted buyers correct"),unsafe_allow_html=True)
        with c3: st.markdown(kpi("Recall",          f"{rec_p*100:.1f}%", "actual buyers caught"),    unsafe_allow_html=True)
        with c4: st.markdown(kpi("F1 Score",        f"{f1_p:.3f}",       "harmonic mean"),           unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            section("📉", "ROC Curve")
            fpr_p,tpr_p,_ = roc_curve(yp_te, yp_proba)
            roc_p = auc(fpr_p, tpr_p)
            fig_rp = go.Figure()
            fig_rp.add_trace(go.Scatter(x=fpr_p, y=tpr_p, mode="lines",
                name=f"AUC = {roc_p:.3f}",
                line=dict(color="#F97316", width=3),
                fill="tozeroy", fillcolor="rgba(249,115,22,0.15)"))
            fig_rp.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                name="Random baseline", line=dict(color="#444", width=1.5, dash="dash")))
            fig_rp.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            card_chart(fig_rp, 360)
        with col_r:
            section("🟥", "Confusion Matrix")
            from sklearn.metrics import confusion_matrix as cm_fn
            cm_vals = cm_fn(yp_te, yp_pred)
            fig_cm = px.imshow(cm_vals,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Predicted No","Predicted Yes"],
                y=["Actual No","Actual Yes"],
                color_continuous_scale=["#0D0D0D","#E8341C","#FBBF24"],
                text_auto=True)
            fig_cm.update_traces(textfont_size=18)
            card_chart(fig_cm, 360)

        section("🏆", "Top Predictive Features — Purchase Intent")
        fi_p = pd.DataFrame({"Feature":X_p.columns,"Importance":clf_p.feature_importances_})
        fi_p = fi_p.sort_values("Importance", ascending=True).tail(15)
        fig_fi_p = px.bar(fi_p, x="Importance", y="Feature", orientation="h",
                          color="Importance",
                          color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"])
        fig_fi_p.update_layout(coloraxis_showscale=False, yaxis_title="")
        card_chart(fig_fi_p, 420)

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            section("🔔", "Buy Probability Distribution")
            fig_pb = px.histogram(x=yp_proba*100, nbins=40,
                color_discrete_sequence=["#E8341C"],
                labels={"x":"Buy Probability (%)"})
            fig_pb.add_vline(x=50, line_color="#FBBF24", line_dash="dash",
                             annotation_text="Decision threshold")
            card_chart(fig_pb, 300)
        with col_r2:
            if "Max_Spend" in df_encoded.columns:
                section("💸", "Spend Forecast: Actual vs Predicted")
                y_sp = df_encoded["Max_Spend"]
                X_sp = df_encoded.drop(columns=[c for c in ["Max_Spend","Will_Buy"] if c in df_encoded.columns])
                Xs_tr,Xs_te,ys_tr,ys_te = train_test_split(X_sp, y_sp, test_size=0.2, random_state=42)
                reg_p = RandomForestRegressor(n_estimators=100, random_state=42)
                reg_p.fit(Xs_tr, ys_tr)
                sp_preds = reg_p.predict(Xs_te)
                r2_sp  = r2_score(ys_te, sp_preds)
                mae_sp = mean_absolute_error(ys_te, sp_preds)
                fig_sp2 = go.Figure()
                fig_sp2.add_trace(go.Scatter(x=ys_te.values, y=sp_preds, mode="markers",
                    marker=dict(color="#F97316", size=5, opacity=0.6), name="Predictions"))
                mn2,mx2 = float(ys_te.min()), float(ys_te.max())
                fig_sp2.add_trace(go.Scatter(x=[mn2,mx2], y=[mn2,mx2], mode="lines",
                    line=dict(color="#FBBF24", width=2, dash="dash"), name="Perfect fit"))
                fig_sp2.update_layout(title=f"R²={r2_sp:.3f}  |  MAE=AED {mae_sp:,.0f}",
                    xaxis_title="Actual (AED)", yaxis_title="Predicted (AED)")
                card_chart(fig_sp2, 300)

        section("⭐", "Top 20 High-Intent, High-Value Customers")
        pred_all  = clf_p.predict_proba(X_p)[:,1]
        df_ranked = df.copy()
        df_ranked["Buy_Probability_%"] = (pred_all*100).round(1)
        if "Max_Spend" in df_ranked.columns:
            df_ranked["Revenue_Potential_AED"] = (df_ranked["Buy_Probability_%"]/100 * df_ranked["Max_Spend"]).round(0)
            show_cols = [c for c in ["Age","Gender","City","Preferred_Category","Monthly_Income_AED",
                                      "Buy_Probability_%","Max_Spend","Revenue_Potential_AED"] if c in df_ranked.columns]
            top20 = df_ranked.nlargest(20, "Revenue_Potential_AED")[show_cols]
            st.dataframe(top20, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB PRESCRIPTIVE — WHAT SHOULD WE DO?
# ══════════════════════════════════════════════════════════════════════════════
with tab_presc:
    section("🧭", "Prescriptive Analysis — What Should We Do?")
    st.markdown("""
    <div style='background:var(--card-bg);border-left:4px solid #FBBF24;border-radius:8px;
                padding:16px 20px;margin-bottom:28px;color:#aaa;font-size:0.9rem;line-height:1.7'>
    <b style='color:#F5F5F5'>What is Prescriptive Analysis?</b><br>
    Prescriptive analysis recommends <em>specific actions</em> a business should take based on
    the patterns found in diagnostic and predictive analysis. Here we translate data insights
    into concrete, actionable GEO-informed marketing strategies for UAE customer segments.
    </div>""", unsafe_allow_html=True)

    if "Will_Buy" not in df_encoded.columns:
        st.warning("Column Will_Buy not found.")
    else:
        X_pr = df_encoded.drop(columns=[c for c in ["Will_Buy","Max_Spend"] if c in df_encoded.columns])
        clf_pr = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_pr.fit(X_pr, df_encoded["Will_Buy"])
        all_proba = clf_pr.predict_proba(X_pr)[:,1]

        df_pr = df.copy()
        df_pr["Buy_Probability"] = all_proba
        med_prob  = df_pr["Buy_Probability"].median()
        med_spend = df_pr["Max_Spend"].median() if "Max_Spend" in df_pr.columns else 5000

        def assign_quadrant(row):
            hp = row["Buy_Probability"] >= med_prob
            hs = row.get("Max_Spend", 0) >= med_spend
            if hp and hs:     return "High Intent · High Value"
            if hp and not hs: return "High Intent · Budget"
            if not hp and hs: return "Low Intent · High Value"
            return "Low Intent · Low Value"

        df_pr["Strategy_Segment"] = df_pr.apply(assign_quadrant, axis=1)
        seg_counts = df_pr["Strategy_Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Count"]

        section("🗺", "Customer Strategy Matrix")
        col_l, col_r = st.columns([2,3])
        icons_map = {
            "High Intent · High Value":"🏆",
            "High Intent · Budget":"🎯",
            "Low Intent · High Value":"💎",
            "Low Intent · Low Value":"💤"
        }
        with col_l:
            for _, row in seg_counts.iterrows():
                pct = row["Count"]/len(df_pr)*100
                ic = icons_map.get(row["Segment"],"•")
                st.markdown(f"""
                <div class='kpi-card' style='margin-bottom:12px;text-align:left;padding:16px 18px;'>
                    <div style='font-size:1rem;font-weight:700;color:#F5F5F5;margin-bottom:4px'>{ic} {row['Segment']}</div>
                    <div style='color:#F97316;font-family:Syne;font-size:1.5rem;font-weight:800'>{row['Count']:,}</div>
                    <div style='color:#666;font-size:0.78rem'>{pct:.1f}% of customers</div>
                </div>""", unsafe_allow_html=True)
        with col_r:
            fig_quad = px.scatter(df_pr,
                x="Buy_Probability", y="Max_Spend",
                color="Strategy_Segment",
                color_discrete_sequence=["#E8341C","#F97316","#FBBF24","#444"],
                opacity=0.65,
                labels={"Buy_Probability":"Buy Probability","Max_Spend":"Max Spend (AED)"})
            fig_quad.add_vline(x=med_prob, line_color="#555", line_dash="dash",
                               annotation_text="Median Probability")
            fig_quad.add_hline(y=med_spend, line_color="#555", line_dash="dash",
                               annotation_text="Median Spend")
            fig_quad.update_traces(marker=dict(size=5))
            card_chart(fig_quad, 420)

        section("💡", "GEO-Informed Marketing Recommendations")
        recs = [
            ("🏆","High Intent · High Value","Priority Tier — Maximum Investment","#E8341C",[
                "Deploy premium AI-optimised product content targeting queries like 'best [category] in Dubai' on ChatGPT and Gemini",
                "Offer exclusive early access to DSF and Ramadan deals with personalised AED spend thresholds",
                "Activate loyalty rewards and referral multipliers — these customers have highest NPS and referral potential",
                "Target with high-production influencer content on Instagram and TikTok to stay visible in AI training data",
            ]),
            ("🎯","High Intent · Budget","Volume Tier — Convert with Value Offers","#F97316",[
                "Promote Buy Now Pay Later (BNPL) prominently — removes affordability barrier for high-intent buyers",
                "Use discount-led email campaigns and push notifications timed around pay cycle (end of month)",
                "Optimise product listings for AI answer engines with budget-friendly search queries",
                "Create social proof content — this segment researches heavily before converting",
            ]),
            ("💎","Low Intent · High Value","Re-Engagement Tier — Win Back with Personalisation","#FBBF24",[
                "Launch personalised re-engagement campaigns referencing past categories and browsing behaviour",
                "Invest in GEO content for premium product categories — these customers have spending capacity but need discovery triggers",
                "Use Ramadan and DSF as re-engagement windows with exclusive high-value offers",
                "Deploy retargeting ads on LinkedIn and YouTube where this segment is most active",
            ]),
            ("💤","Low Intent · Low Value","Nurture Tier — Low-Cost Awareness Only","#555555",[
                "Maintain brand awareness via low-cost channels only (organic social, email newsletters)",
                "Do not allocate significant paid ad budget — ROI is too low for active conversion campaigns",
                "Monitor for behaviour signals indicating segment migration (increased app sessions, email opens)",
                "Use as seed audience for lookalike modelling to find similar but higher-intent new prospects",
            ]),
        ]
        for icon, title, subtitle, color, actions in recs:
            st.markdown(f"""
            <div style='background:var(--card-bg);border:1px solid {color};border-radius:12px;
                        padding:22px 26px;margin-bottom:20px;'>
                <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px'>
                    <span style='font-size:1.8rem'>{icon}</span>
                    <div>
                        <div style='font-family:Syne;font-size:1.05rem;font-weight:700;color:#F5F5F5'>{title}</div>
                        <div style='font-size:0.75rem;color:{color};font-weight:600;
                                    text-transform:uppercase;letter-spacing:0.8px'>{subtitle}</div>
                    </div>
                </div>
                <ul style='margin:10px 0 0 0;padding-left:20px;color:#aaa;font-size:0.88rem;line-height:2.0'>
                    {''.join(f"<li>{a}</li>" for a in actions)}
                </ul>
            </div>""", unsafe_allow_html=True)

        section("💰", "Recommended Marketing Budget Allocation")
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            budget_data = pd.DataFrame({
                "Segment":  ["High Intent High Value","High Intent Budget","Low Intent High Value","Low Intent Low Value"],
                "Budget_%": [45, 30, 20, 5],
            })
            fig_bud = px.pie(budget_data, names="Segment", values="Budget_%",
                             color_discrete_sequence=["#E8341C","#F97316","#FBBF24","#333"],
                             hole=0.52)
            fig_bud.update_traces(textfont_color="white", pull=[0.05,0.03,0.02,0])
            card_chart(fig_bud, 380)
        with col_r2:
            section("📡", "Channel Priority by Segment (1=Low, 5=High)")
            channel_df = pd.DataFrame({
                "Channel":                  ["Influencer/TikTok","Email Campaigns","BNPL Promotions","DSF/Ramadan Events","Retargeting Ads","GEO/AI Content"],
                "High Intent High Value":   [5, 4, 2, 5, 3, 5],
                "High Intent Budget":       [3, 5, 5, 4, 4, 3],
                "Low Intent High Value":    [3, 4, 2, 5, 5, 4],
                "Low Intent Low Value":     [1, 2, 1, 2, 1, 3],
            })
            fig_ch2 = px.bar(channel_df, x="Channel",
                             y=["High Intent High Value","High Intent Budget","Low Intent High Value","Low Intent Low Value"],
                             barmode="group",
                             color_discrete_sequence=["#E8341C","#F97316","#FBBF24","#444"],
                             labels={"value":"Priority (1-5)","variable":"Segment"})
            card_chart(fig_ch2, 380)
