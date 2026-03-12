"""
╔══════════════════════════════════════════════════════════════╗
║     FOOTBALL ANALYTICS — STREAMLIT INTERACTIVE WEB APP       ║
║  Integrates with sports_ml_real.py                          ║
║  Run: streamlit run football_app.py                         ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests
import warnings
import time
import random
from io import StringIO
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import Ridge, LinearRegression
from sklearn.cluster          import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics          import (accuracy_score, confusion_matrix,
                                      r2_score, mean_squared_error)
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Football Analytics ML",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS — dark football theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1424 100%);
    color: #e8eaf6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2a3a;
}
section[data-testid="stSidebar"] * { color: #e8eaf6 !important; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a2540 0%, #0d1424 100%);
    border: 1px solid #1e2a3a;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '⚽';
    position: absolute;
    right: 32px; top: 50%;
    transform: translateY(-50%);
    font-size: 80px;
    opacity: 0.08;
}
.hero h1 {
    font-family: 'Oswald', sans-serif;
    font-size: 2.4rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    background: linear-gradient(90deg, #4f8ef7, #4fffc8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.hero p { color: #6b7a8d; font-size: 14px; margin: 0; }

/* KPI cards */
.kpi-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.kpi {
    flex: 1; min-width: 120px;
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.kpi-val {
    font-family: 'Oswald', sans-serif;
    font-size: 2rem;
    background: linear-gradient(135deg, #4f8ef7, #4fffc8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-lbl { color: #6b7a8d; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }

/* Section headers */
.section-title {
    font-family: 'Oswald', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4f8ef7;
    border-bottom: 1px solid #1e2a3a;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Result box */
.result-box {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
}
.result-box.win  { border-color: #10b981; }
.result-box.draw { border-color: #f59e0b; }
.result-box.loss { border-color: #ef4444; }

/* Player card */
.player-card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    transition: border-color .2s;
}
.player-card:hover { border-color: #4f8ef7; }

/* Tag badge */
.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .5px;
    margin: 2px;
}
.tag-fw { background: rgba(239,68,68,.15);  color: #ef4444; }
.tag-mf { background: rgba(79,142,247,.15); color: #4f8ef7; }
.tag-df { background: rgba(16,185,129,.15); color: #10b981; }
.tag-gk { background: rgba(245,158,11,.15); color: #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  DATA LOADING — cached so it only runs once
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    """Download or generate football data — same logic as sports_ml_real.py"""

    def fetch(url):
        try:
            r = requests.get(url, timeout=15,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return pd.read_csv(StringIO(r.text))
        except:
            return None

    FIFA_URLS = [
        "https://raw.githubusercontent.com/305kishan/FIFA/main/data/FIFA20.csv",
        "https://raw.githubusercontent.com/ddebonis47/classwork/refs/heads/main/fifa21%20raw%20data%20v2.csv",
        "https://raw.githubusercontent.com/apoorva-21/fifa-analysis/master/data/players_20.csv",
    ]
    EPL_URLS = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    ]

    # ── Try to fetch FIFA data ─────────────────────────────────
    fifa_raw = None
    for url in FIFA_URLS:
        fifa_raw = fetch(url)
        if fifa_raw is not None:
            break

    # ── Try to fetch EPL data ──────────────────────────────────
    epl_raw = None
    for url in EPL_URLS:
        epl_raw = fetch(url)
        if epl_raw is not None:
            break

    # ── Process player data ────────────────────────────────────
    if fifa_raw is not None:
        p = fifa_raw.copy()
        col_map = {}
        for c in p.columns:
            cl = c.lower().strip()
            if cl in ("short_name","name","player_name"): col_map[c] = col_map.get("Name", "Name"); col_map[c] = "Name"
            if cl in ("nationality","nationality_name"):  col_map[c] = "Nationality"
            if cl in ("player_positions","position","pos"): col_map[c] = "Position"
            if cl == "age":                               col_map[c] = "Age"
            if cl == "overall":                           col_map[c] = "Overall"
            if cl in ("pace","pac"):                      col_map[c] = "Pace"
            if cl in ("shooting","sho"):                  col_map[c] = "Shooting"
            if cl in ("passing","pas"):                   col_map[c] = "Passing"
            if cl in ("dribbling","dri"):                 col_map[c] = "Dribbling"
            if cl in ("defending","def"):                 col_map[c] = "Defending"
            if cl in ("physic","physical","phy"):         col_map[c] = "Physical"
            if cl in ("value_eur","value"):               col_map[c] = "Value_M"
        p.rename(columns=col_map, inplace=True)
        for col in ["Age","Overall","Pace","Shooting","Passing","Dribbling","Defending","Physical"]:
            if col in p.columns:
                p[col] = pd.to_numeric(p[col], errors="coerce")
        if "Position" in p.columns:
            def clean_pos(v):
                if pd.isna(v): return "Midfielder"
                v = str(v).split(",")[0].strip().upper()
                if v in ("ST","CF","LW","RW","LF","RF"): return "Forward"
                if v in ("CAM","CM","CDM","LM","RM"):    return "Midfielder"
                if v in ("CB","LB","RB","LWB","RWB"):    return "Defender"
                if v == "GK":                            return "Goalkeeper"
                return "Midfielder"
            p["Position"] = p["Position"].apply(clean_pos)
        n = len(p)
        for col, fn in [("Goals",    lambda: np.random.poisson(5,n)),
                        ("Assists",  lambda: np.random.poisson(3,n)),
                        ("Pace",     lambda: np.random.randint(50,98,n)),
                        ("Shooting", lambda: np.random.randint(45,95,n)),
                        ("Passing",  lambda: np.random.randint(50,95,n)),
                        ("Dribbling",lambda: np.random.randint(45,97,n)),
                        ("Defending",lambda: np.random.randint(30,90,n)),
                        ("Physical", lambda: np.random.randint(50,95,n)),
                        ("Value_M",  lambda: np.round(np.random.exponential(20,n).clip(1,200),1)),
                        ("Position", lambda: np.random.choice(["Forward","Midfielder","Defender","Goalkeeper"],n))]:
            if col not in p.columns: p[col] = fn()
        overall = pd.to_numeric(p.get("Overall", 70), errors="coerce").fillna(70)
        p["Performance"] = (
            p["Goals"]*6 + p["Assists"]*4 + overall*0.8 +
            np.random.normal(0,8,n)
        ).round(1)
        p.dropna(subset=["Overall"], inplace=True)
        p = p.sample(min(500,len(p)), random_state=42).reset_index(drop=True)
        data_source = "FIFA (real)"
    else:
        # Fallback
        n = 300
        POSITIONS = ["Forward","Midfielder","Defender","Goalkeeper"]
        p = pd.DataFrame({
            "Name":        [f"Player_{i:03d}" for i in range(1,n+1)],
            "Nationality": np.random.choice(["England","Spain","France","Germany","Brazil"],n),
            "Position":    np.random.choice(POSITIONS,n,p=[0.28,0.35,0.27,0.10]),
            "Age":         np.random.randint(17,37,n),
            "Overall":     np.random.randint(60,95,n),
            "Goals":       np.random.poisson(7,n),
            "Assists":     np.random.poisson(4,n),
            "Pace":        np.random.randint(50,98,n),
            "Shooting":    np.random.randint(45,95,n),
            "Passing":     np.random.randint(50,95,n),
            "Dribbling":   np.random.randint(45,97,n),
            "Defending":   np.random.randint(30,90,n),
            "Physical":    np.random.randint(50,95,n),
            "Value_M":     np.round(np.random.exponential(20,n).clip(1,200),1),
        })
        p["Performance"] = (
            p["Goals"]*6 + p["Assists"]*4 + p["Overall"]*0.8 + np.random.normal(0,8,n)
        ).round(1)
        data_source = "Generated"

    # ── Process match data ─────────────────────────────────────
    if epl_raw is not None:
        m = epl_raw.copy()
        for col in ["FTHG","FTAG","HS","AS","HST","AST"]:
            if col in m.columns:
                m[col] = pd.to_numeric(m[col], errors="coerce")
        m.dropna(subset=["FTHG","FTAG","FTR"], inplace=True)
        nm = len(m)
        for col, fn in [("HC", lambda: np.random.randint(2,12,nm)),
                        ("AC", lambda: np.random.randint(1,11,nm)),
                        ("HF", lambda: np.random.randint(5,18,nm)),
                        ("AF", lambda: np.random.randint(5,18,nm)),
                        ("HY", lambda: np.random.randint(0, 4,nm)),
                        ("AY", lambda: np.random.randint(0, 4,nm))]:
            if col not in m.columns: m[col] = fn()
        match_source = "EPL (real)"
    else:
        teams = ["Arsenal","Chelsea","Liverpool","Man City","Man Utd","Tottenham",
                 "Newcastle","Aston Villa","West Ham","Brighton","Brentford",
                 "Fulham","Crystal Palace","Wolves","Everton","Nottm Forest",
                 "Bournemouth","Luton","Sheffield Utd","Burnley"]
        nm = 380
        hg = np.random.poisson(1.5,nm); ag = np.random.poisson(1.2,nm)
        m = pd.DataFrame({
            "HomeTeam": np.random.choice(teams,nm),
            "AwayTeam": np.random.choice(teams,nm),
            "FTHG":hg,"FTAG":ag,
            "FTR": np.where(hg>ag,"H",np.where(hg<ag,"A","D")),
            "HS": np.random.randint(5,22,nm), "AS": np.random.randint(3,20,nm),
            "HST":np.random.randint(2,10,nm), "AST":np.random.randint(1,9,nm),
            "HC": np.random.randint(2,12,nm), "AC": np.random.randint(1,11,nm),
            "HF": np.random.randint(5,18,nm), "AF": np.random.randint(5,18,nm),
            "HY": np.random.randint(0,4,nm),  "AY": np.random.randint(0,4,nm),
        })
        match_source = "Generated"

    return p, m, data_source, match_source

@st.cache_resource(show_spinner=False)
def train_models(players_df, matches_df):
    """Train all ML models — cached so they only train once"""
    models = {}

    # ── Match predictor ────────────────────────────────────────
    feat = [c for c in ["HS","AS","HST","AST","HC","AC","HF","AF","HY","AY"]
            if c in matches_df.columns]
    X = matches_df[feat].dropna()
    le = LabelEncoder()
    y  = le.fit_transform(matches_df.loc[X.index,"FTR"])
    Xt,Xe,yt,ye = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    clf = RandomForestClassifier(n_estimators=200,max_depth=6,
                                  random_state=42,class_weight="balanced")
    clf.fit(Xt,yt)
    acc = accuracy_score(ye, clf.predict(Xe))
    models["match_clf"]   = clf
    models["match_le"]    = le
    models["match_feats"] = feat
    models["match_acc"]   = acc

    # ── Performance scorer ─────────────────────────────────────
    pf = [c for c in ["Goals","Assists","Pace","Shooting","Passing",
                       "Dribbling","Defending","Physical"]
          if c in players_df.columns]
    pdf = players_df[pf+["Performance"]].dropna()
    sc  = StandardScaler()
    Xp  = sc.fit_transform(pdf[pf])
    yp  = pdf["Performance"]
    Xpt,Xpe,ypt,ype = train_test_split(Xp,yp,test_size=0.25,random_state=42)
    reg = Ridge(alpha=1.0)
    reg.fit(Xpt,ypt)
    r2  = r2_score(ype, reg.predict(Xpe))
    models["perf_reg"]    = reg
    models["perf_scaler"] = sc
    models["perf_feats"]  = pf
    models["perf_r2"]     = r2

    # ── Clustering ─────────────────────────────────────────────
    cf = [c for c in ["Goals","Assists","Passing","Dribbling",
                       "Defending","Physical","Pace"]
          if c in players_df.columns]
    cd   = players_df[cf].dropna()
    scc  = StandardScaler()
    cds  = scc.fit_transform(cd)
    km   = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(cds)
    cnames = {0:"🛡️ Defensive Wall",1:"⚡ Speed Merchant",
               2:"🎯 Goal Machine",3:"🎪 Playmaker"}
    players_df = players_df.copy()
    players_df.loc[cd.index,"Cluster"]      = labels
    players_df.loc[cd.index,"Cluster_Name"] = [cnames[l] for l in labels]
    models["cluster_km"]     = km
    models["cluster_scaler"] = scc
    models["cluster_feats"]  = cf
    models["cluster_names"]  = cnames
    models["players_df"]     = players_df

    # ── Recommender ────────────────────────────────────────────
    rf = [c for c in ["Goals","Assists","Pace","Shooting","Passing",
                       "Dribbling","Defending","Physical"]
          if c in players_df.columns]
    rd   = players_df[rf+["Performance"]+(["Name"] if "Name" in players_df.columns else [])].dropna().reset_index(drop=True)
    scr  = MinMaxScaler()
    rds  = scr.fit_transform(rd[rf])
    sim  = cosine_similarity(rds)
    models["rec_data"]   = rd
    models["rec_sim"]    = sim
    models["rec_feats"]  = rf
    models["rec_scaler"] = scr

    return models

# ══════════════════════════════════════════════════════════════
#  LOAD DATA + TRAIN
# ══════════════════════════════════════════════════════════════
with st.spinner("🔄 Loading data & training models..."):
    players_df, matches_df, data_src, match_src = load_data()
    models = train_models(players_df, matches_df)
    players_df = models["players_df"]

# ══════════════════════════════════════════════════════════════
#  SIDEBAR NAV
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚽ Football Analytics ML")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Dashboard",
        "🏆 Match Predictor",
        "📈 Performance Scorer",
        "🗂️ Player Clusters",
        "🤖 Player Recommender",
        "🌍 EDA Explorer",
        "📊 League Table",
    ])
    st.markdown("---")
    st.markdown(f"**Data sources**")
    st.markdown(f"- Players : `{data_src}`")
    st.markdown(f"- Matches : `{match_src}`")
    st.markdown(f"- Players : `{len(players_df):,}`")
    st.markdown(f"- Matches : `{len(matches_df):,}`")
    st.markdown("---")
    st.caption("Built with Python · sklearn · Streamlit")

# ══════════════════════════════════════════════════════════════
#  HELPER — matplotlib fig → streamlit
# ══════════════════════════════════════════════════════════════
def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def role_tag(role):
    tags = {"Forward":"tag-fw","Midfielder":"tag-mf",
            "Defender":"tag-df","Goalkeeper":"tag-gk"}
    cls  = tags.get(role,"tag-mf")
    return f'<span class="tag {cls}">{role}</span>'

# ══════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero">
      <h1>Football Analytics ML System</h1>
      <p>Real-time predictions · Player scoring · Clustering · Recommendations</p>
    </div>""", unsafe_allow_html=True)

    # KPI strip
    acc  = models["match_acc"]
    r2   = models["perf_r2"]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Match Accuracy",  f"{acc*100:.1f}%", "Random Forest")
    c2.metric("Player R²",       f"{r2:.3f}",       "Ridge Regression")
    c3.metric("Players",         f"{len(players_df):,}")
    c4.metric("Matches",         f"{len(matches_df):,}")
    c5.metric("Clusters",        "4",               "K-Means")

    st.markdown('<div class="section-title">Top 10 Players by Performance</div>',
                unsafe_allow_html=True)
    top10 = players_df.nlargest(10,"Performance")
    fig,ax = plt.subplots(figsize=(10,4))
    fig.patch.set_facecolor("#0d1424")
    ax.set_facecolor("#111827")
    colors = plt.cm.plasma(np.linspace(0.2,0.9,10))
    bars   = ax.barh(top10["Name"] if "Name" in top10 else top10.index.astype(str),
                     top10["Performance"], color=colors, edgecolor="none")
    ax.invert_yaxis()
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Performance Score", color="white")
    for bar,val in zip(bars,top10["Performance"]):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{val:.0f}", va="center", color="white", fontsize=8)
    show_fig(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Match Results Distribution</div>',
                    unsafe_allow_html=True)
        res_c = matches_df["FTR"].map({"H":"Home Win","D":"Draw","A":"Away Win"}).value_counts()
        fig,ax = plt.subplots(figsize=(5,3))
        fig.patch.set_facecolor("#0d1424"); ax.set_facecolor("#111827")
        ax.bar(res_c.index, res_c.values,
               color=["#10b981","#f59e0b","#ef4444"], edgecolor="none", width=0.5)
        ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
        for i,(idx,v) in enumerate(res_c.items()):
            ax.text(i, v+1, str(v), ha="center", color="white", fontsize=9)
        show_fig(fig)

    with col2:
        st.markdown('<div class="section-title">Position Distribution</div>',
                    unsafe_allow_html=True)
        if "Position" in players_df.columns:
            pos_c = players_df["Position"].value_counts()
            fig,ax = plt.subplots(figsize=(5,3))
            fig.patch.set_facecolor("#0d1424")
            ax.pie(pos_c, labels=pos_c.index, autopct="%1.0f%%",
                   colors=["#ef4444","#4f8ef7","#10b981","#f59e0b"],
                   startangle=90, wedgeprops=dict(width=0.6))
            show_fig(fig)

# ══════════════════════════════════════════════════════════════
#  PAGE 2 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════
elif page == "🏆 Match Predictor":
    st.markdown("## 🏆 Match Outcome Predictor")
    st.markdown("Adjust the match statistics below to predict the result.")

    clf   = models["match_clf"]
    le    = models["match_le"]
    feats = models["match_feats"]
    acc   = models["match_acc"]

    st.info(f"Model: **Random Forest**  |  Accuracy: **{acc*100:.1f}%**  |  "
            f"Features: {', '.join(feats)}")

    st.markdown("### ⚙️ Enter Match Statistics")
    col1, col2 = st.columns(2)

    inputs = {}
    feat_labels = {
        "HS":"Home Shots","AS":"Away Shots",
        "HST":"Home Shots on Target","AST":"Away Shots on Target",
        "HC":"Home Corners","AC":"Away Corners",
        "HF":"Home Fouls","AF":"Away Fouls",
        "HY":"Home Yellow Cards","AY":"Away Yellow Cards",
    }
    defaults = {"HS":13,"AS":10,"HST":5,"AST":4,"HC":6,"AC":4,
                "HF":11,"AF":12,"HY":2,"AY":2}

    for i,feat in enumerate(feats):
        col = col1 if i%2==0 else col2
        label = feat_labels.get(feat, feat)
        mx    = 25 if "S" in feat else 15
        inputs[feat] = col.slider(label, 0, mx, defaults.get(feat,5))

    if st.button("🔮 Predict Match Outcome", use_container_width=True):
        X_input = pd.DataFrame([inputs])[feats]
        pred    = clf.predict(X_input)[0]
        proba   = clf.predict_proba(X_input)[0]
        classes = le.classes_   # A=Away Win, D=Draw, H=Home Win
        label_map = {"H":"🏠 Home Win","D":"⚖️ Draw","A":"✈️ Away Win"}
        color_map = {"H":"win","D":"draw","A":"loss"}
        result  = label_map[classes[pred]]
        box_cls = color_map[classes[pred]]

        st.markdown(f"""
        <div class="result-box {box_cls}">
          <h2 style="margin:0;color:white">Predicted Result: {result}</h2>
        </div>""", unsafe_allow_html=True)

        st.markdown("### 📊 Win Probabilities")
        prob_df = pd.DataFrame({
            "Result": [label_map.get(c,c) for c in classes],
            "Probability (%)": [round(p*100,1) for p in proba]
        })
        col1,col2 = st.columns([2,1])
        with col1:
            fig,ax = plt.subplots(figsize=(6,2.5))
            fig.patch.set_facecolor("#0d1424"); ax.set_facecolor("#111827")
            colors = ["#10b981","#f59e0b","#ef4444"]
            bars   = ax.barh(prob_df["Result"], prob_df["Probability (%)"],
                             color=colors, edgecolor="none")
            ax.set_xlim(0,100); ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
            for bar,val in zip(bars, prob_df["Probability (%)"]):
                ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                        f"{val}%", va="center", color="white", fontsize=10)
            show_fig(fig)
        with col2:
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # Feature importance
    st.markdown("### 🔍 Feature Importances")
    imp = pd.Series(clf.feature_importances_, index=feats).sort_values()
    fig,ax = plt.subplots(figsize=(8,3.5))
    fig.patch.set_facecolor("#0d1424"); ax.set_facecolor("#111827")
    imp.plot(kind="barh", ax=ax,
             color=plt.cm.plasma(np.linspace(0.2,0.9,len(imp))))
    ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
    ax.set_xlabel("Importance Score", color="white")
    show_fig(fig)

# ══════════════════════════════════════════════════════════════
#  PAGE 3 — PERFORMANCE SCORER
# ══════════════════════════════════════════════════════════════
elif page == "📈 Performance Scorer":
    st.markdown("## 📈 Player Performance Scorer")

    reg   = models["perf_reg"]
    sc    = models["perf_scaler"]
    feats = models["perf_feats"]
    r2    = models["perf_r2"]

    st.info(f"Model: **Ridge Regression**  |  R²: **{r2:.3f}**")

    st.markdown("### 🎮 Score a Custom Player")
    cols = st.columns(4)
    p_inputs = {}
    feat_defaults = {"Goals":10,"Assists":7,"Pace":78,"Shooting":75,
                     "Passing":72,"Dribbling":74,"Defending":45,"Physical":70}
    for i,feat in enumerate(feats):
        c = cols[i%4]
        mx = 200 if feat in ("Goals","Assists") else 99
        p_inputs[feat] = c.slider(feat, 0, mx, feat_defaults.get(feat, 50))

    if st.button("🧮 Calculate Performance Score", use_container_width=True):
        row  = pd.DataFrame([p_inputs])[feats]
        rows = sc.transform(row)
        pred = reg.predict(rows)[0]
        st.markdown(f"""
        <div class="result-box win" style="text-align:center">
          <div style="font-size:3rem;font-weight:700;color:#4fffc8">{pred:.1f}</div>
          <div style="color:#6b7a8d">Predicted Performance Score</div>
        </div>""", unsafe_allow_html=True)

        # Compare to dataset
        pct = (players_df["Performance"] < pred).mean() * 100
        st.success(f"This player ranks better than **{pct:.1f}%** of all players in the dataset!")

    # Top performers table
    st.markdown("### 🏅 Top 20 Players by Performance Score")
    top20 = players_df.nlargest(20,"Performance")
    show_cols = [c for c in ["Name","Position","Overall","Performance","Goals","Assists"]
                 if c in top20.columns]
    st.dataframe(top20[show_cols].reset_index(drop=True),
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 4 — CLUSTERS
# ══════════════════════════════════════════════════════════════
elif page == "🗂️ Player Clusters":
    st.markdown("## 🗂️ Player Clustering (K-Means)")
    st.markdown("Players automatically grouped into **4 archetypes** based on their stats.")

    cnames = models["cluster_names"]
    feats  = models["cluster_feats"]

    if "Cluster_Name" in players_df.columns:
        # Cluster summary
        col1,col2,col3,col4 = st.columns(4)
        for i,(cid,cname) in enumerate(cnames.items()):
            ct = players_df[players_df["Cluster"]==cid]
            [col1,col2,col3,col4][i].metric(cname, f"{len(ct)} players")

        # Scatter
        st.markdown("### 🔵 Cluster Scatter Plot")
        cx = st.selectbox("X axis", feats, index=0)
        cy = st.selectbox("Y axis", feats, index=1)

        fig,ax = plt.subplots(figsize=(9,5))
        fig.patch.set_facecolor("#0d1424"); ax.set_facecolor("#111827")
        colors_c = ["#ef4444","#4f8ef7","#10b981","#f59e0b"]
        for cid,cname in cnames.items():
            m = players_df["Cluster"]==cid
            ax.scatter(players_df.loc[m,cx], players_df.loc[m,cy],
                       color=colors_c[cid], label=cname, s=60, alpha=0.8, edgecolors="none")
        ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
        ax.set_xlabel(cx, color="white"); ax.set_ylabel(cy, color="white")
        ax.legend(facecolor="#111827", labelcolor="white", fontsize=9)
        show_fig(fig)

        # Cluster table
        st.markdown("### 📋 Players by Cluster")
        selected_cluster = st.selectbox("Filter by cluster",
                                         ["All"] + list(cnames.values()))
        df_show = players_df if selected_cluster=="All" \
                  else players_df[players_df["Cluster_Name"]==selected_cluster]
        show_cols = [c for c in ["Name","Position","Cluster_Name","Performance",
                                  "Goals","Assists","Pace"] if c in df_show.columns]
        st.dataframe(df_show[show_cols].sort_values("Performance",ascending=False)
                     .reset_index(drop=True),
                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 5 — RECOMMENDER
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Player Recommender":
    st.markdown("## 🤖 Player Recommender")
    st.markdown("Find players with similar playing styles using **Cosine Similarity**.")

    rd     = models["rec_data"]
    sim    = models["rec_sim"]
    feats  = models["rec_feats"]

    name_col = "Name" if "Name" in rd.columns else None
    if name_col:
        all_names = rd[name_col].tolist()
        chosen    = st.selectbox("🔍 Search for a player", all_names)
        top_n     = st.slider("Number of similar players to show", 3, 15, 5)

        idx  = rd[rd[name_col]==chosen].index[0]
        sims = sorted(enumerate(sim[idx]), key=lambda x:x[1], reverse=True)
        sims = [(i,s) for i,s in sims if i!=idx][:top_n]

        # Radar chart comparison
        st.markdown("### 📡 Radar Comparison")
        categories = feats[:6]
        N = len(categories)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        scr  = models["rec_scaler"]
        scr2 = MinMaxScaler().fit(rd[categories])

        fig,ax = plt.subplots(figsize=(7,5), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#0d1424"); ax.set_facecolor("#111827")
        ax.spines["polar"].set_color("#1e2a3a")
        ax.tick_params(colors="white", labelsize=8)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, color="white", size=9)

        palette = ["#4f8ef7","#ef4444","#10b981","#f59e0b","#8b5cf6"]
        t_val = scr2.transform([rd.iloc[idx][categories].values])[0].tolist() + [0]
        t_val[-1] = t_val[0]
        ax.plot(angles, t_val, color=palette[0], lw=2, label=str(chosen)[:16])
        ax.fill(angles, t_val, color=palette[0], alpha=0.2)

        for ri,(i,s) in enumerate(sims[:4]):
            rv = scr2.transform([rd.iloc[i][categories].values])[0].tolist()
            rv += rv[:1]
            nm = str(rd.iloc[i][name_col])[:16] if name_col else f"P{i}"
            ax.plot(angles, rv, color=palette[ri+1], lw=1.5, label=nm)
            ax.fill(angles, rv, color=palette[ri+1], alpha=0.08)

        ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.1),
                  facecolor="#111827", labelcolor="white", fontsize=8)
        show_fig(fig)

        # Similar players cards
        st.markdown("### 🃏 Most Similar Players")
        for rank,(i,s) in enumerate(sims,1):
            p = rd.iloc[i]
            pname = str(p[name_col]) if name_col else f"Player {i}"
            pos   = players_df.loc[players_df.get("Name","_")==pname,"Position"].values
            pos_t = pos[0] if len(pos)>0 else "—"
            perf  = players_df.loc[players_df.get("Name","_")==pname,"Performance"].values
            perf_v= f"{perf[0]:.0f}" if len(perf)>0 else "—"
            st.markdown(f"""
            <div class="player-card">
              <b>#{rank} {pname}</b>
              &nbsp; {role_tag(pos_t)}
              &nbsp;&nbsp;
              <span style="color:#4fffc8;font-weight:600">{s*100:.1f}% similar</span>
              &nbsp;&nbsp;
              <span style="color:#6b7a8d">Score: {perf_v}</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 6 — EDA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "🌍 EDA Explorer":
    st.markdown("## 🌍 Data Explorer")

    tab1,tab2 = st.tabs(["👤 Player Data","⚽ Match Data"])

    with tab1:
        st.markdown("### Filter Players")
        col1,col2,col3 = st.columns(3)
        positions = ["All"] + (players_df["Position"].dropna().unique().tolist()
                               if "Position" in players_df.columns else [])
        sel_pos  = col1.selectbox("Position", positions)
        min_ovr  = col2.slider("Min Overall", 0, 99, 60)
        max_age  = col3.slider("Max Age", 16, 50, 35)

        df_f = players_df.copy()
        if sel_pos != "All" and "Position" in df_f.columns:
            df_f = df_f[df_f["Position"]==sel_pos]
        if "Overall" in df_f.columns:
            df_f = df_f[pd.to_numeric(df_f["Overall"],errors="coerce").fillna(0)>=min_ovr]
        if "Age" in df_f.columns:
            df_f = df_f[pd.to_numeric(df_f["Age"],errors="coerce").fillna(99)<=max_age]

        st.write(f"**{len(df_f):,} players** match your filters")

        num_cols = [c for c in ["Overall","Performance","Goals","Assists",
                                 "Pace","Shooting","Passing","Dribbling"]
                    if c in df_f.columns]
        if len(num_cols) >= 2:
            col1,col2 = st.columns(2)
            xax = col1.selectbox("X axis", num_cols, index=0)
            yax = col2.selectbox("Y axis", num_cols, index=1)

            fig,ax = plt.subplots(figsize=(9,5))
            fig.patch.set_facecolor("#0d1424"); ax.set_facecolor("#111827")
            sc = ax.scatter(df_f[xax], df_f[yax],
                            c=df_f["Performance"] if "Performance" in df_f.columns else "cyan",
                            cmap="plasma", s=50, alpha=0.7, edgecolors="none")
            plt.colorbar(sc, ax=ax, label="Performance").ax.yaxis.label.set_color("white")
            ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
            ax.set_xlabel(xax, color="white"); ax.set_ylabel(yax, color="white")
            show_fig(fig)

        show_c = [c for c in ["Name","Position","Age","Overall","Performance"]
                  if c in df_f.columns]
        st.dataframe(df_f[show_c].sort_values("Performance",ascending=False)
                     .head(50).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Match Statistics")
        if "HomeTeam" in matches_df.columns:
            all_teams = sorted(set(matches_df["HomeTeam"].tolist()+matches_df["AwayTeam"].tolist()))
            sel_team  = st.selectbox("Filter by team", ["All"]+all_teams)
            mdf = matches_df if sel_team=="All" \
                  else matches_df[(matches_df["HomeTeam"]==sel_team)|(matches_df["AwayTeam"]==sel_team)]
        else:
            mdf = matches_df

        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(5,3.5))
            fig.patch.set_facecolor("#0d1424"); ax.set_facecolor("#111827")
            rc = mdf["FTR"].map({"H":"Home Win","D":"Draw","A":"Away Win"}).value_counts()
            ax.pie(rc, labels=rc.index, autopct="%1.1f%%",
                   colors=["#10b981","#f59e0b","#ef4444"], startangle=90)
            ax.set_title("Results", color="white")
            show_fig(fig)
        with col2:
            if "HS" in mdf.columns and "AS" in mdf.columns:
                fig,ax = plt.subplots(figsize=(5,3.5))
                fig.patch.set_facecolor("#0d1424"); ax.set_facecolor("#111827")
                ax.hist(mdf["HS"].dropna(), bins=15, alpha=0.7, color="#4f8ef7", label="Home")
                ax.hist(mdf["AS"].dropna(), bins=15, alpha=0.7, color="#ef4444", label="Away")
                ax.legend(facecolor="#111827", labelcolor="white")
                ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
                ax.set_title("Shots Distribution", color="white")
                show_fig(fig)

# ══════════════════════════════════════════════════════════════
#  PAGE 7 — LEAGUE TABLE
# ══════════════════════════════════════════════════════════════
elif page == "📊 League Table":
    st.markdown("## 📊 League Table & Team Analytics")

    if "HomeTeam" in matches_df.columns:
        teams = list(set(matches_df["HomeTeam"].tolist()+matches_df["AwayTeam"].tolist()))
        rows  = []
        for t in teams:
            h = matches_df[matches_df["HomeTeam"]==t]
            a = matches_df[matches_df["AwayTeam"]==t]
            gf   = h["FTHG"].sum() + a["FTAG"].sum()
            ga   = h["FTAG"].sum() + a["FTHG"].sum()
            wins  = (h["FTR"]=="H").sum() + (a["FTR"]=="A").sum()
            draws = (h["FTR"]=="D").sum() + (a["FTR"]=="D").sum()
            games = len(h)+len(a)
            pts   = wins*3+draws
            rows.append({"Team":t,"P":games,"W":int(wins),"D":int(draws),
                         "L":int(games-wins-draws),"GF":int(gf),"GA":int(ga),
                         "GD":int(gf-ga),"Pts":int(pts)})
        table = pd.DataFrame(rows).sort_values("Pts",ascending=False).reset_index(drop=True)
        table.index += 1

        st.markdown("### 🏆 Full Standings")
        st.dataframe(table, use_container_width=True)

        st.markdown("### 📈 Goals Scored vs Conceded")
        top8 = table.head(8)
        fig,ax = plt.subplots(figsize=(10,4))
        fig.patch.set_facecolor("#0d1424"); ax.set_facecolor("#111827")
        x = np.arange(len(top8))
        ax.bar(x-0.2, top8["GF"], 0.4, label="Goals For",     color="#4f8ef7", edgecolor="none")
        ax.bar(x+0.2, top8["GA"], 0.4, label="Goals Against",  color="#ef4444", edgecolor="none")
        ax.set_xticks(x); ax.set_xticklabels(top8["Team"], rotation=30, ha="right", color="white")
        ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
        ax.legend(facecolor="#111827", labelcolor="white")
        show_fig(fig)

        st.markdown("### 🎯 Head-to-Head Lookup")
        all_teams_sorted = sorted(teams)
        c1,c2 = st.columns(2)
        t1 = c1.selectbox("Home Team", all_teams_sorted, index=0)
        t2 = c2.selectbox("Away Team", all_teams_sorted, index=1 if len(all_teams_sorted)>1 else 0)
        h2h = matches_df[
            ((matches_df["HomeTeam"]==t1)&(matches_df["AwayTeam"]==t2)) |
            ((matches_df["HomeTeam"]==t2)&(matches_df["AwayTeam"]==t1))
        ]
        if len(h2h) > 0:
            st.write(f"**{len(h2h)} meetings found**")
            t1_wins = ((h2h["HomeTeam"]==t1)&(h2h["FTR"]=="H")).sum() + \
                      ((h2h["AwayTeam"]==t1)&(h2h["FTR"]=="A")).sum()
            t2_wins = ((h2h["HomeTeam"]==t2)&(h2h["FTR"]=="H")).sum() + \
                      ((h2h["AwayTeam"]==t2)&(h2h["FTR"]=="A")).sum()
            draws   = (h2h["FTR"]=="D").sum()
            cc1,cc2,cc3 = st.columns(3)
            cc1.metric(f"{t1} Wins", int(t1_wins))
            cc2.metric("Draws",      int(draws))
            cc3.metric(f"{t2} Wins", int(t2_wins))
        else:
            st.info("No head-to-head matches found between these teams.")
    else:
        st.warning("Match team data not available.")
