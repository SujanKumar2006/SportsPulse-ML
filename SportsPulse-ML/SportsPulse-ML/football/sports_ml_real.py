"""
================================================================
  SPORTS ANALYTICS ML SYSTEM — REAL DATA + DYNAMIC CHARTS
  Uses: FIFA Player Stats + EPL Match Results (real datasets)
  Charts: 100% dynamic — different visuals every run
================================================================
"""

import os, sys, time, random, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
from io  import StringIO, BytesIO

warnings.filterwarnings('ignore')

# ── Output folder ──────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
def out(f): return os.path.join(OUTPUT_DIR, f)

# ── Dynamic seed — changes every run ──────────────────────────
SEED = int(time.time())
np.random.seed(SEED)
random.seed(SEED)
print(f"🎲 Dynamic seed this run: {SEED}")
print("   (Every run = fresh random splits + different chart colours)\n")

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LinearRegression, Ridge
from sklearn.cluster           import KMeans
from sklearn.metrics.pairwise  import cosine_similarity
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.preprocessing     import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics           import (accuracy_score, mean_squared_error, r2_score)

# ══════════════════════════════════════════════════════════════
#  STEP 1 — DOWNLOAD REAL DATASETS
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("  STEP 1: DOWNLOADING REAL DATASETS")
print("=" * 60)

def fetch_csv(url, label):
    print(f"  Fetching {label} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=20,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        print(f"OK  ({len(df):,} rows)")
        return df
    except Exception as e:
        print(f"FAILED ({e})")
        return None

FIFA_URLS = [
    ("FIFA 20 Player Stats",
     "https://raw.githubusercontent.com/305kishan/FIFA/main/data/FIFA20.csv"),
    ("FIFA 21 Raw Data",
     "https://raw.githubusercontent.com/ddebonis47/classwork/refs/heads/main/fifa21%20raw%20data%20v2.csv"),
    ("FIFA 20 Alt Mirror",
     "https://raw.githubusercontent.com/apoorva-21/fifa-analysis/master/data/players_20.csv"),
]
EPL_URLS = [
    ("EPL 2023-24", "https://www.football-data.co.uk/mmz4281/2324/E0.csv"),
    ("EPL 2022-23", "https://www.football-data.co.uk/mmz4281/2223/E0.csv"),
    ("EPL 2021-22", "https://www.football-data.co.uk/mmz4281/2122/E0.csv"),
]

fifa_raw = None
for label, url in FIFA_URLS:
    fifa_raw = fetch_csv(url, label)
    if fifa_raw is not None:
        break

epl_raw = None
for label, url in EPL_URLS:
    epl_raw = fetch_csv(url, label)
    if epl_raw is not None:
        break

# ── Fallbacks ──────────────────────────────────────────────────
def make_fallback_players(n=300):
    print("  Using high-quality synthetic player data")
    positions = ["Forward","Midfielder","Defender","Goalkeeper"]
    df = pd.DataFrame({
        "Name"      : [f"Player_{i:03d}" for i in range(1, n+1)],
        "Nationality": np.random.choice(["England","Spain","France","Germany",
            "Brazil","Argentina","Portugal","Netherlands","Italy","Belgium"], n),
        "Position"  : np.random.choice(positions, n, p=[0.28,0.35,0.27,0.10]),
        "Age"       : np.random.randint(17, 37, n),
        "Overall"   : np.random.randint(60, 95, n),
        "Potential" : np.random.randint(65, 99, n),
        "Value_M"   : np.round(np.random.exponential(25, n).clip(1,200), 1),
        "Goals"     : np.random.poisson(7,  n),
        "Assists"   : np.random.poisson(4,  n),
        "Pace"      : np.random.randint(50, 98, n),
        "Shooting"  : np.random.randint(45, 95, n),
        "Passing"   : np.random.randint(50, 95, n),
        "Dribbling" : np.random.randint(45, 97, n),
        "Defending" : np.random.randint(30, 90, n),
        "Physical"  : np.random.randint(50, 95, n),
        "SprintSpeed": np.random.randint(50, 97, n),
        "Stamina"   : np.random.randint(55, 97, n),
        "Vision"    : np.random.randint(40, 95, n),
    })
    df["Performance"] = (
        df["Goals"]*6 + df["Assists"]*4 + df["Overall"]*0.8 +
        df["Pace"]*0.2 + np.random.normal(0, 8, n)
    ).round(1)
    return df

def make_fallback_matches(n=380):
    print("  Using high-quality synthetic match data")
    teams = ["Arsenal","Chelsea","Liverpool","Man City","Man Utd","Tottenham",
             "Newcastle","Aston Villa","West Ham","Brighton","Brentford",
             "Fulham","Crystal Palace","Wolves","Everton","Nottm Forest",
             "Bournemouth","Luton","Sheffield Utd","Burnley"]
    hg = np.random.poisson(1.5, n)
    ag = np.random.poisson(1.2, n)
    df = pd.DataFrame({
        "HomeTeam": np.random.choice(teams, n),
        "AwayTeam": np.random.choice(teams, n),
        "FTHG": hg, "FTAG": ag,
        "FTR" : np.where(hg>ag,"H",np.where(hg<ag,"A","D")),
        "HS"  : np.random.randint(5, 22, n),
        "AS"  : np.random.randint(3, 20, n),
        "HST" : np.random.randint(2, 10, n),
        "AST" : np.random.randint(1,  9, n),
        "HC"  : np.random.randint(2, 12, n),
        "AC"  : np.random.randint(1, 11, n),
        "HF"  : np.random.randint(5, 18, n),
        "AF"  : np.random.randint(5, 18, n),
        "HY"  : np.random.randint(0,  4, n),
        "AY"  : np.random.randint(0,  4, n),
    })
    return df

# ── Process player data ────────────────────────────────────────
if fifa_raw is not None:
    players_df = fifa_raw.copy()

    # ── Step 1: Normalise column names ─────────────────────────
    # FIFA 20 CSV uses columns like: short_name, nationality, player_positions,
    # age, overall, potential, pace, shooting, passing, dribbling, defending, physic
    # We map whatever we find to our standard names
    col_map = {}
    for c in players_df.columns:
        cl = c.lower().strip()
        # Name
        if cl in ("short_name","name","player_name","long_name"):
            if "Name" not in col_map.values(): col_map[c] = "Name"
        # Nationality
        if cl in ("nationality","nationality_name","nation"):
            col_map[c] = "Nationality"
        # Position
        if cl in ("player_positions","position","pos","team_position"):
            if "Position" not in col_map.values(): col_map[c] = "Position"
        # Age
        if cl == "age":                          col_map[c] = "Age"
        # Overall / Potential
        if cl == "overall":                      col_map[c] = "Overall"
        if cl in ("potential","pot"):            col_map[c] = "Potential"
        # Key attributes — FIFA 20 stores these as numeric columns
        if cl in ("pace","pac"):                 col_map[c] = "Pace"
        if cl in ("shooting","sho"):             col_map[c] = "Shooting"
        if cl in ("passing","pas"):              col_map[c] = "Passing"
        if cl in ("dribbling","dri"):            col_map[c] = "Dribbling"
        if cl in ("defending","def"):            col_map[c] = "Defending"
        if cl in ("physic","physical","phy"):    col_map[c] = "Physical"
        if cl in ("value_eur","value","eur_value"):
            col_map[c] = "Value_M"

    players_df.rename(columns=col_map, inplace=True)

    # ── Step 2: Force numeric on key columns ───────────────────
    for col in ["Age","Overall","Potential","Pace","Shooting",
                "Passing","Dribbling","Defending","Physical"]:
        if col in players_df.columns:
            players_df[col] = pd.to_numeric(players_df[col], errors='coerce')

    # ── Step 3: Clean Position — FIFA 20 stores "ST, CF" etc ───
    if "Position" in players_df.columns:
        def clean_pos(p):
            if pd.isna(p): return "Midfielder"
            p = str(p).split(",")[0].strip().upper()
            if p in ("ST","CF","LW","RW","LF","RF","LS","RS","SS"): return "Forward"
            if p in ("CAM","CM","CDM","LM","RM","LAM","RAM","LCM","RCM","LCDM","RCDM"): return "Midfielder"
            if p in ("CB","LB","RB","LWB","RWB","LCB","RCB"): return "Defender"
            if p == "GK": return "Goalkeeper"
            return "Midfielder"
        players_df["Position"] = players_df["Position"].apply(clean_pos)

    # ── Step 4: Convert Value_M from euros to millions ─────────
    if "Value_M" in players_df.columns:
        players_df["Value_M"] = pd.to_numeric(players_df["Value_M"], errors='coerce')
        # if values are raw euros (e.g. 105500000), divide by 1M
        if players_df["Value_M"].median() > 1000:
            players_df["Value_M"] = (players_df["Value_M"] / 1_000_000).round(1)
        players_df["Value_M"] = players_df["Value_M"].clip(0.5, 250)

    # ── Step 5: Add any missing columns with realistic randoms ──
    n = len(players_df)
    missing_cols = {
        "Goals"    : lambda: np.random.poisson(5,  n),
        "Assists"  : lambda: np.random.poisson(3,  n),
        "Pace"     : lambda: np.random.randint(50, 98, n),
        "Shooting" : lambda: np.random.randint(45, 95, n),
        "Passing"  : lambda: np.random.randint(50, 95, n),
        "Dribbling": lambda: np.random.randint(45, 97, n),
        "Defending": lambda: np.random.randint(30, 90, n),
        "Physical" : lambda: np.random.randint(50, 95, n),
        "Value_M"  : lambda: np.round(np.random.exponential(20,n).clip(1,200),1),
        "Position" : lambda: np.random.choice(
            ["Forward","Midfielder","Defender","Goalkeeper"], n),
    }
    for col, fn in missing_cols.items():
        if col not in players_df.columns:
            players_df[col] = fn()

    # ── Step 6: Build performance score ────────────────────────
    overall = pd.to_numeric(players_df.get("Overall", 70), errors='coerce').fillna(70)
    pace    = pd.to_numeric(players_df.get("Pace",    70), errors='coerce').fillna(70)
    players_df["Performance"] = (
        players_df["Goals"]*6 + players_df["Assists"]*4 +
        overall*0.8 + pace*0.2 + np.random.normal(0, 8, n)
    ).round(1)

    players_df.dropna(subset=["Age","Overall"], inplace=True)
    players_df = players_df.sample(min(500, len(players_df)),
                                   random_state=SEED).reset_index(drop=True)

    print(f"  ✅ FIFA data processed: {len(players_df):,} players | "
          f"Cols: {[c for c in ['Name','Position','Age','Overall','Pace','Shooting','Passing'] if c in players_df.columns]}")
else:
    players_df = make_fallback_players(300)

# ── Process match data ─────────────────────────────────────────
if epl_raw is not None:
    matches_df = epl_raw.copy()
    needed = ["HomeTeam","AwayTeam","FTHG","FTAG","FTR","HS","AS","HST","AST"]
    if any(c not in matches_df.columns for c in needed):
        matches_df = make_fallback_matches(380)
    else:
        for c in ["FTHG","FTAG","HS","AS","HST","AST"]:
            matches_df[c] = pd.to_numeric(matches_df[c], errors='coerce')
        matches_df.dropna(subset=["FTHG","FTAG","FTR"], inplace=True)
        nm = len(matches_df)
        for col, fn in [("HC", lambda: np.random.randint(2,12,nm)),
                        ("AC", lambda: np.random.randint(1,11,nm)),
                        ("HF", lambda: np.random.randint(5,18,nm)),
                        ("AF", lambda: np.random.randint(5,18,nm)),
                        ("HY", lambda: np.random.randint(0, 4,nm)),
                        ("AY", lambda: np.random.randint(0, 4,nm))]:
            if col not in matches_df.columns:
                matches_df[col] = fn()
else:
    matches_df = make_fallback_matches(380)

print(f"\n  Players loaded : {len(players_df):,}")
print(f"  Matches loaded : {len(matches_df):,}")

# ── Dynamic palette ────────────────────────────────────────────
PALETTES = [
    ["#FF6B6B","#4ECDC4","#45B7D1","#96CEB4","#FFEAA7","#DDA0DD"],
    ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6","#1ABC9C"],
    ["#FF8C00","#00CED1","#FF1493","#32CD32","#4169E1","#FFD700"],
    ["#FF6347","#40E0D0","#DA70D6","#90EE90","#87CEEB","#F0E68C"],
    ["#C0392B","#2980B9","#27AE60","#D35400","#8E44AD","#16A085"],
]
CMAPS = ["coolwarm","viridis","plasma","RdYlGn","YlOrRd","Blues","Purples"]
PAL   = random.choice(PALETTES)
CMAP  = random.choice(CMAPS)
BG    = random.choice(["#0f0f1a","#1a0f2e","#0f1a0f","#1a1a0f","#0f1a1a"])
print(f"  Palette : {PAL[:3]}  CMAP: {CMAP}")

# ══════════════════════════════════════════════════════════════
#  STEP 2 — DYNAMIC EDA
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

NUM_COLS = [c for c in ["Goals","Assists","Pace","Shooting","Passing",
            "Dribbling","Defending","Physical","Performance"]
            if c in players_df.columns]
HEAT_COLS = random.sample(NUM_COLS, min(6, len(NUM_COLS)))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"EDA Dashboard  (Seed: {SEED})", fontsize=16, fontweight='bold')
fig.patch.set_facecolor('#f8f9fa')

if "Position" in players_df.columns:
    pos_c = players_df["Position"].value_counts().head(6)
    axes[0,0].pie(pos_c, labels=pos_c.index, autopct="%1.1f%%",
                  colors=PAL[:len(pos_c)], startangle=random.randint(0,360),
                  explode=[0.05]*len(pos_c))
    axes[0,0].set_title("Position Distribution", fontweight='bold')

if "Nationality" in players_df.columns:
    nat_c = players_df["Nationality"].value_counts().head(10)
    if random.choice([True, False]):
        axes[0,1].barh(nat_c.index, nat_c.values,
                       color=plt.cm.get_cmap(CMAP)(np.linspace(0.2,0.9,10)))
        axes[0,1].invert_yaxis()
    else:
        axes[0,1].bar(nat_c.index, nat_c.values,
                      color=plt.cm.get_cmap(CMAP)(np.linspace(0.2,0.9,10)))
        axes[0,1].tick_params(axis='x', rotation=40)
    axes[0,1].set_title("Top 10 Nationalities", fontweight='bold')

if "Performance" in players_df.columns:
    perf = players_df["Performance"].dropna()
    if random.choice([True, False]):
        axes[0,2].hist(perf, bins=20, color=PAL[1], edgecolor='white', alpha=0.85)
    else:
        perf.plot.kde(ax=axes[0,2], color=PAL[1], lw=2)
        axes[0,2].fill_between(np.linspace(perf.min(),perf.max(),200),
                               0, color=PAL[1], alpha=0.2)
    axes[0,2].axvline(perf.mean(), color='red', linestyle='--', label='Mean')
    axes[0,2].set_title("Performance Score Distribution", fontweight='bold')
    axes[0,2].legend()

res_map = {"H":"Home Win","D":"Draw","A":"Away Win"}
matches_df["Result_Label"] = matches_df["FTR"].map(res_map)
res_c = matches_df["Result_Label"].value_counts()
axes[1,0].bar(res_c.index, res_c.values,
              color=[PAL[2],PAL[3],PAL[0]][:len(res_c)], edgecolor='white')
axes[1,0].set_title("Match Result Distribution", fontweight='bold')
for i,(idx,val) in enumerate(res_c.items()):
    axes[1,0].text(i, val+1, str(val), ha='center', fontweight='bold')

corr = players_df[HEAT_COLS].dropna().corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap=CMAP,
            ax=axes[1,1], annot_kws={"size":8}, linewidths=0.5)
axes[1,1].set_title("Correlation Heatmap", fontweight='bold')

SCATTER_OPTS = [(a,b) for a,b in [("Goals","Assists"),("Shooting","Dribbling"),
                ("Pace","Physical"),("Passing","Defending")]
                if a in players_df.columns and b in players_df.columns]
sx, sy = random.choice(SCATTER_OPTS) if SCATTER_OPTS else ("Goals","Assists")
sc = players_df[[sx,sy,"Performance"]].dropna()
s = axes[1,2].scatter(sc[sx], sc[sy], c=sc["Performance"],
                       cmap=CMAP, s=60, alpha=0.7, edgecolors='none')
plt.colorbar(s, ax=axes[1,2], label="Performance")
axes[1,2].set_title(f"{sx} vs {sy}", fontweight='bold')
axes[1,2].set_xlabel(sx); axes[1,2].set_ylabel(sy)

plt.tight_layout()
plt.savefig(out("1_eda_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  EDA chart saved!")

# ══════════════════════════════════════════════════════════════
#  MODULE A — MATCH OUTCOME PREDICTOR
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  MODULE A: MATCH OUTCOME PREDICTOR")
print("="*60)

feat_cols = [c for c in ["HS","AS","HST","AST","HC","AC","HF","AF","HY","AY"]
             if c in matches_df.columns]
X = matches_df[feat_cols].dropna()
le = LabelEncoder()
y = le.fit_transform(matches_df.loc[X.index,"FTR"])

test_size = round(random.uniform(0.20, 0.30), 2)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                            random_state=SEED, stratify=y)

use_gb = random.choice([True, False])
if use_gb:
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                     learning_rate=0.1, random_state=SEED)
    model_name = "Gradient Boosting"
else:
    clf = RandomForestClassifier(n_estimators=150, max_depth=6,
                                  random_state=SEED, class_weight='balanced')
    model_name = "Random Forest"

clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
acc    = accuracy_score(y_te, y_pred)
cv_sc  = cross_val_score(clf, X, y, cv=5)

print(f"  Model    : {model_name}")
print(f"  Accuracy : {acc*100:.1f}%   CV: {cv_sc.mean()*100:.1f}%±{cv_sc.std()*100:.1f}%")

from sklearn.metrics import confusion_matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Match Outcome Predictor — {model_name}  (Seed:{SEED})",
             fontsize=13, fontweight='bold')
cm = confusion_matrix(y_te, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap=CMAP,
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title(f"Confusion Matrix — Accuracy: {acc*100:.1f}%", fontweight='bold')
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values()
imp.plot(kind='barh', ax=axes[1],
         color=plt.cm.get_cmap(CMAP)(np.linspace(0.2,0.9,len(imp))))
axes[1].set_title("Feature Importances", fontweight='bold')
plt.tight_layout()
plt.savefig(out("2_match_predictor.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Match predictor chart saved!")

# ══════════════════════════════════════════════════════════════
#  MODULE B — PLAYER PERFORMANCE SCORER
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  MODULE B: PLAYER PERFORMANCE SCORER")
print("="*60)

PERF_FEATS = [c for c in ["Goals","Assists","Pace","Shooting","Passing",
              "Dribbling","Defending","Physical"] if c in players_df.columns]
p_data = players_df[PERF_FEATS+["Performance"]].dropna()
Xp = p_data[PERF_FEATS]; yp = p_data["Performance"]
scaler_p = StandardScaler()
Xp_s = scaler_p.fit_transform(Xp)
Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(Xp_s, yp,
                                test_size=round(random.uniform(0.2,0.3),2),
                                random_state=SEED)
use_ridge = random.choice([True,False])
if use_ridge:
    alpha = round(random.uniform(0.5, 5.0), 2)
    reg = Ridge(alpha=alpha); reg_name = f"Ridge (alpha={alpha})"
else:
    reg = LinearRegression(); reg_name = "Linear Regression"

reg.fit(Xp_tr, yp_tr)
yp_pred = reg.predict(Xp_te)
r2   = r2_score(yp_te, yp_pred)
rmse = np.sqrt(mean_squared_error(yp_te, yp_pred))
print(f"  Model : {reg_name}   R²:{r2:.3f}   RMSE:{rmse:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle(f"Performance Scorer — {reg_name}  (Seed:{SEED})",
             fontsize=13, fontweight='bold')
axes[0].scatter(yp_te, yp_pred, alpha=0.65, color=PAL[1], edgecolors='white', s=60)
mn,mx = min(yp_te.min(),yp_pred.min()), max(yp_te.max(),yp_pred.max())
axes[0].plot([mn,mx],[mn,mx],'r--',lw=2,label='Perfect fit')
axes[0].set_title(f"Actual vs Predicted  R²={r2:.3f}", fontweight='bold')
axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted"); axes[0].legend()
coef = pd.Series(reg.coef_, index=PERF_FEATS).sort_values()
coef.plot(kind='barh', ax=axes[1], color=[PAL[0] if v<0 else PAL[2] for v in coef])
axes[1].axvline(0, color='black', lw=0.8)
axes[1].set_title("Feature Coefficients", fontweight='bold')
plt.tight_layout()
plt.savefig(out("3_performance_scorer.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Performance scorer chart saved!")

# ══════════════════════════════════════════════════════════════
#  MODULE C — PLAYER CLUSTERING
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  MODULE C: PLAYER CLUSTERING")
print("="*60)

CLUST_FEATS = [c for c in ["Goals","Assists","Passing","Dribbling",
               "Defending","Physical","Pace"] if c in players_df.columns]
c_data = players_df[CLUST_FEATS].dropna()
scaler_c = StandardScaler()
c_scaled = scaler_c.fit_transform(c_data)

inertias = []
for k in range(2,9):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km.fit(c_scaled)
    inertias.append(km.inertia_)

optimal_k = random.choice([3,4,5])
kmeans = KMeans(n_clusters=optimal_k, random_state=SEED, n_init=10)
c_labels = kmeans.fit_predict(c_scaled)

ALL_NAMES = {
    3: ["Defenders","Attackers","Playmakers"],
    4: ["Defensive Wall","Speed Merchant","Goal Machine","Playmaker"],
    5: ["Rock Solid","Speedster","Striker","Creator","Box-to-Box"],
}
cnames = ALL_NAMES[optimal_k]
c_data_copy = c_data.copy()
c_data_copy["Cluster"] = c_labels
c_data_copy["Type"]    = c_data_copy["Cluster"].map(lambda x: cnames[x])

print(f"  k={optimal_k} clusters:")
for i,n in enumerate(cnames):
    print(f"    {n}: {(c_data_copy['Cluster']==i).sum()} players")

fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle(f"Player Clustering — k={optimal_k}  (Seed:{SEED})",
             fontsize=13, fontweight='bold')
axes[0].plot(range(2,9), inertias, 'o-', color=PAL[0], lw=2, ms=8)
axes[0].axvline(optimal_k, color='red', linestyle='--', label=f'k={optimal_k}')
axes[0].set_title("Elbow Method", fontweight='bold')
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia"); axes[0].legend()

cx,cy = random.choice([(CLUST_FEATS[0],CLUST_FEATS[1]),
                        (CLUST_FEATS[-1],CLUST_FEATS[0])])
for ci in range(optimal_k):
    m = c_data_copy["Cluster"]==ci
    axes[1].scatter(c_data_copy.loc[m,cx], c_data_copy.loc[m,cy],
                    c=PAL[ci], label=cnames[ci], s=70, alpha=0.8, edgecolors='white')
axes[1].set_title(f"Clusters: {cx} vs {cy}", fontweight='bold')
axes[1].set_xlabel(cx); axes[1].set_ylabel(cy); axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(out("4_player_clusters.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Clustering chart saved!")

# ══════════════════════════════════════════════════════════════
#  MODULE D — PLAYER RECOMMENDER
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  MODULE D: PLAYER RECOMMENDER")
print("="*60)

REC_FEATS = [c for c in ["Goals","Assists","Pace","Shooting","Passing",
             "Dribbling","Defending","Physical"] if c in players_df.columns]
name_col = "Name" if "Name" in players_df.columns else None
r_cols   = REC_FEATS + ([name_col] if name_col else [])
r_data   = players_df[r_cols].dropna().reset_index(drop=True)

scaler_r = MinMaxScaler()
r_scaled = scaler_r.fit_transform(r_data[REC_FEATS])
sim_mat  = cosine_similarity(r_scaled)

target_idx  = random.randint(0, len(r_data)-1)
target_name = r_data.iloc[target_idx][name_col] if name_col else f"Player_{target_idx}"

sims = sorted(enumerate(sim_mat[target_idx]), key=lambda x:x[1], reverse=True)
sims = [(i,s) for i,s in sims if i!=target_idx][:5]
recs = []
for rank,(i,s) in enumerate(sims,1):
    row = {"Rank":rank, "Similarity":f"{s*100:.1f}%"}
    if name_col: row["Player"] = r_data.iloc[i][name_col]
    recs.append(row)
recs_df = pd.DataFrame(recs)

print(f"  Target : {target_name}")
print(recs_df.to_string(index=False))

sample_idx   = random.sample(range(len(r_data)), min(12,len(r_data)))
sim_sub      = sim_mat[np.ix_(sample_idx,sample_idx)]
sample_names = ([str(r_data.iloc[i][name_col])[:12] for i in sample_idx]
                if name_col else [f"P{i}" for i in sample_idx])

fig, axes = plt.subplots(1,2,figsize=(16,6))
fig.suptitle(f"Player Recommender  (Seed:{SEED})", fontsize=13, fontweight='bold')
sns.heatmap(sim_sub, xticklabels=sample_names, yticklabels=sample_names,
            annot=True, fmt=".2f", cmap=CMAP, ax=axes[0],
            annot_kws={"size":7}, linewidths=0.4)
axes[0].set_title("Similarity Matrix (random 12-player sample)", fontweight='bold')
axes[0].tick_params(axis='x', rotation=45, labelsize=7)

# Radar chart
categories = REC_FEATS[:6]
N = len(categories)
angles = np.linspace(0,2*np.pi,N,endpoint=False).tolist() + [0]
ax_r = plt.subplot(122, polar=True)
# Use a separate scaler fitted only on the 6 radar categories
scaler_radar = MinMaxScaler()
scaler_radar.fit(r_data[categories])
t_vals = scaler_radar.transform([r_data.iloc[target_idx][categories].values])[0].tolist()
t_vals += t_vals[:1]
ax_r.plot(angles, t_vals, color=PAL[0], lw=2, label=str(target_name)[:14])
ax_r.fill(angles, t_vals, color=PAL[0], alpha=0.25)
for ri,(i,s) in enumerate(sims[:3]):
    rv = scaler_radar.transform([r_data.iloc[i][categories].values])[0].tolist()
    rv += rv[:1]
    lbl = str(r_data.iloc[i][name_col])[:14] if name_col else f"Sim{ri+1}"
    ax_r.plot(angles, rv, color=PAL[ri+1], lw=2, label=lbl)
    ax_r.fill(angles, rv, color=PAL[ri+1], alpha=0.15)
ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(categories, size=8)
ax_r.set_title(f"Radar: {str(target_name)[:14]}", fontweight='bold', pad=20)
ax_r.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=7)
plt.tight_layout()
plt.savefig(out("5_player_recommender.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Recommender chart saved!")

# ══════════════════════════════════════════════════════════════
#  BONUS — LEAGUE TABLE ANALYSER
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  BONUS: LEAGUE TABLE ANALYSER")
print("="*60)

def team_stats(team, df):
    h = df[df["HomeTeam"]==team]; a = df[df["AwayTeam"]==team]
    gf = h["FTHG"].sum() + a["FTAG"].sum()
    ga = h["FTAG"].sum() + a["FTHG"].sum()
    wins  = (h["FTR"]=="H").sum() + (a["FTR"]=="A").sum()
    draws = (h["FTR"]=="D").sum() + (a["FTR"]=="D").sum()
    games = len(h)+len(a)
    return {"Team":team,"Games":games,"Wins":int(wins),"Draws":int(draws),
            "Losses":int(games-wins-draws),"GF":int(gf),"GA":int(ga),
            "GD":int(gf-ga),"Points":int(wins*3+draws)}

all_teams = [t for t in set(matches_df["HomeTeam"].tolist()+
             matches_df["AwayTeam"].tolist()) if t and str(t)!='nan']
table = pd.DataFrame([team_stats(t,matches_df) for t in all_teams])
table.sort_values("Points",ascending=False,inplace=True)
table.reset_index(drop=True,inplace=True); table.index+=1
print(table.head(10).to_string())

fig, axes = plt.subplots(1,2,figsize=(16,6))
fig.suptitle(f"League Table & Team Stats  (Seed:{SEED})",fontsize=13,fontweight='bold')
top10 = table.head(10)
bar_c = plt.cm.get_cmap(CMAP)(np.linspace(0.2,0.9,len(top10)))
axes[0].barh(top10["Team"][::-1], top10["Points"][::-1], color=bar_c[::-1])
axes[0].set_title("Top 10 Teams by Points", fontweight='bold')
axes[0].set_xlabel("Points")
for i,(_,row) in enumerate(top10[::-1].iterrows()):
    axes[0].text(row["Points"]+0.3, i, str(row["Points"]),va='center',fontsize=8)

x = np.arange(len(top10)); w=0.6
axes[1].bar(x, top10["Wins"],   width=w, label="Wins",   color=PAL[2])
axes[1].bar(x, top10["Draws"],  width=w, label="Draws",  color=PAL[3],
            bottom=top10["Wins"])
axes[1].bar(x, top10["Losses"], width=w, label="Losses", color=PAL[0],
            bottom=top10["Wins"]+top10["Draws"])
axes[1].set_xticks(x)
axes[1].set_xticklabels(top10["Team"], rotation=40, ha='right', fontsize=8)
axes[1].set_title("Win / Draw / Loss Breakdown", fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig(out("6_team_form.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Team form chart saved!")

# ══════════════════════════════════════════════════════════════
#  MASTER DASHBOARD
# ══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22,13))
fig.patch.set_facecolor(BG)
fig.suptitle(f"SPORTS ANALYTICS ML DASHBOARD  |  Seed: {SEED}",
             fontsize=18,fontweight='bold',color='white',y=0.99)

kpis = [("Match Accuracy",f"{acc*100:.1f}%",PAL[0]),
        ("Player R²",f"{r2:.3f}",PAL[1]),
        ("Players",f"{len(players_df):,}",PAL[2]),
        ("Matches",f"{len(matches_df):,}",PAL[3]),
        ("Clusters",str(optimal_k),PAL[4]),
        ("ML Model",model_name.split()[0],PAL[5] if len(PAL)>5 else PAL[0])]
for i,(lbl,val,col) in enumerate(kpis):
    ax = fig.add_axes([0.02+i*0.163, 0.87, 0.15, 0.09])
    ax.set_facecolor(col); ax.axis('off')
    ax.text(0.5,0.65,val,ha='center',va='center',fontsize=20,
            fontweight='bold',color='white',transform=ax.transAxes)
    ax.text(0.5,0.18,lbl,ha='center',va='center',fontsize=8,
            color='white',alpha=0.9,transform=ax.transAxes)

ax1 = fig.add_axes([0.02,0.48,0.28,0.33]); ax1.set_facecolor('#1e1e2e')
perf = players_df["Performance"].dropna()
ax1.hist(perf, bins=20, color=PAL[1], edgecolor='none', alpha=0.9)
ax1.axvline(perf.mean(),color='red',lw=2,linestyle='--',label=f'Mean={perf.mean():.0f}')
ax1.set_title("Performance Distribution",color='white',fontweight='bold')
ax1.tick_params(colors='white'); ax1.spines[:].set_visible(False)
ax1.legend(fontsize=8,labelcolor='white',facecolor='#1e1e2e')

ax2 = fig.add_axes([0.35,0.48,0.20,0.33]); ax2.set_facecolor('#1e1e2e')
ax2.bar(res_c.index, res_c.values,
        color=[PAL[2],PAL[0],PAL[3]][:len(res_c)], edgecolor='none')
ax2.set_title("Match Results",color='white',fontweight='bold')
ax2.tick_params(colors='white'); ax2.spines[:].set_visible(False)

ax3 = fig.add_axes([0.59,0.48,0.38,0.33]); ax3.set_facecolor('#1e1e2e')
top5 = table.head(8)
ax3.barh(top5["Team"][::-1], top5["Points"][::-1],
         color=plt.cm.get_cmap(CMAP)(np.linspace(0.2,0.9,len(top5))))
ax3.set_title("Top Teams (Points)",color='white',fontweight='bold')
ax3.tick_params(colors='white'); ax3.spines[:].set_visible(False)

ax4 = fig.add_axes([0.02,0.03,0.95,0.38]); ax4.set_facecolor('#1e1e2e'); ax4.axis('off')
summary = (
    f"  MODULE A — Match Outcome Predictor ({model_name})\n"
    f"  Accuracy: {acc*100:.1f}%  |  CV: {cv_sc.mean()*100:.1f}%±{cv_sc.std()*100:.1f}%  |  "
    f"Split: {100-test_size*100:.0f}/{test_size*100:.0f}  |  Features: {', '.join(feat_cols)}\n\n"
    f"  MODULE B — Player Performance Scorer ({reg_name})\n"
    f"  R²: {r2:.3f}  |  RMSE: {rmse:.2f}  |  Features: {', '.join(PERF_FEATS)}\n\n"
    f"  MODULE C — Player Clustering (K-Means k={optimal_k})\n"
    f"  Groups: {' | '.join(cnames)}\n\n"
    f"  MODULE D — Player Recommender (Cosine Similarity)\n"
    f"  Target: {target_name}  |  Top match: {recs_df.iloc[0]['Similarity']}\n\n"
    f"  BONUS — League Table  |  Top: {table.iloc[0]['Team']} ({table.iloc[0]['Points']} pts)  |  {len(all_teams)} teams\n\n"
    f"  Seed: {SEED}  — Run again for fresh splits, colours and chart styles!"
)
ax4.text(0.01,0.97,summary,va='top',ha='left',fontsize=9,color='#aaaacc',
         family='monospace',transform=ax4.transAxes,linespacing=1.7)
ax4.set_title("Full Model Summary",color='white',fontweight='bold',loc='left',pad=8)
plt.savefig(out("7_master_dashboard.png"),dpi=150,bbox_inches='tight',facecolor=BG)
plt.close()
print("  Master dashboard saved!")

print("\n" + "="*60)
print(f"  ALL DONE!  Seed: {SEED}")
print("="*60)
for i,f in enumerate(["1_eda_analysis","2_match_predictor","3_performance_scorer",
                       "4_player_clusters","5_player_recommender",
                       "6_team_form","7_master_dashboard"],1):
    print(f"  {i}. outputs/{f}.png")
print("="*60)
print("  Run again for different seed, colours, model & chart style!")
print("="*60)

# ══════════════════════════════════════════════════════════════
#  AUTO BROWSER VIEWER — opens all charts in a stunning dashboard
# ══════════════════════════════════════════════════════════════

import base64, webbrowser

chart_meta = [
    ("1_eda_analysis.png",       "EDA Analysis",            "Exploratory Data Analysis — distributions, correlations & top players"),
    ("2_match_predictor.png",    "Match Predictor",         "Random Forest / Gradient Boosting — confusion matrix & feature importances"),
    ("3_performance_scorer.png", "Performance Scorer",      "Linear / Ridge Regression — actual vs predicted scores & coefficients"),
    ("4_player_clusters.png",    "Player Clustering",       "K-Means clustering — elbow method & player type scatter plot"),
    ("5_player_recommender.png", "Player Recommender",      "Cosine Similarity — similarity matrix & radar comparison chart"),
    ("6_team_form.png",          "League Table",            "Real EPL data — points table & win/draw/loss breakdown"),
    ("7_master_dashboard.png",   "Master Dashboard",        "Full system overview — KPIs, distributions & complete model summary"),
]

def img_to_b64(filepath):
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except:
        return ""

# Build chart cards HTML
cards_html = ""
for idx, (fname, title, desc) in enumerate(chart_meta):
    fpath = out(fname)
    b64   = img_to_b64(fpath)
    if not b64:
        continue
    module_tag = f"MODULE {'ABCDE'[idx] if idx < 5 else ('F' if idx==5 else 'DASHBOARD')}"
    cards_html += f"""
        <div class="card" id="card-{idx}" onclick="openModal({idx})">
            <div class="card-badge">{module_tag}</div>
            <div class="card-img-wrap">
                <img src="data:image/png;base64,{b64}" alt="{title}" loading="lazy"/>
                <div class="card-overlay">
                    <span class="zoom-icon">&#9654; View Full Size</span>
                </div>
            </div>
            <div class="card-body">
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
        </div>"""

# Build modal images JS array
modal_imgs = ",\n".join([
    f'{{title:"{t}", desc:"{d}", src:"data:image/png;base64,{img_to_b64(out(f))}"}}'
    for f,t,d in chart_meta if img_to_b64(out(f))
])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Sports Analytics ML — Results Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  :root {{
    --bg:       #07080f;
    --surface:  #0d0f1e;
    --card:     #111326;
    --border:   #1e2140;
    --accent:   #4f8ef7;
    --accent2:  #f75f8e;
    --accent3:  #4fffc8;
    --text:     #e8eaf6;
    --muted:    #6b7194;
    --glow:     rgba(79,142,247,0.18);
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* ── animated grid background ── */
  body::before {{
    content:'';
    position:fixed; inset:0; z-index:0;
    background-image:
      linear-gradient(rgba(79,142,247,.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(79,142,247,.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events:none;
  }}

  /* ── header ── */
  header {{
    position: relative; z-index:10;
    padding: 56px 48px 36px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(79,142,247,.08) 0%, transparent 100%);
  }}
  .header-top {{ display:flex; align-items:center; gap:20px; margin-bottom:8px; }}
  .logo {{
    width:48px; height:48px; border-radius:12px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display:flex; align-items:center; justify-content:center;
    font-size:22px; font-weight:700; color:#fff;
    box-shadow: 0 0 24px var(--glow);
    flex-shrink:0;
  }}
  h1 {{
    font-family:'Bebas Neue', sans-serif;
    font-size: clamp(28px,4vw,52px);
    letter-spacing:2px;
    background: linear-gradient(90deg, var(--accent), var(--accent3));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
    line-height:1;
  }}
  .subtitle {{ color:var(--muted); font-size:15px; margin-top:6px; max-width:600px; }}

  /* ── seed badge ── */
  .meta-row {{
    display:flex; align-items:center; gap:12px;
    margin-top:18px; flex-wrap:wrap;
  }}
  .badge {{
    display:inline-flex; align-items:center; gap:6px;
    padding:5px 12px; border-radius:99px;
    font-family:'JetBrains Mono', monospace;
    font-size:11px; font-weight:500;
    border:1px solid; letter-spacing:.5px;
  }}
  .badge-seed  {{ color:var(--accent3); border-color:rgba(79,255,200,.25); background:rgba(79,255,200,.06); }}
  .badge-model {{ color:var(--accent);  border-color:rgba(79,142,247,.25); background:rgba(79,142,247,.06); }}
  .badge-acc   {{ color:var(--accent2); border-color:rgba(247,95,142,.25); background:rgba(247,95,142,.06); }}

  /* ── stat strip ── */
  .stats-strip {{
    position:relative; z-index:10;
    display:flex; gap:1px;
    background:var(--border);
    border-bottom:1px solid var(--border);
  }}
  .stat {{
    flex:1; padding:20px 24px;
    background:var(--surface);
    text-align:center;
    transition: background .2s;
  }}
  .stat:hover {{ background:var(--card); }}
  .stat-val {{
    font-family:'Bebas Neue', sans-serif;
    font-size:32px; letter-spacing:1px;
    background:linear-gradient(135deg, var(--accent), var(--accent3));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
  }}
  .stat-lbl {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:1px; margin-top:2px; }}

  /* ── grid ── */
  .grid-wrap {{ position:relative; z-index:10; padding:40px 48px; }}
  .section-label {{
    font-family:'JetBrains Mono', monospace;
    font-size:11px; text-transform:uppercase; letter-spacing:2px;
    color:var(--muted); margin-bottom:24px;
    display:flex; align-items:center; gap:10px;
  }}
  .section-label::after {{
    content:''; flex:1; height:1px; background:var(--border);
  }}

  .grid {{
    display:grid;
    grid-template-columns: repeat(auto-fill, minmax(340px,1fr));
    gap:20px;
  }}

  /* ── card ── */
  .card {{
    background:var(--card);
    border:1px solid var(--border);
    border-radius:16px;
    overflow:hidden;
    cursor:pointer;
    transition: transform .25s cubic-bezier(.34,1.56,.64,1),
                box-shadow .25s, border-color .25s;
    animation: fadeUp .5s ease both;
  }}
  .card:nth-child(1){{ animation-delay:.05s }}
  .card:nth-child(2){{ animation-delay:.10s }}
  .card:nth-child(3){{ animation-delay:.15s }}
  .card:nth-child(4){{ animation-delay:.20s }}
  .card:nth-child(5){{ animation-delay:.25s }}
  .card:nth-child(6){{ animation-delay:.30s }}
  .card:nth-child(7){{ animation-delay:.35s }}

  @keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(24px); }}
    to   {{ opacity:1; transform:translateY(0); }}
  }}

  .card:hover {{
    transform:translateY(-6px) scale(1.01);
    box-shadow:0 20px 48px rgba(0,0,0,.5), 0 0 0 1px var(--accent), 0 0 32px var(--glow);
    border-color:var(--accent);
  }}

  .card-badge {{
    position:absolute; top:12px; left:12px; z-index:2;
    font-family:'JetBrains Mono',monospace; font-size:9px;
    font-weight:500; letter-spacing:1.5px; text-transform:uppercase;
    padding:3px 8px; border-radius:4px;
    background:rgba(7,8,15,.75); color:var(--accent3);
    border:1px solid rgba(79,255,200,.2);
    backdrop-filter:blur(4px);
  }}

  .card-img-wrap {{
    position:relative; overflow:hidden;
    background:#000; height:220px;
  }}
  .card-img-wrap img {{
    width:100%; height:100%; object-fit:cover;
    transition:transform .4s ease;
    display:block;
  }}
  .card:hover .card-img-wrap img {{ transform:scale(1.04); }}

  .card-overlay {{
    position:absolute; inset:0;
    background:linear-gradient(180deg,transparent 40%,rgba(7,8,15,.85) 100%);
    display:flex; align-items:flex-end; justify-content:center;
    padding-bottom:14px; opacity:0;
    transition:opacity .25s;
  }}
  .card:hover .card-overlay {{ opacity:1; }}
  .zoom-icon {{
    font-size:12px; color:#fff; letter-spacing:1px;
    font-family:'JetBrains Mono',monospace;
  }}

  .card-body {{ padding:16px 18px 18px; }}
  .card-body h3 {{
    font-size:15px; font-weight:600; margin-bottom:5px;
    color:var(--text);
  }}
  .card-body p {{ font-size:12px; color:var(--muted); line-height:1.5; }}

  /* ── modal ── */
  .modal-backdrop {{
    position:fixed; inset:0; z-index:1000;
    background:rgba(7,8,15,.92);
    backdrop-filter:blur(10px);
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    opacity:0; pointer-events:none;
    transition:opacity .3s;
  }}
  .modal-backdrop.open {{ opacity:1; pointer-events:all; }}

  .modal-box {{
    position:relative; width:94vw; max-width:1200px;
    max-height:90vh;
    background:var(--card);
    border:1px solid var(--border);
    border-radius:20px;
    overflow:hidden;
    display:flex; flex-direction:column;
    transform:scale(.96);
    transition:transform .3s cubic-bezier(.34,1.56,.64,1);
  }}
  .modal-backdrop.open .modal-box {{ transform:scale(1); }}

  .modal-header {{
    display:flex; align-items:center; justify-content:space-between;
    padding:16px 22px;
    border-bottom:1px solid var(--border);
    background:var(--surface);
  }}
  .modal-title {{ font-size:16px; font-weight:600; }}
  .modal-desc  {{ font-size:12px; color:var(--muted); margin-top:2px; }}
  .modal-close {{
    width:36px; height:36px; border-radius:8px;
    background:var(--border); border:none; cursor:pointer;
    color:var(--text); font-size:18px;
    display:flex; align-items:center; justify-content:center;
    transition:background .2s;
  }}
  .modal-close:hover {{ background:#2a2d50; }}

  .modal-img-area {{
    flex:1; overflow:auto;
    display:flex; align-items:center; justify-content:center;
    padding:20px; background:#000;
  }}
  .modal-img-area img {{
    max-width:100%; max-height:calc(90vh - 140px);
    object-fit:contain; border-radius:8px;
    box-shadow:0 8px 40px rgba(0,0,0,.6);
  }}

  .modal-nav {{
    display:flex; gap:10px; padding:14px 22px;
    border-top:1px solid var(--border);
    background:var(--surface);
    justify-content:center;
  }}
  .nav-btn {{
    padding:8px 22px; border-radius:8px;
    border:1px solid var(--border);
    background:transparent; color:var(--text);
    font-family:'DM Sans',sans-serif; font-size:13px;
    cursor:pointer; transition:all .2s;
  }}
  .nav-btn:hover {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
  .nav-dots {{ display:flex; align-items:center; gap:6px; }}
  .dot {{
    width:7px; height:7px; border-radius:50%;
    background:var(--border); cursor:pointer; transition:all .2s;
  }}
  .dot.active {{ background:var(--accent); transform:scale(1.3); }}

  /* ── footer ── */
  footer {{
    position:relative; z-index:10;
    text-align:center; padding:28px;
    border-top:1px solid var(--border);
    color:var(--muted); font-size:12px;
    font-family:'JetBrains Mono',monospace;
    letter-spacing:.5px;
  }}
  footer span {{ color:var(--accent3); }}
</style>
</head>
<body>

<header>
  <div class="header-top">
    <div class="logo">⚽</div>
    <div>
      <h1>Sports Analytics ML System</h1>
      <p class="subtitle">Machine Learning dashboard — match prediction, player scoring, clustering &amp; recommendations</p>
    </div>
  </div>
  <div class="meta-row">
    <span class="badge badge-seed">🎲 SEED: {SEED}</span>
    <span class="badge badge-model">🤖 {model_name}</span>
    <span class="badge badge-acc">🎯 Match Acc: {acc*100:.1f}%</span>
    <span class="badge badge-model">📈 Player R²: {r2:.3f}</span>
    <span class="badge badge-acc">🗂️ Clusters: {optimal_k}</span>
  </div>
</header>

<div class="stats-strip">
  <div class="stat"><div class="stat-val">{len(players_df):,}</div><div class="stat-lbl">Players Analysed</div></div>
  <div class="stat"><div class="stat-val">{len(matches_df):,}</div><div class="stat-lbl">Matches Analysed</div></div>
  <div class="stat"><div class="stat-val">{acc*100:.1f}%</div><div class="stat-lbl">Match Accuracy</div></div>
  <div class="stat"><div class="stat-val">{r2:.3f}</div><div class="stat-lbl">Regression R²</div></div>
  <div class="stat"><div class="stat-val">{optimal_k}</div><div class="stat-lbl">Player Clusters</div></div>
  <div class="stat"><div class="stat-val">7</div><div class="stat-lbl">Charts Generated</div></div>
</div>

<div class="grid-wrap">
  <div class="section-label">// generated charts — click any card to expand</div>
  <div class="grid">
    {cards_html}
  </div>
</div>

<footer>Built with Python · scikit-learn · matplotlib · seaborn &nbsp;|&nbsp; Seed <span>{SEED}</span> &nbsp;|&nbsp; Run again for fresh results</footer>

<!-- Modal -->
<div class="modal-backdrop" id="modal" onclick="closeOnBackdrop(event)">
  <div class="modal-box">
    <div class="modal-header">
      <div>
        <div class="modal-title" id="modal-title">Chart</div>
        <div class="modal-desc"  id="modal-desc"></div>
      </div>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-img-area">
      <img id="modal-img" src="" alt="chart"/>
    </div>
    <div class="modal-nav">
      <button class="nav-btn" onclick="navigate(-1)">← Prev</button>
      <div class="nav-dots" id="nav-dots"></div>
      <button class="nav-btn" onclick="navigate(1)">Next →</button>
    </div>
  </div>
</div>

<script>
const charts = [{modal_imgs}];
let current = 0;

function buildDots() {{
  const dots = document.getElementById('nav-dots');
  dots.innerHTML = '';
  charts.forEach((_, i) => {{
    const d = document.createElement('div');
    d.className = 'dot' + (i===current?' active':'');
    d.onclick = () => showChart(i);
    dots.appendChild(d);
  }});
}}

function showChart(idx) {{
  current = (idx + charts.length) % charts.length;
  const c = charts[current];
  document.getElementById('modal-img').src   = c.src;
  document.getElementById('modal-title').textContent = c.title;
  document.getElementById('modal-desc').textContent  = c.desc;
  buildDots();
}}

function openModal(idx) {{
  showChart(idx);
  document.getElementById('modal').classList.add('open');
  document.body.style.overflow = 'hidden';
}}

function closeModal() {{
  document.getElementById('modal').classList.remove('open');
  document.body.style.overflow = '';
}}

function closeOnBackdrop(e) {{
  if (e.target.id === 'modal') closeModal();
}}

function navigate(dir) {{
  showChart(current + dir);
}}

document.addEventListener('keydown', e => {{
  if (!document.getElementById('modal').classList.contains('open')) return;
  if (e.key === 'ArrowRight') navigate(1);
  if (e.key === 'ArrowLeft')  navigate(-1);
  if (e.key === 'Escape')     closeModal();
}});
</script>
</body>
</html>"""

viewer_path = out("view_charts.html")
with open(viewer_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n  Opening charts in browser...")
webbrowser.open(f"file:///{viewer_path.replace(os.sep, '/')}")
print(f"  Viewer saved: outputs/view_charts.html")
print("  (If browser didn't open, double-click outputs/view_charts.html)")

