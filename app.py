# ───────── VaporIQ Galaxy Dashboard • v10 ─────────
import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns, plotly.express as px
from pathlib import Path
import base64, textwrap, datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, confusion_matrix, precision_score,
                             recall_score, accuracy_score, f1_score,
                             r2_score, mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# ───────── Page + global CSS ─────────
st.set_page_config(page_title="VaporIQ Galaxy", layout="wide")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# starfield background
star = Path("starfield.png")
if star.exists():
    st.markdown(
        textwrap.dedent(
            f"""
        <style>
        body::before {{
          content:""; position:fixed; inset:0; z-index:-4; pointer-events:none;
          background:url("data:image/png;base64,{base64.b64encode(star.read_bytes()).decode()}") repeat;
          background-size:600px; opacity:.35; animation:starDrift 240s linear infinite;
        }}
        @keyframes starDrift {{
          0%   {{transform:translate3d(0,0,0)}} 
          100% {{transform:translate3d(-2000px,1500px,0)}}
        }}
        </style>"""
        ),
        unsafe_allow_html=True,
    )

# smoke layers (class names defined in style.css)
st.markdown('<div class="smoke-layer"></div>', unsafe_allow_html=True)
st.markdown('<div class="smoke-layer-2"></div>', unsafe_allow_html=True)

# watermark
with open("vape_watermark.png", "rb") as f:
    wm_b64 = base64.b64encode(f.read()).decode()
st.markdown(
    f"<img src='data:image/png;base64,{wm_b64}' "
    "style='position:fixed;bottom:15px;right:15px;width:110px;opacity:.8;z-index:1;'/>",
    unsafe_allow_html=True,
)

# ───────── Data loader (with weekly aggregation fix) ─────────
@st.cache_data
def load_data():
    users = pd.read_csv("users_synthetic_enriched.csv")
    trends = pd.read_csv("flavor_trends.csv")
    trends["Date"] = pd.to_datetime(trends["Date"])

    # ensure numerics
    num_cols = [
        "Age",
        "SweetLike",
        "MentholLike",
        "PodsPerWeek",
        "AvgSessionTimeMin",
        "ReferralCount",
        "ReorderRate",
        "SubscriptionTenure",
        "FlavourExplorationRate",
        "NicotineToleranceLevel",
        "AvgPodsPerOrder",
        "SleepHours",
        "StressLevelScale",
        "FlavorBuzzScore",
        "SocialMentions_30D",
    ]
    users[num_cols] = users[num_cols].apply(pd.to_numeric, errors="coerce")

    # -------- Weekly aggregation for extra predictors --------
    # synthetic WeekStart if no date column exists
    base_day = pd.Timestamp("2024-01-01")  # Mon
    users["WeekStart"] = base_day + pd.to_timedelta(users.index // 500, unit="W")

    weekly_metrics = (
        users.groupby("WeekStart")[
            ["PodsPerWeek", "AvgPodsPerOrder", "FlavorBuzzScore", "SocialMentions_30D"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"WeekStart": "Date"})
    )

    # merge + forward-fill to ensure complete series
    trends_full = (
        trends.merge(weekly_metrics, on="Date", how="left")
        .sort_values("Date")
        .fillna(method="ffill")
    )

    return users, trends_full


users_df, trends_df = load_data()

# feature sets
core = ["Age", "SweetLike", "MentholLike", "PodsPerWeek"]
extra = [
    "AvgSessionTimeMin",
    "ReferralCount",
    "ReorderRate",
    "SubscriptionTenure",
    "FlavourExplorationRate",
    "NicotineToleranceLevel",
    "SleepHours",
    "StressLevelScale",
]
all_num = core + extra

# ensure Cluster column
if "Cluster" not in users_df.columns:
    users_df["Cluster"] = KMeans(4, random_state=42).fit_predict(
        MinMaxScaler().fit_transform(users_df[all_num])
    )

# ───────── Tabs ─────────
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Visualization", "TasteDNA", "Forecasting", "Micro-Batch"]
)

# ╭────────────────────────── 1. DATA-VIS ╮
with viz:
    st.header("📊 Data Visualization Explorer")
    g_sel = st.sidebar.multiselect(
        "Gender", users_df["Gender"].unique(), users_df["Gender"].unique()
    )
    c_sel = st.sidebar.multiselect(
        "Purchase Channel",
        users_df["PurchaseChannel"].unique(),
        users_df["PurchaseChannel"].unique(),
    )
    df = users_df[users_df["Gender"].isin(g_sel) & users_df["PurchaseChannel"].isin(c_sel)]

    # Hex density
    st.plotly_chart(
        px.density_heatmap(
            df,
            x="Age",
            y="PodsPerWeek",
            nbinsx=30,
            nbinsy=15,
            color_continuous_scale="magma",
            title="Density of Consumption by Age",
        ),
        use_container_width=True,
    )
    st.caption("Hot-spots reveal which age bands are heavy users.")

    # Bar AvgSessionTime by SignupChannel
    st.plotly_chart(
        px.bar(
            df,
            x="SignupChannel",
            y="AvgSessionTimeMin",
            color="SignupChannel",
            title="Avg Session Time by Signup Channel",
        ),
        use_container_width=True,
    )
    st.caption("Engagement depth differs across acquisition channels.")

    # Box ReorderRate vs PrimaryFlavourNote
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="PrimaryFlavourNote", y="ReorderRate", ax=ax)
    ax.set_title("Reorder-Rate by Primary Flavour Note")
    st.pyplot(fig); plt.close(fig)
    st.caption("Certain flavour notes drive stronger repeat-buy behaviour.")

    # (feel free to re-insert your original scatter / correlation / trend lines here)

# ╭────────────────────────── 2. TASTEDNA ╮
with taste_tab:
    st.header("🔮 TasteDNA")
    mode = st.radio("Mode", ["Classification", "Clustering"], horizontal=True)

    if mode == "Classification":
        algo = st.selectbox(
            "Classifier", ["KNN", "Decision Tree", "Random Forest", "Gradient Boosting"]
        )
        X, y = users_df[all_num], users_df["SubscribeIntent"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        clf_map = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        model = clf_map[algo].fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        for label, func in [
            ("Precision", precision_score),
            ("Recall", recall_score),
            ("Accuracy", accuracy_score),
            ("F1", f1_score),
        ]:
            st.metric(label, f"{func(y_te, y_pred):.2f}")

        fig, ax = plt.subplots()
        sns.heatmap(
            confusion_matrix(y_te, y_pred),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
        )
        st.pyplot(fig); plt.close(fig)

    else:
        k = st.slider("k clusters", 2, 10, 4)
        scaled = MinMaxScaler().fit_transform(users_df[all_num])
        inert = [KMeans(i, random_state=42).fit(scaled).inertia_ for i in range(2, 11)]
        fig, ax = plt.subplots()
        ax.plot(range(2, 11), inert, marker="o")
        ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow Curve")
        st.pyplot(fig); plt.close(fig)

        km = KMeans(k, random_state=42).fit(scaled)
        users_df["Cluster"] = km.labels_
        st.metric("Silhouette", f"{silhouette_score(scaled, km.labels_):.3f}")
        st.dataframe(users_df.groupby("Cluster")[all_num].mean().round(2))

# ╭────────────────────────── 3. FORECAST ╮
with forecast_tab:
    st.header("📈 Forecasting")
    predictor = st.selectbox(
        "Predictor",
        ["PodsPerWeek", "AvgPodsPerOrder", "FlavorBuzzScore", "SocialMentions_30D"],
    )
    reg_name = st.selectbox("Regressor", ["Linear", "Ridge", "Lasso", "Decision Tree"])
    reg = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=0.01),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    }[reg_name]
    X = np.arange(len(trends_df)).reshape(-1, 1)
    y = trends_df[predictor].values
    cut = int(0.8 * len(X))
    reg.fit(X[:cut], y[:cut])
    y_pred = reg.predict(X[cut:])
    st.metric("R²", f"{r2_score(y[cut:], y_pred):.3f}")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y[cut:], y_pred)):.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y[cut:], y_pred, alpha=0.6)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    st.pyplot(fig); plt.close(fig)

# ╭────────────────────────── 4. APRIORI ╮
with rules_tab:
    st.header("🧩 Micro-Batch & Drops (Apriori)")
    sup = st.slider("Support", 0.02, 0.4, 0.05, 0.01)
    conf = st.slider("Confidence", 0.05, 1.0, 0.3, 0.05)

    basket = pd.concat(
        [
            users_df["FlavourFamilies"].str.get_dummies(sep=",").astype(bool),
            pd.get_dummies(users_df["PurchaseChannel"], prefix="Chan").astype(bool),
            pd.get_dummies(users_df["PrimaryFlavourNote"], prefix="Note").astype(bool),
        ],
        axis=1,
    )

    rules = association_rules(
        apriori(basket, min_support=sup, use_colnames=True),
        metric="confidence",
        min_threshold=conf,
    )
    st.dataframe(
        rules.sort_values("confidence", ascending=False).head(10)
        if not rules.empty
        else pd.DataFrame()
    )
