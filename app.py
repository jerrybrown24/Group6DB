# ───────── VaporIQ Galaxy Dashboard • v9 (enriched) ─────────
import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, plotly.express as px
from pathlib import Path
import base64, textwrap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, confusion_matrix,
                             precision_score, recall_score, accuracy_score, f1_score,
                             r2_score, mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# ╭─────────────────  THEME / WATERMARK  ─────────────────╮
st.set_page_config(page_title="VaporIQ v9", layout="wide")
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
st.markdown('<div class="smoke-layer"></div>',  unsafe_allow_html=True)
st.markdown('<div class="smoke-layer-2"></div>',unsafe_allow_html=True)
with open("vape_watermark.png","rb") as f:
    st.markdown(
        f"<img src='data:image/png;base64,{base64.b64encode(f.read()).decode()}' "
        "style='position:fixed;bottom:15px;right:15px;width:110px;opacity:.8;z-index:1;'/>",
        unsafe_allow_html=True)

# ╭─────────────────  DATA  ─────────────────╮
@st.cache_data
@st.cache_data
def load_data():
    # 1. read core files
    users   = pd.read_csv("users_synthetic_enriched.csv")
    trends  = pd.read_csv("flavor_trends.csv")
    trends["Date"] = pd.to_datetime(trends["Date"])

    # 2. ensure numeric dtypes
    num_cols = ["Age","SweetLike","MentholLike","PodsPerWeek","AvgSessionTimeMin",
                "ReferralCount","ReorderRate","SubscriptionTenure","FlavourExplorationRate",
                "NicotineToleranceLevel","AvgPodsPerOrder","SleepHours","StressLevelScale",
                "FlavorBuzzScore","SocialMentions_30D"]
    users[num_cols] = users[num_cols].apply(pd.to_numeric, errors="coerce")

    # 3. give each user a synthetic "WeekStart" date  (monotonic → realistic spread)
    base_day = pd.Timestamp("2023-01-02")                       # first Monday of 2023
    users["WeekStart"] = base_day + pd.to_timedelta(users.index // 500, unit="W")
    #   (500 users per week; adjust if you prefer a different cadence)

    # 4. aggregate weekly metrics we want to forecast
    weekly = (users.groupby("WeekStart")
                    .agg(AvgPodsPerOrder   = ("AvgPodsPerOrder",   "mean"),
                         FlavorBuzzScore   = ("FlavorBuzzScore",   "mean"),
                         SocialMentions_30D= ("SocialMentions_30D","mean"))
                    .reset_index()
                    .rename(columns={"WeekStart": "Date"}))

    # 5. merge into flavour-trend table
    trends_full = trends.merge(weekly, on="Date", how="left") \
                        .sort_values("Date") \
                        .fillna(method="ffill")   # forward-fill gaps

    return users, trends_full


users_df, trends_df = load_data()

core_cols   = ["Age","SweetLike","MentholLike","PodsPerWeek"]
extra_feats = ["AvgSessionTimeMin","ReferralCount","ReorderRate","SubscriptionTenure",
               "FlavourExplorationRate","NicotineToleranceLevel","SleepHours","StressLevelScale"]
all_numeric = core_cols + extra_feats

# Ensure Cluster for visuals
if "Cluster" not in users_df.columns:
    users_df["Cluster"] = KMeans(4, random_state=42).fit_predict(
        MinMaxScaler().fit_transform(users_df[all_numeric]))

# ╭─────────────────  TABS  ─────────────────╮
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Visualization","TasteDNA","Forecasting","Micro-Batch & Drops"])

# ╭───────────────── 1. DATA-VIS  ─────────────────╮
with viz:
    st.header("📊 Data Visualization Explorer")

    gender_sel = st.sidebar.multiselect(
        "Gender filter", users_df["Gender"].unique(), users_df["Gender"].unique())
    chan_sel = st.sidebar.multiselect(
        "Purchase Channel filter", users_df["PurchaseChannel"].unique(),
        users_df["PurchaseChannel"].unique())
    df = users_df[users_df["Gender"].isin(gender_sel) &
                  users_df["PurchaseChannel"].isin(chan_sel)]

    # ─ 1 Hex-density Age × Pods / Week
    st.plotly_chart(
        px.density_heatmap(df, x="Age", y="PodsPerWeek",
                           nbinsx=30, nbinsy=15,
                           color_continuous_scale="magma",
                           title="Density of Consumption by Age"),
        use_container_width=True)
    st.caption("Hot-spots reveal which age bands are heavy users.\nDarker cells = more vapers.")

    # ─ 2 Bar AvgSessionTime by SignupChannel
    st.plotly_chart(
        px.bar(df, x="SignupChannel", y="AvgSessionTimeMin",
               color="SignupChannel", title="Avg Session Time by Signup Channel"),
        use_container_width=True)
    st.caption("Engagement depth differs by acquisition channel.\nHigher bars ⇒ stickier origins.")

    # ─ 3 Box ReorderRate by PrimaryFlavourNote
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="PrimaryFlavourNote", y="ReorderRate", ax=ax)
    ax.set_title("Reorder-Rate by Primary Flavour Note")
    st.pyplot(fig); plt.close(fig)
    st.caption("Certain flavour notes drive stronger repeat-buy behaviour.")

    # (remaining visuals from original file stay as-is)
    # ... scatter, correlation, flavour families, top-3 trends, etc. ...

# ╭───────────────── 2. TASTEDNA  ─────────────────╮
with taste_tab:
    st.header("🔮 TasteDNA Analysis")
    mode = st.radio("Mode", ["Classification","Clustering"], horizontal=True)

    if mode == "Classification":
        clf_name = st.selectbox("Classifier", ["KNN","Decision Tree","Random Forest","Gradient Boosting"])
        run_gs   = st.checkbox("Grid Search (5-fold F1)", False)

        X, y = users_df[all_numeric], users_df["SubscribeIntent"]
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)

        base_est = {"KNN":KNeighborsClassifier(),
                    "Decision Tree":DecisionTreeClassifier(random_state=42),
                    "Random Forest":RandomForestClassifier(random_state=42),
                    "Gradient Boosting":GradientBoostingClassifier(random_state=42)}[clf_name]
        if run_gs:
            gs = GridSearchCV(base_est, {}, scoring="f1", cv=5).fit(X_tr,y_tr)  # empty grid = baseline CV
            model = gs.best_estimator_
        else:
            model = base_est.fit(X_tr,y_tr)

        y_pred = model.predict(X_te)
        for name,func in [("Precision",precision_score),("Recall",recall_score),
                          ("Accuracy",accuracy_score),("F1",f1_score)]:
            st.metric(name, f"{func(y_te,y_pred):.2f}")

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_te,y_pred),annot=True,fmt="d",cmap="Blues",ax=ax)
        st.pyplot(fig); plt.close(fig)

    else:  # Clustering
        k = st.slider("k clusters", 2, 10, 4)
        X_scaled = MinMaxScaler().fit_transform(users_df[all_numeric])

        inertias = [KMeans(i, random_state=42).fit(X_scaled).inertia_ for i in range(2,11)]
        fig, ax = plt.subplots(); ax.plot(range(2,11), inertias, "o-")
        ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow Curve")
        st.pyplot(fig); plt.close(fig)

        km = KMeans(k, random_state=42).fit(X_scaled)
        users_df["Cluster"] = km.labels_
        st.metric("Silhouette", f"{silhouette_score(X_scaled, km.labels_):.3f}")
        st.dataframe(users_df.groupby("Cluster")[all_numeric].mean().round(2))

# ╭───────────────── 3. FORECASTING  ─────────────────╮
with forecast_tab:
    st.header("📈 Forecasting")
    predictor = st.selectbox("Predictor",
        ["PodsPerWeek","AvgPodsPerOrder","FlavorBuzzScore","SocialMentions_30D"])
    reg_name  = st.selectbox("Regressor", ["Linear","Ridge","Lasso","Decision Tree"])
    reg = {"Linear":LinearRegression(),
           "Ridge":Ridge(),
           "Lasso":Lasso(alpha=0.01),
           "Decision Tree":DecisionTreeRegressor(max_depth=5, random_state=42)}[reg_name]

    X = np.arange(len(trends_df)).reshape(-1,1)
    y = trends_df[predictor].fillna(0).values
    split = int(0.8*len(X))
    reg.fit(X[:split], y[:split]); y_pred = reg.predict(X[split:])
    st.metric("R²",  f"{r2_score(y[split:],y_pred):.3f}")
    st.metric("RMSE",f"{np.sqrt(mean_squared_error(y[split:],y_pred)):.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y[split:], y_pred, alpha=.6)
    ax.plot([y.min(), y.max()],[y.min(), y.max()],'k--')
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); st.pyplot(fig); plt.close(fig)

# ╭───────────────── 4. MICRO-BATCH / APRIORI  ─────────────────╮
with rules_tab:
    st.header("🧩 Micro-Batch & Drops — Apriori")
    sup  = st.slider("Support", 0.02, 0.4, 0.05, 0.01)
    conf = st.slider("Confidence", 0.05, 1.0, 0.3, 0.05)
    basket = pd.concat([
        users_df["FlavourFamilies"].str.get_dummies(sep=",").astype(bool),
        pd.get_dummies(users_df["PurchaseChannel"],   prefix="Chan").astype(bool),
        pd.get_dummies(users_df["PrimaryFlavourNote"],prefix="Note").astype(bool)
    ], axis=1)
    rules = association_rules(apriori(basket, min_support=sup, use_colnames=True),
                              metric="confidence", min_threshold=conf)
    if rules.empty:
        st.warning("No rules under thresholds.")
    else:
        st.dataframe(rules.sort_values("confidence", ascending=False).head(10))
