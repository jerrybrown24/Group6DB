# ───────── VaporIQ Galaxy Dashboard • v10-stable ─────────
import streamlit as st, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns, plotly.express as px
from pathlib import Path
import base64, textwrap
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

# ───────── PAGE & THEME ─────────
st.set_page_config(page_title="VaporIQ Galaxy", layout="wide")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.markdown('<div class="smoke-layer"></div>',  unsafe_allow_html=True)
st.markdown('<div class="smoke-layer-2"></div>', unsafe_allow_html=True)
with open("vape_watermark.png","rb") as wf:
    st.markdown(
        f"<img src='data:image/png;base64,{base64.b64encode(wf.read()).decode()}' "
        "style='position:fixed;bottom:15px;right:15px;width:110px;opacity:.8;z-index:1;'/>",
        unsafe_allow_html=True)

# optional star-field
star = Path("starfield.png")
if star.exists():
    st.markdown(
        f"""<style>
        body::before{{content:"";position:fixed;inset:0;z-index:-4;pointer-events:none;
        background:url(data:image/png;base64,{base64.b64encode(star.read_bytes()).decode()}) repeat;
        background-size:600px;opacity:.35;animation:star 240s linear infinite}}
        @keyframes star{{0%{{transform:translate3d(0,0,0)}}100%{{transform:translate3d(-2000px,1500px,0)}}}}
        </style>""",
        unsafe_allow_html=True)

# ───────── DATA LOADER ─────────
@st.cache_data
def load_data():
    users  = pd.read_csv("users_synthetic_enriched.csv")
    trends = pd.read_csv("flavor_trends.csv")
    trends["Date"] = pd.to_datetime(trends["Date"])

    num_cols = ["Age","SweetLike","MentholLike","PodsPerWeek",
                "AvgSessionTimeMin","ReferralCount","ReorderRate","SubscriptionTenure",
                "FlavourExplorationRate","NicotineToleranceLevel","AvgPodsPerOrder",
                "SleepHours","StressLevelScale","FlavorBuzzScore","SocialMentions_30D"]
    users[num_cols] = users[num_cols].apply(pd.to_numeric, errors="coerce")

    # ─ weekly predictors on full Monday index ─
    first, last = trends["Date"].min().normalize(), trends["Date"].max().normalize()
    monday_idx  = pd.date_range(first, last, freq="W-MON")
    users["WeekStart"] = monday_idx[users.index // 500]   # 500 users per week

    weekly = (
        users.groupby("WeekStart")[["PodsPerWeek","AvgPodsPerOrder",
                                    "FlavorBuzzScore","SocialMentions_30D"]]
        .mean()
        .reindex(monday_idx, fill_value=np.nan)            # ensure full index
        .reset_index()
        .rename(columns={"index":"Date"})
        .fillna(method="ffill").fillna(method="bfill")
    )

    trends_full = (trends.merge(weekly, on="Date", how="left")
                          .sort_values("Date")
                          .fillna(method="ffill").fillna(method="bfill"))
    return users, trends_full

users_df, trends_df = load_data()

core  = ["Age","SweetLike","MentholLike","PodsPerWeek"]
extra = ["AvgSessionTimeMin","ReferralCount","ReorderRate","SubscriptionTenure",
         "FlavourExplorationRate","NicotineToleranceLevel","SleepHours","StressLevelScale"]
all_num = core + extra

if "Cluster" not in users_df.columns:
    users_df["Cluster"] = KMeans(4, random_state=42).fit_predict(
        MinMaxScaler().fit_transform(users_df[all_num]))

# ───────── TABS ─────────
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Visualization","TasteDNA","Forecasting","Micro-Batch"])

# 1 ── DATA-VIS
with viz:
    st.header("📊 Explorer")
    st.plotly_chart(
        px.density_heatmap(users_df, x="Age", y="PodsPerWeek",
                           nbinsx=30, nbinsy=15, color_continuous_scale="magma",
                           title="Density of Consumption by Age"),
        use_container_width=True)
    st.plotly_chart(
        px.bar(users_df, x="SignupChannel", y="AvgSessionTimeMin",
               color="SignupChannel", title="Avg Session Time by Signup Channel"),
        use_container_width=True)
    fig, ax = plt.subplots()
    sns.boxplot(data=users_df, x="PrimaryFlavourNote", y="ReorderRate", ax=ax)
    ax.set_title("Reorder-Rate by Primary Flavour Note"); st.pyplot(fig); plt.close(fig)

# 2 ── TASTEDNA
with taste_tab:
    st.header("🔮 TasteDNA")
    if st.radio("Mode",["Classification","Clustering"],horizontal=True)=="Classification":
        algo = st.selectbox("Classifier",["KNN","Decision Tree","Random Forest","Gradient Boosting"])
        X,y  = users_df[all_num], users_df["SubscribeIntent"]
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,stratify=y,test_size=0.25,random_state=42)
        model = {"KNN":KNeighborsClassifier(),
                 "Decision Tree":DecisionTreeClassifier(random_state=42),
                 "Random Forest":RandomForestClassifier(random_state=42),
                 "Gradient Boosting":GradientBoostingClassifier(random_state=42)}[algo]
        y_pred = model.fit(X_tr,y_tr).predict(X_te)
        for lbl,f in [("Precision",precision_score),("Recall",recall_score),
                      ("Accuracy",accuracy_score),("F1",f1_score)]:
            st.metric(lbl,f"{f(y_te,y_pred):.2f}")
        fig, ax = plt.subplots(); sns.heatmap(confusion_matrix(y_te,y_pred),
                                              annot=True,fmt="d",cmap="Blues",ax=ax)
        st.pyplot(fig); plt.close(fig)
    else:
        k = st.slider("k clusters",2,10,4)
        scaled = MinMaxScaler().fit_transform(users_df[all_num])
        inert  = [KMeans(i,random_state=42).fit(scaled).inertia_ for i in range(2,11)]
        fig, ax = plt.subplots(); ax.plot(range(2,11), inert, marker="o"); ax.set_title("Elbow Curve")
        st.pyplot(fig); plt.close(fig)
        km = KMeans(k,random_state=42).fit(scaled); users_df["Cluster"]=km.labels_
        st.metric("Silhouette",f"{silhouette_score(scaled,km.labels_):.3f}")
        st.dataframe(users_df.groupby("Cluster")[all_num].mean().round(2))

# 3 ── FORECASTING (robust)
with forecast_tab:
    st.header("📈 Forecasting")
    predictor = st.selectbox("Predictor",
        ["PodsPerWeek","AvgPodsPerOrder","FlavorBuzzScore","SocialMentions_30D"])
    reg = {"Linear":LinearRegression(),
           "Ridge":Ridge(),
           "Lasso":Lasso(alpha=0.01),
           "Decision Tree":DecisionTreeRegressor(max_depth=5,random_state=42)}[
             st.selectbox("Regressor",["Linear","Ridge","Lasso","Decision Tree"])]
    X_time = np.arange(len(trends_df)).reshape(-1,1).astype(float)
    y_raw  = pd.to_numeric(trends_df[predictor], errors="coerce").astype(float)
    y_filled = (y_raw.interpolate("linear",limit_direction="both")
                      .fillna(method="ffill").fillna(method="bfill")
                      .fillna(y_raw.median()))
    mask = np.isfinite(y_filled); X_c, y_c = X_time[mask], y_filled.values[mask]
    if len(X_c) < 10:
        st.warning(f"Only {len(X_c)} finite points for '{predictor}'."); st.stop()
    cut = int(0.8*len(X_c))
    reg.fit(X_c[:cut], y_c[:cut]); y_pred = reg.predict(X_c[cut:])
    st.metric("R²",f"{r2_score(y_c[cut:],y_pred):.3f}")
    st.metric("RMSE",f"{np.sqrt(mean_squared_error(y_c[cut:],y_pred)):.2f}")
    fig, ax = plt.subplots(); ax.scatter(y_c[cut:],y_pred,alpha=.6)
    ax.plot([y_c.min(),y_c.max()],[y_c.min(),y_c.max()],'k--')
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); st.pyplot(fig); plt.close(fig)

# 4 ── MICRO-BATCH / APRIORI
with rules_tab:
    st.header("🧩 Micro-Batch & Drops")
    sup  = st.slider("Support",0.02,0.4,0.05,0.01)
    conf = st.slider("Confidence",0.05,1.0,0.3,0.05)
    basket = pd.concat([
        users_df["FlavourFamilies"].str.get_dummies(sep=",").astype(bool),
        pd.get_dummies(users_df["PurchaseChannel"], prefix="Chan").astype(bool),
        pd.get_dummies(users_df["PrimaryFlavourNote"], prefix="Note").astype(bool)],axis=1)
    rules = association_rules(apriori(basket,min_support=sup,use_colnames=True),
                              metric="confidence",min_threshold=conf)
    if rules.empty:
        st.info("No rules under current thresholds. Try lowering support/confidence.")
    else:
        st.dataframe(rules.sort_values("confidence",ascending=False).head(10))
