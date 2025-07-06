
# VaporIQ Galaxy Dashboard v11
import streamlit as st, pandas as pd, numpy as np, plotly.express as px
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, accuracy_score, f1_score,
                             confusion_matrix, silhouette_score, r2_score, mean_squared_error)
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules
import base64, textwrap, datetime, io

# --- Theme toggle ---
st.sidebar.radio("Theme", ["Galaxy","Light"], index=0)

# --- Load data ---
@st.cache_data
def load():
    users = pd.read_csv("users_synthetic_enriched.csv")
    trends = pd.read_csv("flavor_trends_enriched_filled.csv")
    trends['Date'] = pd.to_datetime(trends['Date'])
    return users, trends
users_df, trends_df = load()

# Helper: behaviour columns
core = ["Age","SweetLike","MentholLike","PodsPerWeek"]
behaviour = ["AvgSessionTimeMin","ReferralCount","ReorderRate","SubscriptionTenure",
             "FlavourExplorationRate","NicotineToleranceLevel","SleepHours","StressLevelScale"]
weekly_predictors = ["PodsPerWeek","AvgPodsPerOrder","FlavorBuzzScore","SocialMentions_30D"]

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Flavor Forecasting","TasteDNA","MoodSync","Micro-Batch","Compliance"])

# ========= TAB 1 =========
with tab1:
    st.header("Flavor Forecasting Engine")
    pred = st.sidebar.selectbox("Weekly predictor", weekly_predictors)
    model_name = st.selectbox("Regressor", ["Linear","Ridge","Lasso","DecisionTree"])
    reg_map = {"Linear":LinearRegression(),
               "Ridge":Ridge(),
               "Lasso":Lasso(alpha=0.01),
               "DecisionTree":DecisionTreeRegressor(max_depth=5)}
    reg = reg_map[model_name]
    X = np.arange(len(trends_df)).reshape(-1,1)
    y = trends_df[pred].values
    cut = int(0.8*len(X))
    reg.fit(X[:cut], y[:cut])
    y_pred = reg.predict(X[cut:])
    st.metric("R²", f"{r2_score(y[cut:], y_pred):.3f}")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y[cut:], y_pred)):.2f}")
    fig = px.scatter(x=y[cut:], y=y_pred, labels={'x':'Actual','y':'Predicted'},
                     title="Parity Plot")
    fig.add_shape(type='line', x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
    st.plotly_chart(fig,use_container_width=True)
    # next–hit flavour
    slopes = {c: np.polyfit(np.arange(4), trends_df[c].tail(4),1)[0]
              for c in trends_df.columns if c not in weekly_predictors and c!="Date"}
    next_hit = max(slopes, key=slopes.get)
    st.success(f"Next‑hit flavour forecast: **{next_hit}**")

# ========= TAB 2 =========
with tab2:
    st.header("TasteDNA Personalization")
    sub = st.radio("Sub‑panel", ["Classification","Clustering"], horizontal=True)
    X = users_df[core+behaviour]
    if sub=="Classification":
        algo = st.selectbox("Classifier",["KNN","DecisionTree","RandomForest","GradientBoost"])
        clf_map = {"KNN":KNeighborsClassifier(),
                   "DecisionTree":DecisionTreeClassifier(),
                   "RandomForest":RandomForestClassifier(),
                   "GradientBoost":GradientBoostingClassifier()}
        y = users_df["SubscribeIntent"]
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,stratify=y, test_size=0.25, random_state=42)
        model=clf_map[algo].fit(X_tr,y_tr)
        y_pred = model.predict(X_te)
        for lbl,f in [("Precision",precision_score),("Recall",recall_score),
                      ("Accuracy",accuracy_score),("F1",f1_score)]:
            st.metric(lbl, f"{f(y_te,y_pred):.2f}")
        sns.heatmap(confusion_matrix(y_te,y_pred),annot=True,fmt="d",cmap="Blues")
        st.pyplot(plt.gcf()); plt.clf()
    else:
        k = st.slider("k",2,10,4)
        km = KMeans(k, random_state=42).fit(X)
        users_df["Cluster"]=km.labels_
        inertias = [KMeans(i,random_state=42).fit(X).inertia_ for i in range(2,11)]
        plt.plot(range(2,11), inertias, marker="o"); plt.title("Elbow Curve")
        st.pyplot(plt.gcf()); plt.clf()
        st.metric("Silhouette", f"{silhouette_score(X, km.labels_):.3f}")
        st.dataframe(users_df.groupby("Cluster")[behaviour].mean().round(2))

# ========= TAB 3 =========
with tab3:
    st.header("MoodSync Blends")
    mood = st.selectbox("Mood", ["Happy","Anxious","LowEnergy","Energetic"])
    stress = st.slider("Stress",1,10,5)
    sleep = st.slider("Sleep Hours",4,9,6)
    # Simple rule demo
    if stress>7: flavour="Menthol / Vanilla"
    elif mood=="LowEnergy": flavour="Citrus"
    elif mood=="Happy": flavour="Berry"
    else: flavour="Custard"
    st.success(f"Recommended flavour: **{flavour}**")
    st.write("*(Demo rules only)*")

# ========= TAB 4 =========
with tab4:
    st.header("Micro‑Batch & Limited Drops")
    sup = st.slider("Support",0.01,0.2,0.03,0.01)
    conf = st.slider("Confidence",0.1,1.0,0.4,0.05)
    basket = pd.concat([
        users_df["FlavourFamilies"].str.get_dummies(sep=","),
        pd.get_dummies(users_df["PrimaryFlavourNote"],prefix="Note"),
        pd.get_dummies(users_df["Cluster"],prefix="C")],axis=1).astype(bool)
    rules=association_rules(apriori(basket, min_support=sup,use_colnames=True),
                            metric="confidence",min_threshold=conf)
    if rules.empty: st.info("No rules; lower thresholds.")
    else: st.dataframe(rules.nlargest(10,"confidence")[["antecedents","consequents","confidence","lift"]])

# ========= TAB 5 =========
with tab5:
    st.header("Compliance & Nicotine Control")
    country = st.selectbox("Country", users_df["CountryRegCode"].unique())
    sub_df = users_df[users_df["CountryRegCode"]==country]
    pct = (sub_df["OverLimitWarning"].mean()*100).round(1)
    st.metric("% Over Limit", f"{pct}%")
    fig = px.scatter(sub_df, x="PodsPerWeek", y="NicotineToleranceLevel",
                     trendline="ols", title="Tolerance vs Usage")
    st.plotly_chart(fig,use_container_width=True)
    if st.button("Generate taper CSV"):
        taper = sub_df[["UserID","NicotineToleranceLevel"]].copy()
        taper["Week1"] = taper["NicotineToleranceLevel"]*0.9
        taper["Week2"] = taper["NicotineToleranceLevel"]*0.8
        taper["Week3"] = taper["NicotineToleranceLevel"]*0.7
        taper["Week4"] = taper["NicotineToleranceLevel"]*0.6
        st.download_button("Download", taper.to_csv(index=False).encode(),
                           "taper_plan.csv","text/csv")
