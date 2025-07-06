# ───────── VaporIQ Galaxy Dashboard • v11-hp ─────────
import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path; import base64, datetime, textwrap, warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (precision_score, recall_score, accuracy_score,
                             f1_score, confusion_matrix, silhouette_score,
                             r2_score, mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# statsmodels for Plotly OLS trendline
try:
    import statsmodels.api as sm
except ModuleNotFoundError:
    sm = None
    warnings.warn("statsmodels not installed → OLS trendline disabled.")

# ╭──────────  THEME  ─────────╮  (unchanged)
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

# ╭──────────  DATA  ─────────╮
@st.cache_data
def load():
    users  = pd.read_csv("users_synthetic_enriched.csv")
    trends = pd.read_csv("flavor_trends_enriched_filled.csv")
    trends["Date"] = pd.to_datetime(trends["Date"])
    return users, trends

users_df, trends_df = load()

core  = ["Age","SweetLike","MentholLike","PodsPerWeek"]
behav = ["AvgSessionTimeMin","ReferralCount","ReorderRate","SubscriptionTenure",
         "FlavourExplorationRate","NicotineToleranceLevel","SleepHours","StressLevelScale"]
weekly_predictors = ["PodsPerWeek","AvgPodsPerOrder","FlavorBuzzScore","SocialMentions_30D"]

# ╭──────────  TABS  ─────────╮
viz, taste_tab, forecast_tab, rules_tab = st.tabs(
    ["Data Vis","TasteDNA","Forecasting","Micro-Batch"])

# 1 ── SIMPLE DATA-VIS (unchanged) ──────────────────────────────
with viz:
    st.header("📊 Quick Viz")
    st.plotly_chart(
        px.bar(users_df, x="SignupChannel", y="AvgSessionTimeMin",
               color="SignupChannel", title="Avg Session Time by Channel"),
        use_container_width=True)

# 2 ── TASTEDNA (Scaler + GridSearch) ───────────────────────────
with taste_tab:
    st.header("🔮 TasteDNA")
    mode = st.radio("Panel",["Classification","Clustering"],horizontal=True)

    X = users_df[core+behav]
    if mode=="Classification":
        algo = st.selectbox("Classifier",["KNN","DecisionTree","RandomForest","GradientBoost"])
        pipe_map = {
            "KNN": Pipeline([('sc',StandardScaler()),('clf',KNeighborsClassifier())]),
            "DecisionTree": Pipeline([('sc',StandardScaler()),
                                      ('clf',DecisionTreeClassifier(random_state=42))]),
            "RandomForest": Pipeline([('sc',StandardScaler()),
                                      ('clf',RandomForestClassifier(random_state=42))]),
            "GradientBoost": Pipeline([('sc',StandardScaler()),
                                       ('clf',GradientBoostingClassifier(random_state=42))])
        }
        gs_param = {"KNN": {'clf__n_neighbors':[3,5,7]},
                    "DecisionTree": {'clf__max_depth':[None,5,10]},
                    "RandomForest": {'clf__n_estimators':[100,200]},
                    "GradientBoost": {'clf__n_estimators':[100,200],
                                      'clf__learning_rate':[0.05,0.1]}}
        y = users_df["SubscribeIntent"]
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,stratify=y,test_size=0.25,random_state=42)
        grid = GridSearchCV(pipe_map[algo], gs_param[algo],
                            scoring="f1", cv=3, n_jobs=-1).fit(X_tr,y_tr)
        best = grid.best_estimator_
        y_pred = best.predict(X_te)
        st.write(f"Best params: {grid.best_params_}")
        for lbl,f in [("Precision",precision_score),("Recall",recall_score),
                      ("Accuracy",accuracy_score),("F1",f1_score)]:
            st.metric(lbl,f"{f(y_te,y_pred):.2f}")
        sns.heatmap(confusion_matrix(y_te,y_pred),annot=True,fmt="d",cmap="Blues")
        st.pyplot(plt.gcf()); plt.clf()
    else:
        k = st.slider("k clusters",2,10,4)
        X_scaled = StandardScaler().fit_transform(X)
        km = KMeans(k, random_state=42).fit(X_scaled)
        users_df["Cluster"]=km.labels_
        inert = [KMeans(i,random_state=42).fit(X_scaled).inertia_ for i in range(2,11)]
        plt.plot(range(2,11), inert, marker="o"); plt.title("Elbow Curve")
        st.pyplot(plt.gcf()); plt.clf()
        st.metric("Silhouette",f"{silhouette_score(X_scaled,km.labels_):.3f}")
        st.dataframe(users_df.groupby("Cluster")[behav].mean().round(2))

# 3 ── FORECASTING (hyper-tuned) ───────────────────────────────
with forecast_tab:
    st.header("📈 Forecasting")
    pred = st.selectbox("Predictor", weekly_predictors)
    reg_name = st.selectbox("Regressor",["Linear","Ridge","Lasso","DecisionTree"])
    if reg_name=="Linear":
        reg = LinearRegression()
        param = {}
    elif reg_name=="Ridge":
        reg = Ridge()
        param = {"alpha":[0.1,1.0,10]}
    elif reg_name=="Lasso":
        reg = Lasso(max_iter=5000)
        param = {"alpha":[0.001,0.01,0.1]}
    else:
        reg = DecisionTreeRegressor(random_state=42)
        param = {"max_depth":[3,5,7]}
    X = np.arange(len(trends_df)).reshape(-1,1)
    y = trends_df[pred].values
    cut = int(0.8*len(X))
    gs = GridSearchCV(reg, param, cv=3, scoring="neg_mean_squared_error").fit(X[:cut],y[:cut])
    best = gs.best_estimator_
    y_pred = best.predict(X[cut:])
    st.write(f"Best params: {gs.best_params_}")
    st.metric("R²",f"{r2_score(y[cut:],y_pred):.3f}")
    st.metric("RMSE",f"{np.sqrt(mean_squared_error(y[cut:],y_pred)):.2f}")
    fig = px.scatter(x=y[cut:], y=y_pred, labels={'x':'Actual','y':'Predicted'},
                     title="Parity Plot")
    fig.add_shape(type='line', x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
    st.plotly_chart(fig,use_container_width=True)

# 4 ── MICRO-BATCH  (unchanged display tweak) ──────────────────
with rules_tab:
    st.header("🧩 Micro-Batch")
    sup = st.slider("Support",0.01,0.2,0.03,0.01)
    conf= st.slider("Confidence",0.1,1.0,0.4,0.05)
    basket = pd.concat([
        users_df["FlavourFamilies"].str.get_dummies(sep=","),
        pd.get_dummies(users_df["PrimaryFlavourNote"],prefix="Note"),
        pd.get_dummies(users_df["Cluster"],prefix="C")],axis=1).astype(bool)
    rules = association_rules(apriori(basket,min_support=sup,use_colnames=True),
                              metric="confidence",min_threshold=conf)
    if rules.empty:
        st.info("No rules under thresholds.")
    else:
        st.dataframe(rules.nlargest(10,"confidence")[["antecedents","consequents","confidence","lift"]])

# 5 ── COMPLIANCE (trendline guard) ────────────────────────────
with st.tabs(["Compliance"])[0]:
    st.header("🛡️ Compliance & Nicotine Control")
    ctry = st.selectbox("Country", sorted(users_df["CountryRegCode"].unique()))
    sub  = users_df[users_df["CountryRegCode"]==ctry]
    pct  = sub["OverLimitWarning"].mean()*100
    st.metric("% Over Limit", f"{pct:.1f}%")
    if sm:  # only if statsmodels installed
        fig = px.scatter(sub, x="PodsPerWeek", y="NicotineToleranceLevel",
                         trendline="ols", title="Nicotine vs Usage")
    else:
        fig = px.scatter(sub, x="PodsPerWeek", y="NicotineToleranceLevel",
                         title="Nicotine vs Usage (install statsmodels for OLS line)")
    st.plotly_chart(fig,use_container_width=True)
