import streamlit as st
import pandas as pd
import numpy as np
import shap
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="TITANIC AI PLATFORM", layout="wide")

# =====================================================
# CINEMATIC DEEP SEA + GLACIER EFFECT
# =====================================================
st.markdown("""
<style>

/* Deep moving ocean */
.stApp {
    background: radial-gradient(circle at 30% 20%, #012a4a 0%, #001d3d 50%),
                linear-gradient(-45deg, #000814, #001d3d, #003566, #001845);
    background-size: 400% 400%;
    animation: oceanFlow 40s ease infinite;
    color: white;
}

/* Slow darkening effect */
@keyframes oceanFlow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Vignette */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at center, transparent 40%, rgba(0,0,0,0.6) 100%);
    pointer-events: none;
}

/* Floating iceberg silhouette */
.stApp::after {
    content: "";
    position: fixed;
    bottom: 5%;
    right: 5%;
    width: 300px;
    height: 200px;
    background: linear-gradient(to top, #90e0ef, #caf0f8);
    clip-path: polygon(50% 0%, 90% 70%, 10% 70%);
    opacity: 0.15;
    animation: floatIce 8s ease-in-out infinite;
}

@keyframes floatIce {
    0% {transform: translateY(0px);}
    50% {transform: translateY(15px);}
    100% {transform: translateY(0px);}
}

/* Glass panels */
.block-container {
    background: rgba(0, 20, 40, 0.65);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 2rem;
}

/* Ambient glow metrics */
div[data-testid="metric-container"] {
    background: rgba(0,100,150,0.2);
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0 0 15px rgba(0,200,255,0.3);
}

/* Floating title */
h1 {
    animation: floatTitle 6s ease-in-out infinite;
}
@keyframes floatTitle {
    0% {transform: translateY(0px);}
    50% {transform: translateY(6px);}
    100% {transform: translateY(0px);}
}
</style>
""", unsafe_allow_html=True)

st.title("🚢 TITANIC AI PLATFORM")
st.caption("Immersive Deep Sea Survival Intelligence")

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("titanic.csv")

# =====================================================
# NAVIGATION
# =====================================================
page = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "🤖 Prediction Model",
    "📈 Explanation",
    "💬 Chatbot"
])

# =====================================================
# DASHBOARD
# =====================================================
if page == "📊 Dashboard":

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Passengers", len(df))
    col2.metric("Survival Rate", f"{round(df['Survived'].mean()*100, 2)}%")
    col3.metric("Average Age", round(df['Age'].mean(), 1))
    col4.metric("Average Fare", round(df['Fare'].mean(), 2))

    st.markdown("---")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Survival by Gender")
        st.bar_chart(df.groupby("Sex")["Survived"].mean())

    with colB:
        st.subheader("Passenger Class Distribution")
        st.bar_chart(df["Pclass"].value_counts())

# =====================================================
# MODEL
# =====================================================
elif page == "🤖 Prediction Model":

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[features]
    y = df["Survived"]

    numeric = ["Age", "Fare", "SibSp", "Parch"]
    categorical = ["Sex", "Embarked", "Pclass"]

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            class_weight="balanced",
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with st.spinner("Training deep sea survival engine..."):
        model.fit(X_train, y_train)
        time.sleep(1)

    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Model Accuracy: {round(acc*100, 2)}%")
    st.progress(int(acc*100))

    st.session_state.model = model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

    st.markdown("### Predict Survival")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 1, 80, 30)
    fare = st.slider("Fare", 0, 500, 50)
    sibsp = st.slider("Siblings", 0, 5, 0)
    parch = st.slider("Parents/Children", 0, 5, 0)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])

    input_df = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    if st.button("Predict"):
        prob = model.predict_proba(input_df)[0][1]
        st.progress(int(prob*100))
        if prob > 0.5:
            st.success(f"Likely Survived — {round(prob*100, 2)}%")
        else:
            st.error(f"Likely Did Not Survive — {round(prob*100, 2)}%")

# =====================================================
# EXPLANATION
# =====================================================
elif page == "📈 Explanation":

    if "model" not in st.session_state:
        st.warning("Train the model first.")
        st.stop()

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
    st.pyplot(fig1)

    st.subheader("ROC Curve")
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {round(roc_auc, 2)}")
    ax2.plot([0, 1], [0, 1], '--')
    ax2.legend()
    st.pyplot(fig2)

# =====================================================
# CHATBOT
# =====================================================
elif page == "💬 Chatbot":

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask about Titanic data...")

    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        q = user_input.lower()

        if "survival rate" in q:
            response = f"Survival rate is {round(df['Survived'].mean()*100, 2)}%."
        elif "male" in q:
            percent = df["Sex"].value_counts(normalize=True)["male"]*100
            response = f"{round(percent, 2)}% were male."
        elif "fare" in q:
            response = f"Average fare was {round(df['Fare'].mean(), 2)}."
        else:
            response = "Ask about survival rate, gender %, or fare."

        placeholder = st.chat_message("assistant").empty()
        typed = ""
        for char in response:
            typed += char
            placeholder.markdown(typed)
            time.sleep(0.01)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response})
