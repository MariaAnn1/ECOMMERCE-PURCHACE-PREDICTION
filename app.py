
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# ShopSense - E-Commerce Purchase Prediction
# PBADT504 | Big Data Analytics
# ============================================================

st.set_page_config(
    page_title="ShopSense | Purchase Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "balanced_purchase_prediction_model.pkl"
DATASET_PATH = "online_shoppers_intention.csv"

FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "June",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

VISITOR_TYPES = [
    "Returning_Visitor",
    "New_Visitor",
    "Other",
]


# -------------------- Styling --------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f5f6ff 0%, #ffffff 45%, #f7f9fc 100%);
}
.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(120deg, #20245c, #5156d9, #887cf6);
    color: white;
    margin-bottom: 20px;
}
.hero h1 {
    color: white;
    font-size: 2.5rem;
    margin-bottom: 8px;
}
.hero p {
    color: #f0f1ff;
    font-size: 1.05rem;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5e7f5;
    padding: 16px;
    border-radius: 15px;
    box-shadow: 0 3px 12px rgba(30, 35, 80, 0.04);
}
div[data-testid="stSidebar"] {
    background: #f0f1ff;
}
div.stButton > button,
div.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- Model loading --------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"

    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:
        return None, f"Could not load model: {exc}"


model, model_error = load_model()


def get_purchase_probability(model, row):
    """Extract the probability associated with the purchase class."""
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The saved model does not support predict_proba()."
        )

    probabilities = model.predict_proba(row)[0]
    classes = getattr(model, "classes_", None)

    if classes is not None:
        for index, label in enumerate(classes):
            if str(label).lower() in {
                "true", "1", "yes", "purchase"
            }:
                return float(probabilities[index])

    # For a binary classifier, the second class is commonly positive.
    if len(probabilities) == 2:
        return float(probabilities[1])

    raise ValueError("Could not identify the purchase class.")


# -------------------- Dataset loading --------------------

def demo_data():
    """
    Generate clearly labelled illustrative data.
    This is NOT the original UCI dataset.
    """
    rng = np.random.default_rng(42)
    n = 300

    visitors = rng.choice(
        VISITOR_TYPES, size=n, p=[0.72, 0.22, 0.06]
    )
    page_values = rng.gamma(1.5, 8, n)
    chance = 1 / (
        1 + np.exp(
            -(-2.0 + 0.055 * page_values
              + 0.45 * (visitors == "New_Visitor"))
        )
    )
    revenue = rng.random(n) < chance

    return pd.DataFrame({
        "Administrative": rng.integers(0, 8, n),
        "Administrative_Duration": rng.gamma(1.2, 35, n),
        "Informational": rng.integers(0, 5, n),
        "Informational_Duration": rng.gamma(1.1, 25, n),
        "ProductRelated": rng.integers(1, 35, n),
        "ProductRelated_Duration": rng.gamma(2, 55, n),
        "BounceRates": rng.uniform(0, 0.2, n),
        "ExitRates": rng.uniform(0.01, 0.25, n),
        "PageValues": page_values,
        "SpecialDay": rng.choice(
            [0, 0.2, 0.4, 0.6, 0.8, 1.0], n
        ),
        "Month": rng.choice(MONTHS, n),
        "OperatingSystems": rng.integers(1, 9, n),
        "Browser": rng.integers(1, 14, n),
        "Region": rng.integers(1, 10, n),
        "TrafficType": rng.integers(1, 21, n),
        "VisitorType": visitors,
        "Weekend": rng.choice([True, False], n),
        "Revenue": revenue,
    })


def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "Uploaded CSV"

    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH), DATASET_PATH

    return demo_data(), "Illustrative demo data (not original UCI data)"


def normalise_revenue(df):
    """Normalise common representations of the Revenue target."""
    df = df.copy()

    if "Revenue" not in df.columns:
        df["Revenue"] = False
    else:
        values = df["Revenue"]

        if pd.api.types.is_bool_dtype(values):
            df["Revenue"] = values.fillna(False)
        elif pd.api.types.is_numeric_dtype(values):
            df["Revenue"] = values.fillna(0).astype(float) != 0
        else:
            df["Revenue"] = (
                values.astype(str).str.strip().str.lower()
                .isin(["true", "1", "yes"])
            )

    df["PurchaseStatus"] = np.where(
        df["Revenue"], "Purchase", "No purchase"
    )
    return df


# -------------------- Sidebar --------------------

st.sidebar.markdown("# 🛍️ ShopSense")
st.sidebar.caption("E-commerce purchase intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Live Prediction",
        "Customer Insights",
        "Model Lab",
        "Data Explorer",
        "Project Guide",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### System status")

if model is not None:
    st.sidebar.success("Prediction model loaded")
else:
    st.sidebar.warning("Prediction model not connected")
    st.sidebar.caption(
        f"Place `{MODEL_PATH}` beside app.py to enable predictions."
    )

with st.sidebar.expander("Dataset settings"):
    uploaded_csv = st.file_uploader(
        "Upload your session dataset",
        type=["csv"],
        help="Expected to contain the project session features.",
    )
    st.caption(
        f"Without an upload, the app looks for `{DATASET_PATH}`."
    )

data, data_source = load_data(uploaded_csv)
data = normalise_revenue(data)

if "VisitorType" not in data.columns:
    data["VisitorType"] = "Unknown"

if "PageValues" not in data.columns:
    data["PageValues"] = 0.0


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":
    hero(
        "ShopSense",
        "Understand customer sessions. Explore behaviour. "
        "Estimate purchase likelihood.",
    )

    st.write(
        "An interactive dashboard for the E-Commerce Purchase "
        "Prediction Using Big Data project."
    )

    if "Illustrative demo data" in data_source:
        st.info(
            "Illustrative data is being used for dashboard charts. "
            "Upload the original UCI CSV for dataset-specific insights."
        )

    total = len(data)
    purchases = int(data["Revenue"].sum())
    purchase_rate = data["Revenue"].mean() * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions in view", f"{total:,}")
    c2.metric("Purchase sessions", f"{purchases:,}")
    c3.metric("Purchase rate", f"{purchase_rate:.2f}%")
    c4.metric(
        "Model status",
        "Ready" if model is not None else "Not connected",
    )

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Purchase outcome distribution")

        counts = (
            data["PurchaseStatus"]
            .value_counts()
            .reindex(["Purchase", "No purchase"], fill_value=0)
            .rename_axis("Outcome")
            .reset_index(name="Sessions")
        )

        fig = px.pie(
            counts,
            names="Outcome",
            values="Sessions",
            hole=0.58,
            color="Outcome",
            color_discrete_map={
                "Purchase": "#5b5ce2",
                "No purchase": "#dce0f3",
            },
        )
        fig.update_layout(margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Platform features")

        st.markdown("""
        **Explore customer behaviour**
        
        Filter visitor types and compare session characteristics.

        **Predict purchase likelihood**
        
        Enter session features and inspect the trained model's output.

        **Compare model performance**
        
        Review accuracy, precision, recall, F1-score and threshold results.

        **Explore the Big Data workflow**
        
        Learn how Spark SQL, Spark MLlib and Parquet fit into the project.
        """)

        st.caption(
            "Predictions are estimates and are not guarantees of customer behaviour."
        )

    st.subheader("Start exploring")

    a, b, c = st.columns(3)

    with a:
        st.markdown("### 📊 Customer Insights")
        st.write("Filter data and compare customer behaviour.")
        if st.button("Open Customer Insights", use_container_width=True):
            st.session_state["next_page"] = "Customer Insights"
            st.rerun()

    with b:
        st.markdown("### 🎯 Live Prediction")
        st.write("Enter a sample session to estimate purchase likelihood.")
        if st.button("Open Prediction", use_container_width=True):
            st.session_state["next_page"] = "Live Prediction"
            st.rerun()

    with c:
        st.markdown("### 🧪 Model Lab")
        st.write("Compare the reported model evaluation results.")
        if st.button("Open Model Lab", use_container_width=True):
            st.session_state["next_page"] = "Model Lab"
            st.rerun()

    st.subheader("Project snapshot")

    p1, p2, p3 = st.columns(3)
    p1.metric("Reported dataset size", "12,330 sessions")
    p2.metric("Reported purchases", "1,908")
    p3.metric("Reported purchase rate", "15.47%")

    st.caption(
        "These project figures are reported in the presentation. "
        "They may differ from the currently uploaded dataset."
    )


# ============================================================
# PAGE 2: LIVE PREDICTION
# ============================================================

elif page == "Live Prediction":
    hero(
        "Live Purchase Prediction",
        "Enter session behaviour and explore the estimated outcome.",
    )

    if model is None:
        st.error(
            f"{model_error} Add your saved trained pipeline before "
            "using this page for real predictions."
        )

    st.caption(
        "The model must have been trained with the same 17 feature names "
        "and compatible category values."
    )

    with st.form("prediction_form"):
        st.subheader("1. Page activity")

        a, b, c = st.columns(3)

        administrative = a.number_input(
            "Administrative pages", 0, 100, 1
        )
        admin_duration = a.number_input(
            "Administrative duration (seconds)", 0.0, value=20.0
        )

        informational = b.number_input(
            "Informational pages", 0, 100, 0
        )
        info_duration = b.number_input(
            "Informational duration (seconds)", 0.0, value=0.0
        )

        product_related = c.number_input(
            "Product-related pages", 0, 500, 8
        )
        product_duration = c.number_input(
            "Product-related duration (seconds)", 0.0, value=180.0
        )

        st.subheader("2. Engagement signals")

        d, e, f = st.columns(3)

        bounce = d.number_input(
            "Bounce rate", 0.0, 1.0, 0.02, step=0.01, format="%.3f"
        )
        exit_rate = e.number_input(
            "Exit rate", 0.0, 1.0, 0.04, step=0.01, format="%.3f"
        )
        page_values = f.number_input(
            "PageValues", 0.0, value=5.0, step=1.0
        )

        special_day = d.select_slider(
            "SpecialDay proximity",
            options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            value=0.0,
        )

        st.subheader("3. Visitor and environment")

        g, h, i = st.columns(3)

        month = g.selectbox("Month", MONTHS, index=8)
        visitor_type = h.selectbox(
            "Visitor type", VISITOR_TYPES, index=0
        )
        weekend = i.toggle("Weekend visit", value=False)

        operating_system = g.selectbox(
            "Operating system code", list(range(1, 9)), index=1
        )
        browser = h.selectbox(
            "Browser code", list(range(1, 14)), index=1
        )
        region = i.selectbox(
            "Region code", list(range(1, 10)), index=0
        )
        traffic_type = g.selectbox(
            "Traffic type code", list(range(1, 21)), index=1
        )

        threshold = st.slider(
            "Purchase decision threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help=(
                "A lower threshold flags more sessions as potential "
                "purchases, usually with more false positives."
            ),
        )

        submitted = st.form_submit_button(
            "🔍 Predict purchase likelihood",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        values = {
            "Administrative": administrative,
            "Administrative_Duration": admin_duration,
            "Informational": informational,
            "Informational_Duration": info_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_duration,
            "BounceRates": bounce,
            "ExitRates": exit_rate,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_system,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": bool(weekend),
        }

        row = pd.DataFrame([values], columns=FEATURES)

        if model is not None:
            try:
                probability = get_purchase_probability(model, row)
                prediction = (
                    "Purchase likely"
                    if probability >= threshold
                    else "Purchase unlikely"
                )

                st.divider()
                st.subheader("Prediction result")

                r1, r2, r3 = st.columns(3)
                r1.metric(
                    "Estimated purchase probability",
                    f"{probability:.1%}",
                )
                r2.metric("Decision threshold", f"{threshold:.2f}")
                r3.metric("Model classification", prediction)

                st.progress(
                    min(max(probability, 0.0), 1.0),
                    text=f"Purchase probability: {probability:.1%}",
                )

                if probability >= threshold:
                    st.success(
                        "The model classifies this session as a potential purchase."
                    )
                else:
                    st.info(
                        "The model classifies this session as a potential non-purchase."
                    )

                st.caption(
                    "This is a model estimate, not a guarantee. "
                    "Probability quality depends on the training data and model calibration."
                )

                with st.expander("View submitted feature values"):
                    st.dataframe(
                        row, use_container_width=True, hide_index=True
                    )

                st.download_button(
                    "Download this session as CSV",
                    row.to_csv(index=False).encode("utf-8"),
                    "sample_session.csv",
                    "text/csv",
                )

            except Exception as exc:
                st.error(
                    "The model could not process the inputs. Check the "
                    "saved pipeline, feature names and categorical values."
                )
                st.exception(exc)
        else:
            st.warning(
                "No real prediction was made because the trained model is missing."
            )


# ============================================================
# PAGE 3: CUSTOMER INSIGHTS
# ============================================================

elif page == "Customer Insights":
    hero(
        "Customer Insights",
        "Filter sessions and discover patterns in purchase behaviour.",
    )

    if "Illustrative demo data" in data_source:
        st.info(
            "Charts currently use illustrative demo data. Upload the "
            "original UCI CSV for actual dataset analysis."
        )

    visitor_options = sorted(
        data["VisitorType"].dropna().astype(str).unique().tolist()
    )

    f1, f2 = st.columns(2)

    selected_visitors = f1.multiselect(
        "Filter visitor types",
        visitor_options,
        default=visitor_options,
    )

    outcome_filter = f2.selectbox(
        "Filter session outcome",
        ["All", "Purchase only", "No purchase only"],
    )

    filtered = data.copy()

    if selected_visitors:
        filtered = filtered[
            filtered["VisitorType"].astype(str).isin(selected_visitors)
        ]
    else:
        filtered = filtered.iloc[0:0]

    if outcome_filter == "Purchase only":
        filtered = filtered[filtered["Revenue"]]
    elif outcome_filter == "No purchase only":
        filtered = filtered[~filtered["Revenue"]]

    m1, m2, m3 = st.columns(3)
    m1.metric("Filtered sessions", f"{len(filtered):,}")
    m2.metric("Purchases", f"{int(filtered['Revenue'].sum()):,}")
    m3.metric(
        "Purchase rate",
        f"{filtered['Revenue'].mean():.1%}" if len(filtered) else "—",
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Purchase rate by visitor type")

        if len(filtered):
            summary = (
                filtered.groupby("VisitorType", dropna=False)["Revenue"]
                .agg(Sessions="count", PurchaseRate="mean")
                .reset_index()
            )
            summary["PurchaseRate"] *= 100

            fig = px.bar(
                summary,
                x="VisitorType",
                y="PurchaseRate",
                color="VisitorType",
                text=summary["PurchaseRate"].map(lambda x: f"{x:.1f}%"),
                labels={
                    "VisitorType": "Visitor type",
                    "PurchaseRate": "Purchase rate (%)",
                },
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sessions match the selected filters.")

    with right:
        st.subheader("PageValues by outcome")

        if len(filtered):
            fig = px.box(
                filtered,
                x="PurchaseStatus",
                y="PageValues",
                color="PurchaseStatus",
                color_discrete_map={
                    "Purchase": "#5b5ce2",
                    "No purchase": "#b8c1df",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data to display.")

    st.subheader("Explore feature relationships")

    choices = [
        col for col in [
            "ProductRelated",
            "ProductRelated_Duration",
            "Administrative",
            "PageValues",
            "BounceRates",
            "ExitRates",
        ]
        if col in filtered.columns
    ]

    if len(choices) >= 2 and len(filtered):
        x_axis = st.selectbox("Horizontal axis", choices, index=0)
        y_axis = st.selectbox("Vertical axis", choices, index=1)

        sample = (
            filtered.sample(1500, random_state=42)
            if len(filtered) > 1500 else filtered
        )

        fig = px.scatter(
            sample,
            x=x_axis,
            y=y_axis,
            color="PurchaseStatus",
            color_discrete_map={
                "Purchase": "#5b5ce2",
                "No purchase": "#b8c1df",
            },
            hover_data=[
                col for col in ["VisitorType", "Month", "PageValues"]
                if col in sample.columns
            ],
            opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View filtered records"):
        st.dataframe(
            filtered.head(1000),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download filtered sessions",
            filtered.to_csv(index=False).encode("utf-8"),
            "filtered_sessions.csv",
            "text/csv",
            disabled=filtered.empty,
        )


# ============================================================
# PAGE 4: MODEL LAB
# ============================================================

elif page == "Model Lab":
    hero(
        "Model Lab",
        "Compare reported evaluation metrics and threshold trade-offs.",
    )

    st.info(
        "These are the reported project results from your presentation. "
        "They are not recalculated from the uploaded CSV."
    )

    metrics = pd.DataFrame([
        {
            "Model": "Original Logistic Regression",
            "Accuracy": 88.94,
            "Precision": 77.18,
            "Recall": 41.62,
            "F1-score": 54.08,
        },
        {
            "Model": "Balanced Logistic Regression",
            "Accuracy": 85.13,
            "Precision": 51.64,
            "Recall": 78.53,
            "F1-score": 62.31,
        },
        {
            "Model": "Spark MLlib (threshold 0.50)",
            "Accuracy": 88.09,
            "Precision": 76.60,
            "Recall": 37.11,
            "F1-score": 50.00,
        },
    ])

    chosen_metrics = st.multiselect(
        "Choose metrics to compare",
        ["Accuracy", "Precision", "Recall", "F1-score"],
        default=["Accuracy", "Precision", "Recall", "F1-score"],
    )

    if chosen_metrics:
        chart_data = metrics.melt(
            id_vars="Model",
            value_vars=chosen_metrics,
            var_name="Metric",
            value_name="Score (%)",
        )

        fig = px.bar(
            chart_data,
            x="Model",
            y="Score (%)",
            color="Metric",
            barmode="group",
            text_auto=".1f",
            range_y=[0, 100],
        )
        fig.update_layout(xaxis_title="", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.subheader("Spark MLlib threshold experiment")

    thresholds = pd.DataFrame([
        {
            "Threshold": 0.30,
            "Accuracy": 88.79,
            "Precision": 69.44,
            "Recall": 53.87,
            "F1-score": 60.67,
        },
        {
            "Threshold": 0.35,
            "Accuracy": 88.67,
            "Precision": 71.11,
            "Recall": 49.48,
            "F1-score": 58.36,
        },
        {
            "Threshold": 0.40,
            "Accuracy": 88.59,
            "Precision": 74.14,
            "Recall": 44.33,
            "F1-score": 55.48,
        },
        {
            "Threshold": 0.45,
            "Accuracy": 88.46,
            "Precision": 76.59,
            "Recall": 40.46,
            "F1-score": 52.95,
        },
        {
            "Threshold": 0.50,
            "Accuracy": 88.09,
            "Precision": 76.60,
            "Recall": 37.11,
            "F1-score": 50.00,
        },
    ])

    metric_choice = st.selectbox(
        "Choose threshold metric",
        ["F1-score", "Recall", "Precision", "Accuracy"],
    )

    fig = px.line(
        thresholds,
        x="Threshold",
        y=metric_choice,
        markers=True,
        text=metric_choice,
        range_y=[0, 100],
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="top center",
    )
    fig.update_layout(
        yaxis_title=f"{metric_choice} (%)",
        xaxis_title="Decision threshold",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(thresholds, use_container_width=True, hide_index=True)

    st.subheader("How to interpret these results")

    st.markdown("""
    - **Accuracy:** the share of all predictions that were correct.
    - **Precision:** how many predicted purchases were actual purchases.
    - **Recall:** how many actual purchases the model detected.
    - **F1-score:** a balance between precision and recall.
    """)

    st.success(
        "In the reported Spark MLlib threshold experiment, 0.30 had "
        "the highest F1-score among the five tested thresholds: 60.67%."
    )

    st.warning(
        "For a rigorous final estimate, choose the threshold using validation "
        "data and evaluate once on an untouched test set."
    )


# ============================================================
# PAGE 5: DATA EXPLORER
# ============================================================

elif page == "Data Explorer":
    hero(
        "Data Explorer",
        "Inspect records, data quality and feature distributions.",
    )

    st.caption(f"Current data source: {data_source}")

    if "Illustrative demo data" in data_source:
        st.warning(
            "This is illustrative data, not the original UCI dataset. "
            "Upload your original CSV to inspect the real project data."
        )

    tab1, tab2, tab3 = st.tabs([
        "Data Preview",
        "Data Quality",
        "Feature Distributions",
    ])

    with tab1:
        rows = st.slider(
            "Number of rows to preview",
            min_value=5,
            max_value=100,
            value=15,
            step=5,
        )
        st.dataframe(
            data.head(rows),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download current dataset",
            data.to_csv(index=False).encode("utf-8"),
            "session_data.csv",
            "text/csv",
        )

    with tab2:
        quality = pd.DataFrame({
            "Column": data.columns,
            "Data type": [str(data[col].dtype) for col in data.columns],
            "Missing values": [
                int(data[col].isna().sum()) for col in data.columns
            ],
            "Missing (%)": [
                round(data[col].isna().mean() * 100, 2)
                for col in data.columns
            ],
            "Unique values": [
                int(data[col].nunique(dropna=True))
                for col in data.columns
            ],
        })

        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(data):,}")
        b.metric("Columns", f"{len(data.columns):,}")
        c.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")

        st.dataframe(
            quality,
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        numeric_columns = [
            col for col in data.select_dtypes(include=np.number).columns
            if col != "Revenue"
        ]

        if numeric_columns:
            feature = st.selectbox(
                "Select a numeric feature", numeric_columns
            )

            fig = px.histogram(
                data,
                x=feature,
                color="PurchaseStatus",
                marginal="box",
                barmode="overlay",
                opacity=0.7,
                color_discrete_map={
                    "Purchase": "#5b5ce2",
                    "No purchase": "#b8c1df",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Summary statistics by outcome")
            st.dataframe(
                data.groupby("PurchaseStatus")[feature]
                .describe()
                .round(3),
                use_container_width=True,
            )
        else:
            st.info("No numeric features are available.")


# ============================================================
# PAGE 6: PROJECT GUIDE
# ============================================================

elif page == "Project Guide":
    hero(
        "Project Guide",
        "Understand the dataset, Big Data workflow and project findings.",
    )

    st.subheader("Project overview")

    st.write(
        "The project analyses customer session data to predict whether "
        "an e-commerce visit will result in a purchase. The target column "
        "is Revenue, where True represents a purchase and False represents "
        "no purchase."
    )

    st.subheader("Project objectives")

    st.markdown("""
    - Analyse customer purchasing behaviour.
    - Build a Purchase / No Purchase classifier.
    - Evaluate model performance using multiple metrics.
    - Demonstrate data analysis with Spark SQL and Spark MLlib.
    """)

    st.subheader("Big Data workflow")

    workflow = [
        ("1. Dataset input", "Load and inspect the CSV session records."),
        ("2. Preprocessing", "Check data quality and prepare features."),
        ("3. Spark DataFrame", "Represent and process data with Apache Spark."),
        ("4. Spark SQL", "Analyse purchase counts and visitor behaviour."),
        ("5. Machine learning", "Train and evaluate purchase classifiers."),
        ("6. Evaluation", "Compare metrics and decision thresholds."),
        ("7. Storage", "Use Parquet and save the trained model."),
    ]

    for title, description in workflow:
        with st.expander(title):
            st.write(description)

    st.subheader("Reported dataset findings")

    a, b, c = st.columns(3)
    a.metric("Total sessions", "12,330")
    b.metric("Purchases", "1,908")
    c.metric("Purchase rate", "15.47%")

    st.markdown("""
    **Insights reported in the project presentation**

    - New visitors had the highest purchase rate: **24.91%**.
    - Returning visitors contributed the most purchases: **1,470**.
    - Average PageValues was **27.26** for purchase sessions and **1.98**
      for non-purchase sessions.
    """)

    st.subheader("Technology stack")

    st.write(
        "Python • Pandas • scikit-learn • Apache Spark • Spark SQL • "
        "Spark MLlib • Parquet • Streamlit • Plotly"
    )

    st.subheader("Implementation status")

    st.success("""
    Reported as completed in the presentation:
    dataset inspection, preprocessing pipeline, baseline and balanced
    models, Spark SQL analysis, Spark MLlib training and evaluation,
    threshold comparison, Parquet storage and model saving.
    """)

    st.warning("""
    HDFS and Apache Hive integration were marked as not yet confirmed
    in the project presentation. Only mark these as completed after
    verifying the services and successful queries.
    """)

    st.subheader("Dataset source")

    st.markdown(
        "[UCI Online Shoppers Purchasing Intention Dataset]"
        "(https://archive.ics.uci.edu/dataset/468/"
        "online+shoppers+purchasing+intention+dataset)"
    )

    st.divider()

    st.markdown("### Future scope")

    st.markdown("""
    - Complete and verify HDFS and Hive integration.
    - Test the pipeline on larger datasets and distributed nodes.
    - Compare additional classification algorithms.
    - Tune decision thresholds using validation data.
    - Extend the prototype into a real-time analytics system.
    """)

    st.caption(
        "PBADT504 • Big Data Analytics | E-Commerce Purchase Prediction"
    )
