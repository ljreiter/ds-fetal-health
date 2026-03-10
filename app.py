import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import shap
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fetal Health Analytics",
    page_icon="🫀",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}
.metric-card {
    background: #f8f4ef;
    border-left: 4px solid #c0392b;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.caption-box {
    background: #f0f4f8;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.9rem;
    color: #444;
    margin-top: 0.25rem;
}
.prediction-box {
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
}
.pred-normal   { background:#d4edda; color:#155724; border: 2px solid #28a745; }
.pred-suspect  { background:#fff3cd; color:#856404; border: 2px solid #ffc107; }
.pred-pathological { background:#f8d7da; color:#721c24; border: 2px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Decision Tree":       joblib.load("models/decision_tree.pkl"),
        "Random Forest":       joblib.load("models/random_forest.pkl"),
        "XGBoost":             joblib.load("models/xgboost.pkl"),
    }
    scaler      = joblib.load("models/scaler.pkl")
    le_xgb      = joblib.load("models/xgb_label_encoder.pkl")

    try:
        from tensorflow.keras.models import load_model
        nn = load_model("models/neural_network.keras")
        le_nn = joblib.load("models/nn_label_encoder.pkl")
        models["Neural Network"] = nn
    except Exception:
        le_nn = None

    feature_names    = joblib.load("artifacts/feature_names.pkl")
    default_values   = joblib.load("artifacts/default_feature_values.pkl")
    results_table    = joblib.load("artifacts/results_table.pkl")
    best_params_table = joblib.load("artifacts/best_params_table.pkl")

    df = pd.read_csv("fetal_health.csv")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    return models, scaler, le_xgb, le_nn, feature_names, default_values, results_table, best_params_table, df

models, scaler, le_xgb, le_nn, feature_names, default_values, results_table, best_params_table, df = load_all()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary",
    "📊 Descriptive Analytics",
    "🏆 Model Performance",
    "🔍 Explainability & Prediction",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Summary
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🫀 Fetal Health Risk Classification")
    st.subheader("An Analytics Dashboard for Clinicians and Healthcare Decision-Makers")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### What Is This?")
        st.markdown("""
        This dashboard presents a machine learning analysis of **cardiotocogram (CTG) data** — 
        electronic fetal monitoring measurements taken during pregnancy. The goal is to automatically 
        classify fetal health into one of three categories:

        - 🟢 **Class 1 — Normal**: No signs of distress; routine monitoring recommended.
        - 🟡 **Class 2 — Suspect**: Borderline signs that warrant closer observation.
        - 🔴 **Class 3 — Pathological**: Clear signs of distress requiring immediate clinical attention.

        Early and accurate identification of at-risk fetuses can support timely intervention and 
        **reduce preventable child and maternal mortality** — a global health priority.
        """)

        st.markdown("### The Data")
        st.markdown("""
        The dataset contains **2,126 CTG records** with **21 clinical features** such as heart rate 
        variability, accelerations, decelerations, and uterine contractions. There are no missing values. 
        The classes are imbalanced: Class 1 accounts for roughly 78% of records, while Classes 2 and 3 
        are much smaller, making accurate detection of high-risk cases especially challenging.
        """)

        st.markdown("### What We Did")
        st.markdown("""
        We trained and compared **five machine learning models**: Logistic Regression (baseline), 
        Decision Tree, Random Forest, XGBoost, and a Neural Network. Each model was tuned using 
        5-fold cross-validation. We evaluated them using weighted F1-score and multiclass AUC-ROC 
        to account for class imbalance.
        """)

        st.markdown("### Key Findings")
        st.markdown("""
        - **XGBoost was the best-performing model**, achieving a weighted F1-score of **0.935** and 
          AUC-ROC of **0.982** on the held-out test set.
        - The most predictive features were **abnormal short-term variability**, **histogram mean**, 
          and **mean value of long-term variability** — all physiologically meaningful signals.
        - Even the simplest model (Logistic Regression) achieved 89% accuracy, confirming that 
          CTG data contains strong, learnable patterns.
        - An interactive prediction tool (Tab 4) lets clinicians enter real measurements and receive 
          an instant risk classification with probability estimates.
        """)

    with col2:
        st.markdown("### At a Glance")
        st.markdown('<div class="metric-card"><b>Dataset Size</b><br>2,126 records · 21 features</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><b>Best Model</b><br>XGBoost</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><b>Best F1-Score</b><br>0.935</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><b>Best AUC-ROC</b><br>0.982</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><b>Models Compared</b><br>5 models evaluated</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><b>Validation Strategy</b><br>5-Fold Stratified CV</div>', unsafe_allow_html=True)

        st.markdown("### So What?")
        st.markdown("""
        A model that flags high-risk cases with 93%+ accuracy could meaningfully support clinical 
        workflows — helping busy providers prioritize which patients need immediate follow-up. 
        This is not a replacement for clinical judgment, but a decision-support tool that surfaces 
        patterns in data that are easy to miss.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Descriptive Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📊 Descriptive Analytics")
    st.markdown("Exploratory visualizations of the fetal health dataset.")
    st.markdown("---")

    def show_plot(path, caption):
        if os.path.exists(path):
            st.image(path)
            st.markdown(f'<div class="caption-box">{caption}</div>', unsafe_allow_html=True)
        else:
            st.warning(f"Plot not found: {path}")

    st.subheader("Target Class Distribution")
    show_plot("plots/target_distribution.png",
        "The dataset is heavily imbalanced: Class 1 (Normal) contains the majority of records (~78%), "
        "while Classes 2 (Suspect) and 3 (Pathological) are much smaller. This imbalance means that "
        "accuracy alone is a misleading metric — a model that always predicts Class 1 would score high "
        "but completely miss the high-risk cases that matter most clinically.")

    st.markdown("---")
    st.subheader("Feature Distributions by Fetal Health Class")

    col1, col2 = st.columns(2)
    with col1:
        show_plot("plots/accelerations_boxplot.png",
            "Class 1 (Normal) fetuses show noticeably higher acceleration values than Classes 2 and 3, "
            "which cluster near zero. Accelerations in fetal heart rate are generally a reassuring sign, "
            "so this pattern aligns with clinical expectations and suggests accelerations may be a useful "
            "discriminating feature.")
        show_plot("plots/abnormal_stv_boxplot.png",
            "Abnormal short-term variability is lowest for Class 1 and progressively higher for Classes 2 "
            "and 3. Reduced short-term variability in fetal heart rate is a recognized clinical indicator "
            "of fetal compromise, so this feature's distribution strongly supports its predictive value in "
            "the models.")

    with col2:
        show_plot("plots/prolongued_decelerations_boxplot.png",
            "Class 3 (Pathological) cases show much higher prolonged deceleration values and greater spread "
            "than the other classes. Prolonged decelerations — sustained drops in fetal heart rate — are "
            "considered a serious warning sign in obstetric monitoring, and this plot confirms that they are "
            "strongly associated with the most severe fetal health outcomes in this dataset.")
        show_plot("plots/histogram_variance_boxplot.png",
            "Histogram variance is substantially higher and more spread out in Class 3 compared to Classes 1 "
            "and 2. This feature captures the variability in the distribution of fetal heart rate values over "
            "time. A wider spread may indicate irregular or erratic heart rate patterns, which can signal "
            "underlying fetal distress.")

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    show_plot("plots/correlation_heatmap.png",
        "The correlation heatmap reveals that several histogram-derived features are moderately correlated "
        "with each other, which is expected since they all describe the shape of the fetal heart rate "
        "distribution. Most other feature pairs show weaker correlations. For tree-based models like "
        "Random Forest and XGBoost, this collinearity is not a concern, but it is worth noting for "
        "the Logistic Regression baseline, which may be slightly affected by overlapping predictors.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("🏆 Model Performance")
    st.markdown("Comparison of all models trained on the fetal health dataset.")
    st.markdown("---")

    st.subheader("Model Comparison Table")
    st.dataframe(results_table.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weighted F1-Score by Model")
        show_plot("plots/model_comparison_f1.png",
            "XGBoost achieved the highest weighted F1-score (0.935), closely followed by Random Forest "
            "(0.927). Both ensemble methods substantially outperformed Logistic Regression (0.895), "
            "which served as the baseline. The Neural Network performed comparably to the Decision Tree "
            "but did not surpass the ensemble models on this structured tabular dataset.")

    with col2:
        st.subheader("Best Hyperparameters")
        st.dataframe(best_params_table, use_container_width=True)

    st.markdown("---")
    st.subheader("ROC Curves")
    col3, col4 = st.columns(2)

    with col3:
        show_plot("plots/random_forest_roc.png",
            "Random Forest ROC curves show strong separation for all three classes. Class 1 and Class 3 "
            "are classified with very high AUC scores, while Class 2 (Suspect) is slightly harder to "
            "distinguish, reflecting its position as a borderline category between Normal and Pathological.")

    with col4:
        show_plot("plots/xgboost_roc.png",
            "XGBoost achieves near-perfect ROC curves for Classes 1 and 3, and solid performance on "
            "Class 2. The AUC-ROC of 0.982 confirms that the model is highly effective at ranking and "
            "distinguishing all three fetal health classes, making it the most reliable model overall.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Explainability & Interactive Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("🔍 Explainability & Interactive Prediction")
    st.markdown("---")

    # ── SHAP plots ──────────────────────────────────────────────────────────
    st.subheader("SHAP Explainability (XGBoost — Class 3: Pathological)")

    col1, col2 = st.columns(2)
    with col1:
        show_plot("plots/shap_summary_class3.png",
            "The SHAP beeswarm plot shows that abnormal_short_term_variability, histogram_mean, and "
            "mean_value_of_long_term_variability are the top drivers of Class 3 predictions. Red points "
            "(high feature values) pushing right indicate those values increase the probability of a "
            "Pathological classification.")
    with col2:
        show_plot("plots/shap_bar_class3.png",
            "The SHAP bar chart ranks features by mean absolute SHAP value — a model-wide measure of "
            "overall importance for predicting Class 3. The top features align closely with known clinical "
            "indicators of fetal distress, suggesting the model has learned medically meaningful patterns.")

    st.markdown("---")
    show_plot("plots/shap_waterfall_class3.png",
        "The SHAP waterfall plot explains a single Class 3 case from the test set. It shows how each "
        "feature pushed the prediction up (toward Pathological) or down (away from it). The strongest "
        "positive contributor was prolongued_decelerations, followed by abnormal_short_term_variability. "
        "This local explanation helps build trust by showing the model's reasoning for one specific patient.")

    st.markdown("---")

    # ── Interactive Prediction ──────────────────────────────────────────────
    st.subheader("Interactive Prediction Tool")
    st.markdown("Adjust the sliders below to enter fetal monitoring measurements, then click **Predict**.")

    # Interactive features identified from SHAP analysis
    interactive_features = [
        "abnormal_short_term_variability",
        "histogram_mean",
        "mean_value_of_long_term_variability",
        "uterine_contractions",
        "percentage_of_time_with_abnormal_long_term_variability",
        "prolongued_decelerations",
        "histogram_variance",
    ]

    col_left, col_right = st.columns([1, 1])

    user_input = {}

    # Fill all non-interactive features with dataset means
    for feat in feature_names:
        user_input[feat] = default_values.get(feat, 0.0)

    with col_left:
        st.markdown("**Heart Rate Variability**")
        user_input["abnormal_short_term_variability"] = st.slider(
            "Abnormal Short-Term Variability (%)",
            min_value=0, max_value=100,
            value=int(default_values.get("abnormal_short_term_variability", 20)),
            help="Percentage of time with abnormal short-term variability"
        )
        user_input["mean_value_of_long_term_variability"] = st.slider(
            "Mean Long-Term Variability",
            min_value=0, max_value=50,
            value=int(default_values.get("mean_value_of_long_term_variability", 8)),
        )
        user_input["percentage_of_time_with_abnormal_long_term_variability"] = st.slider(
            "% Time with Abnormal Long-Term Variability",
            min_value=0, max_value=100,
            value=int(default_values.get("percentage_of_time_with_abnormal_long_term_variability", 5)),
        )

    with col_right:
        st.markdown("**CTG Signal Features**")
        user_input["histogram_mean"] = st.slider(
            "Histogram Mean",
            min_value=50, max_value=200,
            value=int(default_values.get("histogram_mean", 137)),
        )
        user_input["histogram_variance"] = st.slider(
            "Histogram Variance",
            min_value=0, max_value=300,
            value=int(default_values.get("histogram_variance", 18)),
        )
        user_input["uterine_contractions"] = st.slider(
            "Uterine Contractions",
            min_value=0.000, max_value=0.020,
            value=float(default_values.get("uterine_contractions", 0.004)),
            step=0.001, format="%.3f",
        )
        user_input["prolongued_decelerations"] = st.slider(
            "Prolonged Decelerations",
            min_value=0.000, max_value=0.010,
            value=float(default_values.get("prolongued_decelerations", 0.0)),
            step=0.001, format="%.3f",
        )

    # Model selector
    st.markdown("---")
    model_choice = st.selectbox(
        "Select Model for Prediction",
        options=["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Neural Network"],
    )

    if st.button("🔮 Predict Fetal Health", use_container_width=True):
        input_df = pd.DataFrame([user_input])[feature_names]

        label_map = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}
        css_map   = {1.0: "pred-normal", 2.0: "pred-suspect", 3.0: "pred-pathological"}

        try:
            if model_choice == "Neural Network":
                input_scaled = scaler.transform(input_df)
                probs = models["Neural Network"].predict(input_scaled)[0]
                pred_encoded = int(np.argmax(probs))
                pred_label_raw = le_nn.inverse_transform([pred_encoded])[0]
                prob_dict = {float(le_nn.classes_[i]): float(probs[i]) for i in range(len(probs))}
            elif model_choice == "Logistic Regression":
                input_scaled = scaler.transform(input_df)
                probs = models[model_choice].predict_proba(input_scaled)[0]
                pred_label_raw = models[model_choice].predict(input_scaled)[0]
                classes = models[model_choice].classes_
                prob_dict = {float(classes[i]): float(probs[i]) for i in range(len(classes))}
            else:
                probs = models[model_choice].predict_proba(input_df)[0]
                pred_label_raw = models[model_choice].predict(input_df)[0]
                if model_choice == "XGBoost":
                    pred_label_raw = le_xgb.inverse_transform([int(pred_label_raw)])[0]
                    classes = le_xgb.classes_
                else:
                    classes = models[model_choice].classes_
                prob_dict = {float(classes[i]): float(probs[i]) for i in range(len(probs))}

            pred_label_raw = float(pred_label_raw)
            label_str = label_map.get(pred_label_raw, str(pred_label_raw))
            css_class  = css_map.get(pred_label_raw, "pred-normal")

            st.markdown(f"""
            <div class="prediction-box {css_class}">
                Predicted Class: <b>{int(pred_label_raw)} — {label_str}</b>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Class Probabilities")
            prob_df = pd.DataFrame({
                "Class": [f"Class {int(k)} — {label_map[k]}" for k in sorted(prob_dict)],
                "Probability": [prob_dict[k] for k in sorted(prob_dict)],
            })
            st.dataframe(prob_df.style.bar(subset=["Probability"], color="#5b9bd5"), use_container_width=True)

            # SHAP waterfall for custom input (XGBoost only)
            if model_choice == "XGBoost":
                st.markdown("#### SHAP Explanation for This Prediction")
                with st.spinner("Calculating SHAP values..."):
                    try:
                        background = df.drop("fetal_health", axis=1).sample(200, random_state=42)
                        explainer  = shap.Explainer(models["XGBoost"], background)
                        shap_vals  = explainer(input_df)
                        class_idx  = list(le_xgb.classes_).index(pred_label_raw)

                        fig, ax = plt.subplots()
                        shap.plots.waterfall(shap_vals[0, :, class_idx], max_display=10, show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        st.caption(
                            "This waterfall plot shows which features pushed the model toward or away from "
                            "the predicted class for your specific input values. Features in red increased "
                            "the predicted probability; features in blue decreased it."
                        )
                    except Exception as e:
                        st.warning(f"SHAP waterfall could not be generated: {e}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
