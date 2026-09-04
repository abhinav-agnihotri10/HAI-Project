#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# In[2]:


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Responsible Recidivism Prediction",
    page_icon="⚖️",
    layout="wide"
)


# In[4]:


# ============================================================
# LOADING TRAINED MODEL
# ============================================================

APP_DIR = Path(__file__).parent
file_path = APP_DIR / "compas_final_model.pkl"

# home_dir = Path.home()
# file_path = home_dir / "Downloads" / "compas_final_model.pkl"

if not file_path.exists():
    st.error(f"""Model file was not found.Expected location:{file_path}"""
            )
    st.stop()

model = joblib.load(file_path)


# In[5]:


# The saved object is expected to be:
# Pipeline([
#   ("preprocessor", ColumnTransformer(...)),
#   ("model", LogisticRegression(...))
# ])

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]


# In[7]:


# ============================================================
# EXPLAINABILITY FUNCTIONS
# ============================================================

def clean_feature_name(name):
    """
    Converts transformed sklearn feature names into
    human-readable labels for the interface.
    """
    replacements = {"numeric__age": "Age",
                    "numeric__priors_count": "Prior offences",
                    "numeric__juv_fel_count": "Juvenile felony count",
                    "numeric__juv_misd_count": "Juvenile misdemeanor count",
                    "numeric__juv_other_count": "Other juvenile offence count",
                    "categorical__sex_Male": "Sex = Male",
                    "categorical__c_charge_degree_M": "Charge degree = Misdemeanor",
                   }
    return replacements.get(name, name.replace("numeric__", "").replace("categorical__", ""))


def get_global_explanation():
    """
    Returns the Logistic Regression coefficients.

    For numeric features, the pipeline standardizes the variables.
    Therefore each numeric coefficient corresponds approximately
    to a one-standard-deviation increase in that feature.

    Positive coefficient -> pushes prediction toward class 1.
    Negative coefficient -> pushes prediction toward class 0.
    """
    transformed_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    df = pd.DataFrame({"Feature": [clean_feature_name(x) for x in transformed_names],
                       "Coefficient": coefficients
                      })

    df["Absolute importance"] = df["Coefficient"].abs()
    return df.sort_values("Absolute importance", ascending=True)


def get_local_explanation(case):
    """
    Exact additive explanation for this Logistic Regression model.

    Logistic Regression:
        log_odds = intercept + sum(transformed_feature_i * coefficient_i)

    Each row below is therefore the exact contribution of one
    transformed feature to this case's log-odds.

    Positive contribution -> pushes toward class 1.
    Negative contribution -> pushes toward class 0.
    """
    transformed_case = preprocessor.transform(case)

    if hasattr(transformed_case, "toarray"):
        transformed_case = transformed_case.toarray()

    transformed_case = np.asarray(transformed_case)[0]
    transformed_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    contributions = transformed_case * coefficients

    df = pd.DataFrame({
        "Feature": [clean_feature_name(x) for x in transformed_names],
        "Transformed value": transformed_case,
        "Coefficient": coefficients,
        "Contribution": contributions
    })

    df["Absolute contribution"] = df["Contribution"].abs()
    return df.sort_values("Absolute contribution", ascending=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[8]:


# ============================================================
# HEADER
# ============================================================

st.title("Responsible Recidivism Prediction Demonstrator")

st.info(
    """
    This application demonstrates a machine-learning model that estimates
    the probability of the recorded two-year recidivism outcome.

    The prediction is a statistical estimate, not a determination of
    future behaviour. The system is intended as an educational
    Human-AI Interaction and Responsible-AI prototype.
    """
)


# In[9]:


# =============
# TABS
# =============

tab_predict, tab_global, tab_about = st.tabs(["Prediction & Local Explanation",
                                              "Global Model Explanation",
                                              "How to Interpret Explanations"]
                                            )


# In[10]:


# ============================================================
# TAB 1 — PREDICTION + LOCAL EXPLANATION
# ============================================================

with tab_predict:

    st.header("Case Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

        priors_count = st.number_input(
            "Number of Prior Offences",
            min_value=0,
            value=0,
            step=1
        )

        juv_fel_count = st.number_input(
            "Juvenile Felony Count",
            min_value=0,
            value=0,
            step=1
        )

        juv_misd_count = st.number_input(
            "Juvenile Misdemeanor Count",
            min_value=0,
            value=0,
            step=1
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            ["Female", "Male"]
        )

        juv_other_count = st.number_input(
            "Other Juvenile Offence Count",
            min_value=0,
            value=0,
            step=1
        )

        charge_degree = st.selectbox(
            "Charge Degree",
            ["F", "M"],
            format_func=lambda x:
            "Felony" if x == "F" else "Misdemeanor"
        )

    analyse = st.button(
        "ANALYSE CASE",
        type="primary"
    )

    if analyse:

        case = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "priors_count": [priors_count],
            "juv_fel_count": [juv_fel_count],
            "juv_misd_count": [juv_misd_count],
            "juv_other_count": [juv_other_count],
            "c_charge_degree": [charge_degree]
        })

        prediction = int(model.predict(case)[0])
        probability = float(model.predict_proba(case)[0, 1])

        st.divider()
        st.header("Model Estimate")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Estimated probability of recorded two-year recidivism",
                f"{probability:.1%}"
            )

        with metric_col2:
            st.metric(
                "Model classification",
                "Positive prediction" if prediction == 1
                else "Negative prediction"
            )

        if 0.45 <= probability <= 0.55:
            st.warning(
                """
                The predicted probability is close to the model's
                classification threshold. Interpret this result
                with additional caution.
                """
            )

        st.warning(
            """
            This is a statistical model estimate. It is NOT a determination
            that an individual will or will not reoffend. The model may make
            mistakes and should not be used as the sole basis for
            consequential decisions.
            """
        )

        # ----------------------------------------------------
        # LOCAL EXPLANATION
        # ----------------------------------------------------

        st.divider()
        st.header("Why did the model give this result?")

        st.write(
            """
            The chart below shows how each input contributed to this
            individual prediction.

            **Positive contribution:** pushes the model toward the
            recorded recidivism class (1).

            **Negative contribution:** pushes the model toward the
            no-recorded-recidivism class (0).
            """
        )

        local_df = get_local_explanation(case)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.barh(
            local_df["Feature"],
            local_df["Contribution"]
        )
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Contribution to model log-odds")
        ax.set_title("Local feature contributions for this case")
        fig.tight_layout()
        st.pyplot(fig)

        # exact reconstruction of model output
        intercept = float(classifier.intercept_[0])
        contribution_sum = float(local_df["Contribution"].sum())
        local_log_odds = intercept + contribution_sum
        reconstructed_probability = sigmoid(local_log_odds)

        with st.expander("See technical explanation"):
            st.write(
                """
                Logistic Regression combines an intercept with the contribution
                from every transformed input:
                """
            )
            st.code(
                "log-odds = intercept + Σ(feature value × coefficient)"
            )

            st.write(f"Model intercept: **{intercept:.3f}**")
            st.write(
                f"Sum of feature contributions: **{contribution_sum:.3f}**"
            )
            st.write(
                f"Reconstructed probability: "
                f"**{reconstructed_probability:.1%}**"
            )

            display_local = local_df[
                ["Feature", "Contribution"]
            ].copy()

            display_local["Contribution"] = (
                display_local["Contribution"].round(4)
            )

            st.dataframe(
                display_local.sort_values(
                    "Contribution",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

        # Plain language top factors
        increasing = (
            local_df[local_df["Contribution"] > 0]
            .sort_values("Contribution", ascending=False)
            .head(3)
        )

        decreasing = (
            local_df[local_df["Contribution"] < 0]
            .sort_values("Contribution", ascending=True)
            .head(3)
        )

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Main factors pushing the estimate upward")
            if len(increasing) == 0:
                st.write("No feature made a positive contribution.")
            else:
                for _, row in increasing.iterrows():
                    st.write(
                        f"• {row['Feature']} "
                        f"(contribution {row['Contribution']:.3f})"
                    )

        with c2:
            st.subheader("Main factors pushing the estimate downward")
            if len(decreasing) == 0:
                st.write("No feature made a negative contribution.")
            else:
                for _, row in decreasing.iterrows():
                    st.write(
                        f"• {row['Feature']} "
                        f"(contribution {row['Contribution']:.3f})"
                    )

        st.caption(
            """
            These contributions describe how the model calculated this
            prediction. They do not show causal effects and should not be
            interpreted as statements that a feature causes recidivism.
            """
        )


# In[11]:


# ===============================
# TAB 2 — GLOBAL EXPLANATION
# ===============================

with tab_global:

    st.header("How does the model generally make predictions?")

    st.write(
        """
        The deployed model is Logistic Regression. Its learned coefficients
        provide a transparent global description of how the model generally
        combines features.

        **Positive coefficients** push predictions toward recorded
        two-year recidivism.

        **Negative coefficients** push predictions toward no recorded
        two-year recidivism.

        Larger absolute values indicate stronger influence in the fitted
        model. Numeric inputs were standardized during training, so their
        coefficients represent changes on the standardized scale.
        """
    )

    global_df = get_global_explanation()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(
        global_df["Feature"],
        global_df["Coefficient"]
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Logistic Regression coefficient")
    ax.set_title("Global model coefficients")
    fig.tight_layout()
    st.pyplot(fig)

    display_global = global_df[
        ["Feature", "Coefficient", "Absolute importance"]
    ].copy()

    display_global["Coefficient"] = (
        display_global["Coefficient"].round(4)
    )

    display_global["Absolute importance"] = (
        display_global["Absolute importance"].round(4)
    )

    st.dataframe(
        display_global.sort_values(
            "Absolute importance",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        Important: coefficient magnitude describes the behaviour of this
        fitted model, not causal importance in the real world. Correlated
        features and data-collection processes can affect coefficients.
        """
    )


# In[12]:


# ==============================
# TAB 3 — EXPLANATION STRATEGY
# ==============================

with tab_about:

    st.header("How to Interpret the Explanations")

    st.subheader("1. Global explanation")

    st.write(
        """
        The global coefficient view answers questions such as:

        • What features does the model generally rely on?

        • Which features tend to increase or decrease predicted risk?

        • Is the model behaving in a way that is broadly understandable?

        This is useful for analysts, reviewers, and model auditors who want
        to understand the model as a whole rather than one particular case.
        """
    )

    st.subheader("2. Local explanation")

    st.write(
        """
        The local contribution view answers questions such as:

        • Why did this particular case receive this prediction?

        • Which inputs pushed the estimate upward?

        • Which inputs pushed the estimate downward?

        • Was the result driven mainly by one feature or by several features?

        Because the classifier is Logistic Regression, these local
        contributions are calculated directly from the fitted model and
        exactly reconstruct its log-odds for the case.
        """
    )

    st.subheader("3. Why not use the explanation as a causal statement?")

    st.write(
        """
        An explanation describes the model's computation, not the real-world
        cause of recidivism. For example, a positive contribution from prior
        offences means that this feature increased the model's prediction;
        it does not prove that prior offences caused the future outcome.
        """
    )

    st.subheader("4. Why this approach is suitable for this model")

    st.write(
        """
        The deployed model is already interpretable Logistic Regression.
        Native coefficient and contribution explanations are therefore
        preferred for the primary interface because they are deterministic,
        computationally efficient, and faithful to the actual prediction.

        Model-agnostic tools such as LIME or SHAP can be useful when the
        underlying model is more complex. For this particular classifier,
        adding an approximate surrogate explanation is unnecessary for the
        core interface and can make interpretation harder.
        """
    )

    st.warning(
        """
        Explainability does not make the prediction automatically fair,
        correct, or appropriate for high-impact use. Explanation should be
        considered together with predictive performance, uncertainty,
        subgroup fairness, dataset limitations, and responsible-use
        constraints.
        """
    )

