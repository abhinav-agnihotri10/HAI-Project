import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path

# ---------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------
APP_DIR = Path(__file__).resolve().parent
file_path = APP_DIR / "compas_final_model.pkl"

if not file_path.exists():
    st.error(f"""Model file was not found.
    Expected location:{file_path}"""
    )
    st.stop()

model = joblib.load(filename=file_path)
print("Model loaded successfully")

# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="Responsible Recidivism Prediction",
    page_icon="⚖️",
    layout="wide"
)

# ---------------------------------------
# HEADER
# ---------------------------------------

st.title("Responsible Recidivism Prediction Demonstrator")

st.info("""This application demonstrates a machine-learning model
    that estimates the probability of recorded two-year
    recidivism.

    It is intended as an educational Human-AI Interaction
    and Responsible-AI prototype."""
)


# ---------------------------------------
# INPUT SECTION
# ---------------------------------------

st.header("Case Information")


age = st.number_input("Age",min_value=18,max_value=100,value=30,step=1)

sex = st.selectbox("Sex",["Male", "Female"])

priors_count = st.number_input("Number of Prior Offences",
                               min_value=0,value=0,step=1
                               )


juv_fel_count = st.number_input("Juvenile Felony Count",
                                min_value=0,value=0,step=1
                                )


juv_misd_count = st.number_input("Juvenile Misdemeanor Count",
                                 min_value=0,value=0,step=1
                                 )


juv_other_count = st.number_input("Other Juvenile Offence Count",
                                  min_value=0,value=0,step=1
                                  )


charge_degree = st.selectbox("Charge Degree",["F", "M"],
                             format_func=lambda x:
                             "Felony" if x == "F"
                                      else "Misdemeanor"
                             )

# ---------------------------------------
# PREDICTION BUTTON
# ---------------------------------------

analyse = st.button("ANALYSE CASE",type="primary")


# ---------------------------------------
# MAKE PREDICTION
# ---------------------------------------

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


    prediction = model.predict(case)[0]

    probability = model.predict_proba(case)[0, 1]


    # -----------------------------------
    # RESULT
    # -----------------------------------

    st.header("Model Estimate")

    st.metric(
        "Estimated probability of recorded "
        "two-year recidivism",
        f"{probability:.1%}"
    )

    if prediction == 1:

        st.write(
            "**Model classification:** "
            "Positive prediction"
        )

    else:

        st.write(
            "**Model classification:** "
            "Negative prediction"
        )


    # -----------------------------------
    # UNCERTAINTY WARNING
    # -----------------------------------

    if 0.45 <= probability <= 0.55:
        st.warning(
            """
            The predicted probability is close to the
            classification threshold.

            Please Interpret this result with additional caution.
            """
        )


    # -----------------------------------
    # RESPONSIBLE USE
    # -----------------------------------

    st.warning(
        """
        This is a statistical model estimate.

        It is NOT a determination that an individual
        will or will not reoffend.

        The model may make mistakes and should not be
        used as the sole basis for consequential
        decisions.
        """
    )
