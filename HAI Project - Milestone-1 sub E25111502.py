#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================================================
# HAI Project - COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) 
#               dataset for predicting Recidivism
# ============================================================================================


# In[2]:


# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,brier_score_loss,confusion_matrix,
                            classification_report,roc_auc_score,roc_curve
                            )

# Sampling and Reweighting
from imblearn.over_sampling import RandomOverSampler

from IPython.display import display

# Fairness Evaluation
from fairlearn.metrics import (demographic_parity_difference,equalized_odds_difference,MetricFrame,selection_rate)
from fairlearn.postprocessing import ThresholdOptimizer

# Display 
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 140)

RANDOM_STATE = 42


# In[3]:


# ================================
# Loading COMPAS Dataset
# ================================

url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"

df = pd.read_csv(url)

print("Shape of dataset:")
print(df.shape)

print("\nFirst five rows:")
display(df.head())


# In[4]:


# ============================================
# Defining protected attributes 
# ============================================

# Protected attributes used in fairness analysis
protected_attributes = ["race", "sex", "age_cat"]

print("Protected Attributes Selected for Fairness Analysis:\n")

# Printing the proportion of counts of selected Protected attributes
for attribute in protected_attributes:
    print(df[attribute].value_counts(normalize=True).to_frame())
    print()


# In[5]:


# ============================================
# Specifying "two_year_recid" column as target variable
# two_year_recid = {0 => No recorded recidivism within two years}
#                  {1 => Recorded recidivism within two years}
# ============================================

target = "two_year_recid"

print("Target values:")
display(df[target].value_counts().to_frame("Count"))

print("\nTarget proportions:")
display(df[target].value_counts(normalize=True).to_frame("Proportion"))


# In[6]:


# ============================================
# Visualizing Protected attributes
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(data=df, x="race", ax=axes[0])
axes[0].set_title("Race Distribution")
axes[0].tick_params(axis='x', rotation=45)

sns.countplot(data=df, x="sex", ax=axes[1])
axes[1].set_title("Sex Distribution")

sns.countplot(data=df, x="age_cat", ax=axes[2])
axes[2].set_title("Age Category Distribution")

plt.tight_layout()
plt.show()


# In[7]:


# ============================================
# Sampling / Representation Bias
# ============================================

protected_attributes = ["race","sex","age_cat"]

for attribute in protected_attributes:
    distribution = (df[attribute].value_counts(dropna=False).rename("count").to_frame())
    distribution["proportion"] = (distribution["count"] / len(df))

    print(f"\n {attribute.upper()} ")

    display(distribution.style.format({"proportion": "{:.2%}"}))


# In[8]:


# Cross-tabulation

print("Recidivism by race:")
display(pd.crosstab(df["race"],df[target]))

print("\nRecidivism by sex:")
display(pd.crosstab(df["sex"],df[target]))

print("\nRecidivism by age category:")
display(pd.crosstab(df["age_cat"],df[target]))


# In[9]:


# ============================================
# Potential Measurement / Historical Bias check
# ============================================

# In COMPAS system - "decile_score" is the overall recidivism risk score, expressed on a scale from 1 to 10
# A higher decile_score indicates that COMPAS assessed the person as having a higher risk of recidivism
# priors_count is the number of Prior offenses recorded for an offender

print("COMPAS decile score by race:")

display(df.groupby("race")["decile_score"].agg(count="count",
                                                  mean="mean",
                                              median="median",
                                                    std="std",
                                                    min="min",
                                                    max="max").round(2))

print("\nPrior offenses by race:")

display(df.groupby("race")["priors_count"].agg(count="count",
                                                  mean="mean",
                                              median="median",
                                                    std="std",
                                                    min="min",
                                                    max="max").round(2))


# In[10]:


# Bias visualizations

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=df,x="race",y="decile_score",ax=axes[0])
axes[0].set_title("Mean COMPAS Decile Score by Race")
axes[0].tick_params(axis="x", rotation=45)

sns.barplot(data=df,x="race",y="priors_count",ax=axes[1])
axes[1].set_title("Mean Prior Offenses by Race")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()


# In[11]:


# ============================================
# Heatmap Visualization of Bias 
# ============================================

race_recid = pd.crosstab(df["race"],df["two_year_recid"],normalize="index").mul(100).round(2)

sex_recid = pd.crosstab(df["sex"],df["two_year_recid"],normalize="index").mul(100).round(2)


plt.figure(figsize=(10,5))

sns.heatmap(race_recid,annot=True,cmap="YlGnBu",fmt=".1f")

plt.title("Two-Year Recidivism (%) by Race")
plt.ylabel("Race")
plt.xlabel("Recidivism")
plt.show()

sns.heatmap(sex_recid,annot=True,cmap="YlGnBu",fmt=".1f")
plt.title("Two-Year Recidivism (%) by Gender")
plt.ylabel("Race")
plt.xlabel("Recidivism")
plt.show()


# In[12]:


# ============================================
# Selecting Features and Label and
# using a common, untouched test_set prior to 
# applying Fairness metrics
# ============================================
features = ["age","sex","race","priors_count","juv_fel_count","juv_misd_count",
            "juv_other_count","c_charge_degree"
           ]

X = df[features].copy()
y = df[target].astype(int).copy()

print("Model features:")
print(features)

# Features used for baseline model
train_idx, test_idx = train_test_split(df.index,test_size=0.2,
                                       random_state=RANDOM_STATE,stratify=y
                                      )

X_train = X.loc[train_idx].copy()
X_test = X.loc[test_idx].copy()

y_train = y.loc[train_idx].copy()
y_test = y.loc[test_idx].copy()

# Protected attributes for auditing ONLY
race_test = df.loc[test_idx, "race"]
sex_test = df.loc[test_idx, "sex"]
age_test = df.loc[test_idx, "age_cat"]

print("Training observations:", len(X_train))
print("Testing observations :", len(X_test))


# In[13]:


# ============================================
# Defining preprocessing
# ============================================


categorical_features = ["sex","race","c_charge_degree"]

numeric_features = ["age","priors_count","juv_fel_count","juv_misd_count","juv_other_count"]


def create_preprocessor(include_race=True):

    categorical = categorical_features.copy()

    if not include_race:
        categorical.remove("race")

    numeric = numeric_features.copy()

    preprocessor = ColumnTransformer(transformers=[("numeric",StandardScaler(),numeric),
                                                   ("categorical",OneHotEncoder(handle_unknown="ignore",drop="first"),
                                                    categorical)],remainder="drop")
    return preprocessor


# In[14]:


# ============================================
# Logistic Regression
# ============================================

def create_model(include_race=True):

    preprocessor = create_preprocessor(include_race=include_race)

    model = LogisticRegression(max_iter=2000,random_state=RANDOM_STATE)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor),("model", model)])

    return pipeline


# In[15]:


# ============================================
# Evaluation function
# ============================================

def evaluate_model(model, X_eval, y_eval):

    predictions = model.predict(X_eval)

    probabilities = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_eval,predictions),

        "Precision": precision_score(y_eval,predictions,zero_division=0),

        "Recall": recall_score(y_eval,predictions,zero_division=0),

        "F1": f1_score(y_eval,predictions,zero_division=0),

        "ROC-AUC": roc_auc_score(y_eval,probabilities),

        "Brier": brier_score_loss(y_eval,probabilities)
    }

    return {"metrics": metrics,"pred": predictions,"prob": probabilities}


# In[16]:


# ============================================
# Fairness evaluation function
# ============================================

def safe_divide(a, b):
    if b == 0:
        return np.nan
    return a / b


def fairness_metrics(y_true,y_pred,protected_attribute):

    data = pd.DataFrame({"y_true": np.asarray(y_true),
                         "y_pred": np.asarray(y_pred),
                         "group": np.asarray(protected_attribute)})

    rows = []

    for group, subset in data.groupby("group"):

        y_g = subset["y_true"]
        p_g = subset["y_pred"]

        tp = ((y_g == 1) & (p_g == 1)).sum()
        tn = ((y_g == 0) & (p_g == 0)).sum()
        fp = ((y_g == 0) & (p_g == 1)).sum()
        fn = ((y_g == 1) & (p_g == 0)).sum()

        selection_rate = p_g.mean()

        tpr = safe_divide(tp,tp + fn)

        fpr = safe_divide(fp,fp + tn)

        accuracy = safe_divide(tp + tn,len(subset))

        rows.append({"group": group,"count": len(subset),"selection_rate": selection_rate,
                     "TPR": tpr,"FPR": fpr,"accuracy": accuracy})

    by_group = (pd.DataFrame(rows).set_index("group"))

    selection_rates = (
        by_group["selection_rate"]
    )

    tprs = by_group["TPR"]
    fprs = by_group["FPR"]

    dp_difference = (selection_rates.max()- selection_rates.min())

    dp_ratio = safe_divide(selection_rates.min(),selection_rates.max())

    tpr_difference = (tprs.max() - tprs.min())

    fpr_difference = (fprs.max() - fprs.min())

    equalized_odds_difference = max(tpr_difference,fpr_difference)

    return {"by_group": by_group,
            "DP Difference": dp_difference,
            "DP Ratio": dp_ratio,
            "TPR Difference": tpr_difference,
            "FPR Difference": fpr_difference,
            "Equalized Odds Difference":equalized_odds_difference
           }


# In[17]:


# ============================================
# Training baseline model
# ============================================

baseline_model = create_model(include_race=True)

baseline_model.fit(X_train,y_train)

baseline = evaluate_model(baseline_model,X_test,y_test)

baseline_fair_race = fairness_metrics(y_test,baseline["pred"],race_test)

print("BASELINE MODEL PERFORMANCE")

display(pd.DataFrame([baseline["metrics"]]).round(4))


# In[18]:


print("BASELINE FAIRNESS BY RACE")

display(baseline_fair_race["by_group"].round(4))

print("\nAggregate fairness metrics:")

for key, value in baseline_fair_race.items():

    if key != "by_group":
        print(f"{key}: {value:.4f}"if pd.notna(value)
                                   else f"{key}: NaN"
             )


# In[19]:


# ============================================
# Baseline fairness by sex
# ============================================

baseline_fair_sex = fairness_metrics(y_test,baseline["pred"],sex_test)

print("BASELINE FAIRNESS BY SEX")

display(baseline_fair_sex["by_group"].round(4))


# In[20]:


# ============================================
# Baseline fairness by age group
# ============================================

baseline_fair_age = fairness_metrics(y_test,baseline["pred"],age_test)

print("BASELINE FAIRNESS BY AGE GROUP")

display(baseline_fair_age["by_group"].round(4))


# In[21]:


# ============================================
# Baseline confusion matrix
# ============================================

cm = confusion_matrix(y_test,baseline["pred"])

plt.figure(figsize=(5, 4))

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=["No Recidivism", "Recidivism"],
            yticklabels=["No Recidivism", "Recidivism"]
           )

plt.title("Baseline Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# In[22]:


# ============================================
# Baseline ROC curve
# ============================================

fpr, tpr, thresholds = roc_curve(y_test,baseline["prob"])

plt.figure(figsize=(6, 5))

plt.plot(fpr,tpr,label=f"ROC-AUC = {baseline['metrics']['ROC-AUC']:.3f}")

plt.plot([0, 1],[0, 1],linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Baseline ROC Curve")

plt.legend()
plt.tight_layout()
plt.show()


# In[23]:


# ============================================
# Feature elimination
# ============================================

feature_model = create_model(include_race=False)

X_train_no_race = X_train.drop(columns=["race"])

X_test_no_race = X_test.drop(columns=["race"])

feature_model.fit(X_train_no_race,y_train)

feature = evaluate_model(feature_model,X_test_no_race,y_test)

feature_fair_race = fairness_metrics(y_test,feature["pred"],race_test)

print("FEATURE ELIMINATION PERFORMANCE")

display(pd.DataFrame([feature["metrics"]]).round(4))

print("\nFEATURE ELIMINATION - FAIRNESS BY RACE")

display(feature_fair_race["by_group"].round(4))


# In[24]:


# ============================================
# Reweighting
# ============================================

# Calculate race frequencies ONLY in training data
race_train = df.loc[train_idx, "race"].copy()

train_race_counts = (race_train.value_counts())

n_train = len(race_train)

n_race_groups = len(train_race_counts)

# Inverse-frequency race weights
sample_weights = race_train.map(lambda group:n_train / (n_race_groups * train_race_counts[group]))

print("Training race counts:")
display(train_race_counts.rename("count").to_frame())

print("\nTraining race weights:")
display(sample_weights.describe().to_frame("weight"))


# In[25]:


# ============================================
# Training Reweighted model
# ============================================

weighted_model = create_model(include_race=True)

weighted_model.fit(X_train,y_train,model__sample_weight=sample_weights.to_numpy())

weighted = evaluate_model(weighted_model,X_test,y_test)

weighted_fair_race = fairness_metrics(y_test,weighted["pred"],race_test)

print("REWEIGHTED MODEL PERFORMANCE")

display(pd.DataFrame([weighted["metrics"]]).round(4)
)

print("\nREWEIGHTED MODEL FAIRNESS BY RACE")

display(weighted_fair_race["by_group"].round(4))


# In[26]:


# ============================================
# Training-only race-balanced Resampling
# ============================================

# Add race temporarily so that the training data can be balanced
train_for_sampling = X_train.copy()
train_for_sampling["_target"] = y_train.values

print("Original training race distribution:")

display(train_for_sampling["race"].value_counts().rename("count").to_frame())


# In[27]:


# ============================================
# Balance race representation in training only
# ============================================

# Find the largest race group in TRAINING data
max_group_size = (train_for_sampling["race"].value_counts().max())

balanced_parts = []

for race_value, group_data in (train_for_sampling.groupby("race")):

    sampled_group = group_data.sample(n=max_group_size,replace=True,random_state=RANDOM_STATE)

    balanced_parts.append(sampled_group)

train_balanced = pd.concat(balanced_parts,axis=0)

train_balanced = train_balanced.sample(frac=1,random_state=RANDOM_STATE)

X_train_balanced = train_balanced.drop(columns=["_target"])

y_train_balanced = train_balanced["_target"].astype(int)

print("Balanced training race distribution:")

display(X_train_balanced["race"].value_counts().rename("count").to_frame())

print("\nBalanced training target distribution:")

display(y_train_balanced.value_counts().rename("count").to_frame())


# In[28]:


# ============================================
# Train resampled model
# ============================================

oversampled_model = create_model(include_race=True)

oversampled_model.fit(X_train_balanced,y_train_balanced)

oversampled = evaluate_model(oversampled_model,X_test,y_test)

oversampled_fair_race = fairness_metrics(y_test,oversampled["pred"],race_test)

print("RESAMPLED MODEL PERFORMANCE")

display(pd.DataFrame([oversampled["metrics"]]).round(4))

print("\nRESAMPLED MODEL FAIRNESS BY RACE")

display(oversampled_fair_race["by_group"].round(4))


# In[29]:


# ============================================
# Calibration
# ============================================

calibrated_base_model = create_model(include_race=True)

calibrated_model = CalibratedClassifierCV(estimator=calibrated_base_model,method="sigmoid",cv=5)

calibrated_model.fit(X_train,y_train)

calibrated = evaluate_model(calibrated_model,X_test,y_test)

calibrated_fair_race = fairness_metrics(y_test,calibrated["pred"],race_test)

print("CALIBRATED MODEL PERFORMANCE")

display(pd.DataFrame([calibrated["metrics"]]).round(4))

print("\nCALIBRATED MODEL FAIRNESS BY RACE")

display(calibrated_fair_race["by_group"].round(4))


# In[30]:


# ============================================
# Collect all model results
# ============================================

models = {
    "Baseline": {
        "result": baseline,
        "fairness": baseline_fair_race
    },

    "Feature Elimination": {
        "result": feature,
        "fairness": feature_fair_race
    },

    "Reweighting": {
        "result": weighted,
        "fairness": weighted_fair_race
    },

    "Resampling": {
        "result": oversampled,
        "fairness": oversampled_fair_race
    },

    "Calibration": {
        "result": calibrated,
        "fairness": calibrated_fair_race
    }
}


# In[31]:


# ============================================
# Overall performance comparison
# ============================================

performance_rows = []

for model_name, model_data in models.items():

    row = {"Model": model_name}

    row.update(model_data["result"]["metrics"])

    performance_rows.append(row)

performance_comparison = (pd.DataFrame(performance_rows).set_index("Model"))

print("OVERALL PERFORMANCE COMPARISON")

display(performance_comparison.round(4))


# In[32]:


# ============================================
# Fairness comparison
# ============================================

fairness_rows = []

for model_name, model_data in models.items():

    fairness = model_data["fairness"]

    fairness_rows.append({"Model": model_name,"DP Difference":fairness["DP Difference"],
                          "DP Ratio":fairness["DP Ratio"],
                          "TPR Difference":fairness["TPR Difference"],
                          "FPR Difference":fairness["FPR Difference"],
                          "Equalized Odds Difference":fairness["Equalized Odds Difference"]
                         })

fairness_comparison = (pd.DataFrame(fairness_rows).set_index("Model"))

print("FAIRNESS COMPARISON — RACE")

display(fairness_comparison.round(4))


# In[33]:


# ============================================
# Calculating change from baseline
# ============================================

baseline_performance = (performance_comparison.loc["Baseline"])

baseline_fairness = (fairness_comparison.loc["Baseline"])

performance_change = (performance_comparison.subtract(baseline_performance))

fairness_change = (fairness_comparison.subtract(baseline_fairness))

print("CHANGE IN PERFORMANCE RELATIVE TO BASELINE")

display(performance_change.round(4))

print("\nCHANGE IN FAIRNESS METRICS RELATIVE TO BASELINE")

display(fairness_change.round(4))


# In[34]:


# ============================================
# Performance graph
# ============================================

performance_comparison[["Accuracy",
                        "Precision",
                        "ROC-AUC"]
    ].plot(kind="bar",figsize=(12, 6))

plt.title("Performance Comparison: Baseline vs Mitigation")

plt.ylabel("Score")
plt.xlabel("Model")

plt.xticks(rotation=20,ha="right")

plt.ylim(0, 1)

plt.tight_layout()
plt.show()


# In[35]:


performance_comparison[["Recall",
                        "F1",
                       ]
    ].plot(kind="bar",figsize=(12, 6))

plt.title("Performance Comparison: Baseline vs Mitigation")

plt.ylabel("Score")
plt.xlabel("Model")

plt.xticks(rotation=20,ha="right")

plt.ylim(0, 1)

plt.tight_layout()
plt.show()


# In[36]:


# ============================================
# Fairness graph
# ============================================

fairness_comparison[["DP Difference","TPR Difference","FPR Difference","Equalized Odds Difference"]].plot(kind="bar",figsize=(12, 6))

plt.title("Fairness Comparison by Race")

plt.ylabel("Difference")
plt.xlabel("Model")

plt.xticks(rotation=20,ha="right")

plt.tight_layout()
plt.show()


# In[37]:


# ============================================
# Demographic parity ratio graph
# ============================================

fairness_comparison[["DP Ratio"]].plot(kind="bar",figsize=(9, 5))

plt.axhline(1.0,linestyle="--",label="Perfect parity")

plt.title("Demographic Parity Ratio for each Model")

plt.ylabel("Minimum selection rate / Maximum selection rate")

plt.xlabel("Model")

plt.xticks(rotation=20,ha="right")

plt.legend()

plt.tight_layout()
plt.show()


# In[38]:


# ============================================
# Equalized Odds Difference graph
# ============================================

fairness_comparison[["Equalized Odds Difference"]].plot(kind="bar",figsize=(9, 5))

plt.axhline(0.0,linestyle="--",label="Perfect equality")

plt.title("Equalized Odds Difference for each Model")

plt.ylabel("Equalized Odds Difference")

plt.xlabel("Model")

plt.xticks(rotation=20,ha="right")

plt.legend()

plt.tight_layout()
plt.show()


# In[39]:


# ============================================
# Subgroup performance function
# ============================================

def subgroup_performance(y_true,y_pred,protected_attribute):

    data = pd.DataFrame({"y_true": np.asarray(y_true),"y_pred": np.asarray(y_pred),"group": np.asarray(protected_attribute)})

    rows = []

    for group, subset in data.groupby("group"):

        rows.append({"group": group,"count": len(subset),
                     "accuracy":accuracy_score(subset["y_true"],subset["y_pred"]),
                     "precision":precision_score(subset["y_true"],subset["y_pred"],zero_division=0
                                                ),
                     "recall":recall_score(subset["y_true"],subset["y_pred"],zero_division=0),
                     "F1":f1_score(subset["y_true"],subset["y_pred"],zero_division=0)}
                   )

    return (pd.DataFrame(rows).set_index("group"))


# In[40]:


# ============================================
# Baseline subgroup performance
# ============================================

print("BASELINE PERFORMANCE BY RACE")

baseline_subgroup = subgroup_performance(y_test,baseline["pred"],race_test)

display(baseline_subgroup.round(4))


# In[41]:


# ============================================
# Reweighted subgroup performance
# ============================================

print("REWEIGHTED PERFORMANCE BY RACE")

weighted_subgroup = subgroup_performance(y_test,weighted["pred"],race_test)

display(weighted_subgroup.round(4))


# In[42]:


# ============================================
# Comparing baseline and reweighted subgroup performance
# ============================================

comparison_subgroup = pd.concat({"Baseline": baseline_subgroup,
                                 "Reweighting": weighted_subgroup},axis=1
                               )

display(comparison_subgroup.round(4))


# In[43]:


# ============================================
# All models: subgroup fairness table
# ============================================

all_subgroup_fairness = {}

for model_name, model_data in models.items():

    all_subgroup_fairness[model_name] = model_data["fairness"]["by_group"].copy()

print("Subgroup fairness tables are available in:")
print("all_subgroup_fairness")

for model_name, table in all_subgroup_fairness.items():

    print(f"\n===== {model_name.upper()} =====")

    display(table.round(4))


# In[44]:


# ============================================
# Fairness by sex for all models
# ============================================

sex_fairness_results = {}

for model_name, model_data in models.items():

    sex_fairness_results[model_name] = fairness_metrics(y_test,model_data["result"]["pred"],sex_test)

for model_name, result in sex_fairness_results.items():

    print(f"\n===== {model_name.upper()} — SEX =====")

    display(result["by_group"].round(4))


# In[45]:


# ============================================
# Fairness by age group for all models
# ============================================

age_fairness_results = {}

for model_name, model_data in models.items():

    age_fairness_results[model_name] = fairness_metrics(y_test,model_data["result"]["pred"],age_test)

for model_name, result in age_fairness_results.items():

    print(f"\n===== {model_name.upper()} — AGE =====")

    display(result["by_group"].round(4))


# In[46]:


# ============================================
# Identifying underrepresented groups
# ============================================

representation_threshold = 0.01

race_distribution = (df["race"].value_counts(normalize=True))

potentially_underrepresented = (race_distribution[race_distribution <representation_threshold])

print("Groups representing less than 1% of the dataset:")

display(potentially_underrepresented.to_frame("proportion").style.format({"proportion": "{:.2%}"}))


# In[47]:


# ============================================
# inspecting data choose model
# ============================================

print("PERFORMANCE")
display(performance_comparison.round(4))

print("\nFAIRNESS")
display(fairness_comparison.round(4))


# In[48]:


candidate_models = fairness_comparison.sort_values(by="DP Difference")

display(candidate_models.round(4))


# In[49]:


baseline_dp = fairness_comparison.loc["Baseline","DP Difference"]

baseline_f1 = performance_comparison.loc["Baseline","F1"]

eligible = performance_comparison[performance_comparison["F1"]>= baseline_f1 - 0.03].index

selection_table = fairness_comparison.loc[eligible].sort_values(by="DP Difference")

print("Candidate models satisfying the F1 constraint:")
display(selection_table.round(4))


# In[50]:


# ============================================
# Saving evaluation tables
# ============================================

performance_comparison.to_csv("compas_performance_comparison.csv")

fairness_comparison.to_csv("compas_fairness_comparison.csv")


print("Evaluation tables saved.")


# In[51]:


import joblib
from pathlib import Path

final_model = feature_model
home_dir = Path.home()
download_path = home_dir / "Downloads" / "compas_final_model.pkl"

joblib.dump(final_model,filename=download_path)

print("Final model saved as compas_final_model.pkl")


# In[52]:


loaded_model = joblib.load("compas_final_model.pkl")

verification = evaluate_model(loaded_model,X_test,y_test)

print("FINAL MODEL VERIFICATION")

display(pd.DataFrame([verification["metrics"]]).round(4))


# In[53]:


final_comparison = performance_comparison.copy()

final_comparison["DP Difference"] = fairness_comparison["DP Difference"]

final_comparison["DP Ratio"] = fairness_comparison["DP Ratio"]

final_comparison["Equalized Odds Difference"] = fairness_comparison["Equalized Odds Difference"]

display(final_comparison.round(4))

