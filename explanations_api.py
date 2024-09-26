__author__ = "Everton Romanzini Colombo"
__email__ = "everton.colombo@students.ic.unicamp.br"    # if you're reading this, feel free to reach out!
__credits__ = ["Everton Romanzini Colombo", "Larissa Ayumi Okabayashi"]

# v2 differs from v1 in that it now uses a logistic regression model instead of a random forest model, and it uses the MAPOCAM algorithm to generate counterfactuals.

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
# import json

import pandas as pd
import numpy as np

# import xgboost as xgb

from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
import shap

from cfmining.algorithms import MAPOCAM
from cfmining.predictors import MonotoneClassifier
from cfmining.visualization import buildTable
from cfmining.action_set import ActionSet
from cfmining.criteria import PercentileCalculator, PercentileCriterion
from sklearn.linear_model import LogisticRegression

app = FastAPI()

## Setting up context:
data = pd.read_csv("data/german_encoded.csv") # Load the data
X = data.drop(columns=["GoodCustomer", "OtherLoansAtStore"]) # OtherLoansAtStore was droped, since it only contained one value, and mapocam didnt like that
y = data["GoodCustomer"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Check for columns in X that have only one unique value
# single_value_columns = [col for col in X.columns if X[col].nunique() == 1]
# if single_value_columns:
#     print(f"Columns with a single unique value: {single_value_columns}")
# else:
#     print("No columns with a single unique value found.")

clf = LogisticRegression()
clf.fit(X_train, y_train)

## MAPOCAM SETUP:
clf_xgb_ = MonotoneClassifier(clf, X_train, y_train, threshold=0.5)
coefficients = clf.coef_[0]
intercept = clf.intercept_[0]

action_set = ActionSet(X = X)
action_set.embed_linear_clf(coefficients=coefficients)

for feat in action_set:
    if feat.name in ['LoanDuration']:
        feat.step_type ="absolute"
        feat.step_size = 6
        feat.update_grid()
    else:
        feat.step_type ="relative"
        feat.step_size = 0.1
    feat.update_grid()

for feat in action_set:
    if feat.name in ['Age', 'Single', 'JobClassIsSkilled', 'ForeignWorker', 'OwnsHouse', 'RentsHouse', 'isMale', 'Age', 'LoanDuration', 'LoanAmount']:
        feat.mutable = False
        # feat.flip_direction = 1
        # feat.step_direction = 1
    if feat.name in ['LoanRateAsPercentOfIncome', 'NumberOfOtherLoansAtBank', 'MissedPayments',
                     'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank', 'RentsHouse', 'Unemployed', 'YearsAtCurrentJob_lt_1']:
        feat.mutable = True
        # feat.flip_direction = -1
        # feat.step_direction = -1
    if feat.name in ['YearsAtCurrentHome', 'NumberOfLiableIndividuals', 'HasTelephone', 'CheckingAccountBalance_geq_0',
                     'CheckingAccountBalance_geq_200', 'SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500',
                     'NoCurrentLoan', 'HasCoapplicant', 'HasGuarantor', 'OwnsHouse', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled']:
        feat.mutable = True
        # feat.flip_direction = 1
        # feat.step_direction = 1
    feat.update_grid()

# Preparing the criteria:
percCalc = PercentileCalculator(action_set=action_set)

@app.get("/counterfactual")
def get_mapocam_counterfactuals(client_index: int):
    individual = X.loc[client_index].values
    percCriteria = PercentileCriterion(individual, percCalc)

    en_nd_feat = MAPOCAM(action_set, individual, clf_xgb_, max_changes=3, recursive=True, clean_suboptimal=False)
    en_nd_feat.fit()
    names = action_set.df['name'].values
    counterfactuals = buildTable(en_nd_feat, individual, percCriteria, names, include_original=True, include_cost=False)
    c_t = counterfactuals.T
    # Return a sample of the first entry plus five random entries of counterfactuals:
    sample =  c_t.iloc[[0]].append(c_t.sample(5)).T
    # convert all NaN floats in sample to -1:
    sample = sample.fillna(-1)

    return JSONResponse(sample.to_dict())


## Defining endpoints:
@app.get("/client")
def get_client(index: int):
    """Get the client data for a given client index.
    
    :param client_index: The index of the client for which to get data.
    :return: A json of the form {"client_index": [CLIENT_INDEX], "client_data": {"feature_name": [FEATURE_VALUE], ...}}
    """

    # client_data = {feature_name: feature_value for feature_name, feature_value in zip(X.columns, X.iloc[index])}
    # {"feature_name": [FEATURE_VALUE], ...}
    client_class = y.iloc[index]

    # Convert numpy types to native python types, so that they can be serialized to json
    client_data = {feature_name: feature_value.item() if isinstance(feature_value, np.generic) else feature_value for feature_name, feature_value in zip(X.columns, X.iloc[index])}
    client_class = y.iloc[index].item() if isinstance(y.iloc[index], np.generic) else y.iloc[index]
    predicted_class = clf.predict(X.iloc[[index]])[0].item()

    return JSONResponse(content={"client_index": index, "client_class": client_class, "predicted_class": predicted_class, "client_data": client_data})


@app.get("/pdp")
def get_pdp_results(feature_name: str):
    """Get partial dependence results for a given feature name.

    :param feature_name: The name of the feature for which to get partial dependence results.
    :return: A json of the form {"feature_name": [FEATURE_NAME], "pdp_results": [{"feature_value": [FEATURE_VALUE], "average_target_value": [AVERAGE_TARGET_VALUE]}, ...]}
    """

    raw_results = partial_dependence(clf, X, [feature_name])
    partial_dependence_results = [{"feature_value": feature_value, "average_target_value": average_target_value} 
                                  for feature_value, average_target_value in zip(raw_results["grid_values"][0].astype(float), raw_results["average"][0].astype(float))]
    # [{"feature_value": [FEATURE_VALUE], "average_target_value": [AVERAGE_TARGET_VALUE]}, ...]

    return JSONResponse({"feature_name": feature_name, "pdp_results": partial_dependence_results})

@app.get("/shap")
def get_shap_results(client_index: int):
    """Get SHAP results for a given client index.
    
    :param client_index: The index of the client for which to get SHAP results.
    :return: A json of the form {"client_index": [CLIENT_INDEX], "shap_results": [{"feature": [FEATURE], "shap_value": [SHAP_VALUE]}, ...]}, orderdered by descending feature importance.
    """

    explainer = shap.LinearExplainer(clf, X_train)
    shap_values = explainer(X)

    sorted_feature_indices = np.argsort(abs(shap_values.values[client_index]))[::-1]
    sorted_features = X.columns[sorted_feature_indices]
    sorted_shap_values = shap_values.values[client_index][sorted_feature_indices].astype(float)

    client_data = X.iloc[client_index]
    shap_results = [{"feature": feature, "feature_value": client_data[feature].item(), "shap_value": shap_value} 
                    for feature, shap_value in zip(sorted_features, sorted_shap_values)] # [{"feature": [FEATURE], "feature_value": [FEATURE_VALUE], "shap_value": [SHAP_VALUE]}, ...]
    print(shap_results)

    return JSONResponse(content={"client_index": client_index, "shap_results": shap_results})


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)