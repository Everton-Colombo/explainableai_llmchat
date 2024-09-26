__author__ = "Everton Romanzini Colombo"
__email__ = "everton.colombo@students.ic.unicamp.br"    # if you're reading this, feel free to reach out!
__credits__ = ["Everton Romanzini Colombo", "Larissa Ayumi Okabayashi"]

import requests
from groq import *
# import json
import gradio as gr
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import os

# v2 differs from v1 in that it uses langchain instead of groq directly, so the code is tidier and more readable.
# It also now allows for counterfactual explanations to be generated using the MAPOCAM algorithm.
# The free chat now also keeps track of previous prompts and tool calls.

EXPLANATION_API_URL = "http://localhost:5000"   # make sure explanations_api.py is already running
# client = Groq(api_key="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")
os.environ["GROQ_API_KEY"] = "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"
MODEL = 'llama-3.1-70b-versatile'
llm = ChatGroq(model=MODEL) # MAKE SURE GROQ API IS IN VENV, RUN: export GROQ_API_KEY='[YOUR_API_KEY]'

SYSTEM_PROMPT = """
                You are an assistant for bank costumers who want to know more about the decisions of the 
                bank's algorithm that is used to classify them as either a good payer or a 
                bad payer. Questions regarding the reasons why a client was classified as either a good or a bad 
                customer, or how a certain feature contributes the model's decison, should be answered 
                without giving too much away about the inner workings of the machine learning model used to make those decisions. 
                You'll have access to explainable AI tools, such as Partial Dependence and SHAP values, and 
                you should interpret their results to give a non-techinical response to the customers' questions.

                Whenever referencing a feature name, use the 'get_feature_description' function to get a description of the feature,
                so that you NEVER use the feature's variable name directly in your response. e.g.: instead of saying 'LoanAmount',
                you should use the get_feature_description function to come up with something like 'The amount of the loan in euros'. 

                Whenever you need to know information about a specific client, such as their features and how the model 
                classified them, use the 'get_client_info' function. This function will return the client's features,
                the client's actual classification of whether thay are or are not a good payer, and the model's prediction for that client.

                Whenever a question about a specific feature's impact on the final decision is asked, use 
                the 'get_pdp_results' function to get the Partial Dependence results for a given feature.
                When using this function, you should be able to identify the range of values for the feature
                that make the model more likely to classify the customer as a good or a bad payer, (i.e. values
                for the feature that maximize and minimize the average target value).

                Whenever a question about a specific instance (client) is asked, use the 'get_shap_results' function
                to get the SHAP values for the features fo a given client index. When using this function, you should
                be able to identify the features that have the most impact on the model's decision for that specific client. 
                Limit your analysis to the THREE most relevant features for the client.

                Whenever a prompt askas for information on actions that a client could take to change the model's decision,
                use the 'get_counterfactual' function to get the counterfactual explanations for a given client index.
                Counterfactual explanations are the minimal changes to the client's features that would change the model's decision.
                Use this information to give the client advice on how they could change their features to be classified as a good payer.

                When possible, use as many of the functions to give a more complete answer to the customer's question.
                Avoid giving answers that contain probabilities or technical terms that the customer might not understand. 
                Try as much as possible to round the values to up to 2 significant digits.
                When referecing the target value, always use its product name of 'probability of being a good customer'.
                Always reference the model as "the bank's algorithm" or "the system". 
                NEVER use terms like "machine learning model", "feature", "target_variable", "SHAP", "PDP", "Partial Dependence", "counterfactual", etc; instead, describe what they mean.
                Avoid revealing the inner workings of the model at all costs, never metion the features variable names or categorical mapping values directly.
                """

@tool
def get_client_info(client_index):
    """
    Gets the client data for a given client index. The results are of the form 
    {'client_index': [CLIENT_INDEX], , 'client_class': [CLIENT_CLASS], 'predicted_class': [PREDICTED_CLASS], 'client_data': {'feature_name': [FEATURE_VALUE], ...}}.
    Use this function to answer questions about how a specific client was classified by the model.
    'predicted_class' is the model's prediction for the client, 'client_class' is the actual class of the client, and 'client_data' is the client's features.
    
    client_index: The index of the client for which to get data.
    """
    print("get_client_info() was called")

    url = f"{EXPLANATION_API_URL}/client?index={client_index}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "API request failed", "status_code": response.status_code}

@tool
def get_feature_description(feature_variable_name):
    """
    Gets the description of a feature given its variable name. Use this function to get a non-technical description of a feature
    when answering questions about how a specific feature impacts the model's decision.

    feature_variable_name: The name of the feature for which to get a description.
    """

    print("get_feature_description() was called")

    return {
        'GoodCustomer':                     "Categorical variable, where 1 indicates that the client is a good customer and 0 indicates that the client is a bad customer. This is the target variable that the model is trying to predict.",
        'Gender':                           "Categorical variable, where 1 indicates that the client is male, and 0 indicates that the client is female.",
        'ForeignWorker':                    "Categorical variable, where 1 indicates that the client is a foreign worker, and 0 indicates that the client is not a foreign worker.",
        'Single':                           "Categorical variable, where 1 indicates that the client is single, and 0 indicates that the client is not single.",
        'Age':                              "The age of the client in years.",
        'LoanDuration':                     "The duration of the loan in months.",
        'PurposeOfLoan':                    "The purpose of the loan, encoded as a categorical variable, where 1 means 'Education', 2 means 'Electronics', 3 means 'Furniture', 4 means 'Home Appliances', 5 means 'New Car', 6 means 'Other', 7 means 'Repairs', 8 means 'Retraining', and 9 means 'Used Car'.",
        'LoanAmount':                       "The amount of the loan in euros.",
        'LoanRateAsPercentOfIncome':        "The rate of the loan as a percentage of the client's income.",
        'YearsAtCurrentHome':               "The number of years the client has lived at their current home.",
        'NumberOfOtherLoansAtBank':         "The number of other loans the client has at the bank.",
        'NumberOfLiableIndividuals':        "The number of individuals that are liable for the loan.",
        'HasTelephone':                     "Categorical variable, where 1 indicates that the client has a telephone, and 0 indicates that the client does not have a telephone.",
        'CheckingAccountBalance_geq_0':     "Categorical variable, where 1 indicates that the client has a checking account balance greater than or equal to 0 euros, and 0 indicates that the client does not have a checking account balance greater than or equal to 0 euros.",
        'CheckingAccountBalance_geq_200':   "Categorical variable, where 1 indicates that the client has a checking account balance greater than or equal to 200 euros, and 0 indicates that the client does not have a checking account balance greater than or equal to 200 euros.",
        'SavingsAccountBalance_geq_100':    "Categorical variable, where 1 indicates that the client has a savings account balance greater than or equal to 100 euros, and 0 indicates that the client does not have a savings account balance greater than or equal to 100 euros.",
        'SavingsAccountBalance_geq_500':    "Categorical variable, where 1 indicates that the client has a savings account balance greater than or equal to 500 euros, and 0 indicates that the client does not have a savings account balance greater than or equal to 500 euros.",
        'MissedPayments':                   "Categorical variable, where 1 indicates that the client has missed payments, and 0 indicates that the client has not missed payments.",
        'NoCurrentLoan':                    "Categorical variable, where 1 indicates that the client has no current loan, and 0 indicates that the client has a current loan.",
        'CriticalAccountOrLoansElsewhere':  "Categorical variable, where 1 indicates that the client has a critical account or loans elsewhere, and 0 indicates that the client does not have a critical account or loans elsewhere.",
        'OtherLoansAtBank':                 "Categorical variable, where 1 indicates that the client has other loans at a bank, and 0 indicates that the client does not have other loans at a bank.",
        'OtherLoansAtStore':                "Categorical variable, where 1 indicates that the client has other loans at a store, and 0 indicates that the client does not have other loans at a store.",
        'HasCoapplicant':                   "Categorical variable, where 1 indicates that the client has a coapplicant, and 0 indicates that the client does not have a coapplicant.",
        'HasGuarantor':                     "Categorical variable, where 1 indicates that the client has a guarantor, and 0 indicates that the client does not have a guarantor.",
        'OwnsHouse':                        "Categorical variable, where 1 indicates that the client owns their house, and 0 indicates that the client does not own their house.",
        'RentsHouse':                       "Categorical variable, where 1 indicates that the client rents their house, and 0 indicates that the client does not rent their house.",
        'Unemployed':                       "Categorical variable, where 1 indicates that the client is unemployed, and 0 indicates that the client is employed.",
        'YearsAtCurrentJob_lt_1':           "Categorical variable, where 1 indicates that the client has been at their current job for less than 1 year, and 0 indicates that the client has been at their current job for 1 year or more.",
        'YearsAtCurrentJob_geq_4':          "Categorical variable, where 1 indicates that the client has been at their current job for 4 years or more, and 0 indicates that the client has been at their current job for less than 4 years.",
        'JobClassIsSkilled':                "Categorical variable, where 1 indicates that the client's job class is skilled, and 0 indicates that the client's job class is not skilled."
    }[feature_variable_name]

@tool
def get_shap_results(client_index):
    """
    Gets the SHAP values for each of the features for a given client index. The results are of the form 
    {'client_index': [CLIENT_INDEX], 'shap_results': [{'feature': [FEATURE], 'shap_value': [SHAP_VALUE]}, ...]}.
    Limit your analysis to the THREE most relevant features for the client.
    NEVER use the term 'SHAP' in your response.
    Use this function to answer questions about how the model made a decision for a specific client, or about 
    how relevant each feature was for the final decision.

    client_index: The index of the client for which to get SHAP results.
    """

    print("get_shap_results() was called")

    url = f"{EXPLANATION_API_URL}/shap?client_index={client_index}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "API request failed", "status_code": response.status_code}

@tool
def get_pdp_results(feature_name):
    """
    Gets the results of Partial Dependence for a given feature name. The results are of the form 
    {'feature_name': [FEATURE_NAME], 'pdp_results': [{'feature_value': [FEATURE_VALUE], 'average_target_value': [AVERAGE_TARGET_VALUE]}, ...]}.

    Use this function to answer questions about how a specific feature impacts the model's decision. NEVER use the term 'pdp' or 'Partial Dependence' in your response.
    These results should be interpreted by analyzing each range of feature values and their corresponding impact on the average target value.
    ALWAYS keep in mind that these values do not take into account the contributions of other features, and should not be used alone as a basis for making decisions.

    feature_name: The name of the feature for which to get Partial Dependence results.
    """

    print("get_pdp_results() was called")

    url = f"{EXPLANATION_API_URL}/pdp?feature_name={feature_name}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "API request failed", "status_code": response.status_code}

@tool
def get_counterfactual(client_index):
    """
    Gets the counterfactual explanations for a given client index. Counterfactual explanations are the minimal changes to the client's features that would change the model's decision.
    Use this function to answer questions about what actions a client could take to change the model's decision.
    The results are a dict containing the original client's features and up to five counterfactuals, identified by CXY, where XY are numbers;
    The dict is of the form {'Orig': [ORIGINAL_FEATURES], 'CXX': [COUNTERFACTUAL_FEATURES], 'CXY': [COUNTERFACTUAL_FETURES]}.
    The counter factuals contain the values to which the features should be changed to change the model's decision. When
    a counterfactual contains a feature that isn't chagens, it is marked with a -1 value.

    MAKE SURE that client_index corresponds to a client that was classified as a bad customer.

    client_index: The index of the client for which to get counterfactual explanations.
    """

    print("get_counterfactual() was called")

    url = f"{EXPLANATION_API_URL}/counterfactual?client_index={client_index}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "API request failed", "status_code": response.status_code}

available_tools = {
    "get_client_info": get_client_info,
    "get_feature_description": get_feature_description,
    "get_pdp_results": get_pdp_results,
    "get_shap_results": get_shap_results,
    "get_counterfactual": get_counterfactual
} # maps function names to their implementations
tools = [get_client_info, get_feature_description, get_pdp_results, get_shap_results, get_counterfactual] # list of tools that the LLM can call

llm_with_tools = llm.bind_tools(tools)


def run_conversation(user_prompt):
    '''
    :return: Tuple of the final response from the LLM and a list of dictionaries containing the name of the tool called, the arguments passed to the tool, and the response from the tool.
    '''

    tool_call_log = []

    # Based off of this example (https://github.com/groq/groq-api-cookbook/blob/main/tutorials/function-calling-101-ecommerce/Function-Calling-101-Ecommerce.ipynb) on tool calling with Groq.
    # Changes allow for multiple tool calls that depend on each other (non-parallel) to be made at once.
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_prompt)]
    
    tool_call_identified = True
    while tool_call_identified:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        for tool_call in ai_msg.tool_calls:
            selected_tool = available_tools[tool_call["name"]]
            # print(f"Calling tool {tool_call['name']} with args {tool_call['args']}")
            tool_output = selected_tool.invoke(tool_call["args"])
            # print(f"Tool output: {tool_output}")
            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
            tool_call_log.append({"name": tool_call["name"], "args": tool_call["args"], "response": tool_output})
        if len(ai_msg.tool_calls) == 0:
            tool_call_identified = False

    return ai_msg.content, tool_call_log


# gradio UI definition
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Free Chat"):
            with gr.Row():
                with gr.Column():
                    user_prompt = gr.Textbox(lines=5, label="User Prompt")
                    submit = gr.Button(value="Submit")
                with gr.Column():
                    output = gr.Textbox(lines=5, label="Output")
                    toolcalls_output = gr.JSON(label="Tool Calls")
            submit.click(run_conversation, user_prompt, outputs=[output, toolcalls_output])
        
        with gr.TabItem("Specific Features"):
            with gr.Row():
                with gr.Column():
                    feature_name = gr.Dropdown(
                        ['Gender', 'ForeignWorker', 'Single', 'Age',
                        'LoanDuration', 'PurposeOfLoan', 'LoanAmount',
                        'LoanRateAsPercentOfIncome', 'YearsAtCurrentHome',
                        'NumberOfOtherLoansAtBank', 'NumberOfLiableIndividuals', 'HasTelephone',
                        'CheckingAccountBalance_geq_0', 'CheckingAccountBalance_geq_200',
                        'SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500',
                        'MissedPayments', 'NoCurrentLoan', 'CriticalAccountOrLoansElsewhere',
                        'OtherLoansAtBank', 'OtherLoansAtStore', 'HasCoapplicant',
                        'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed',
                        'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4',
                        'JobClassIsSkilled'], label="Feature", info="Analyze how each feature relates to the target variable"
                    ), 
                    submit = gr.Button(value="Submit")
                with gr.Column():
                    output = gr.Textbox(lines=5, label="Output")
                    toolcalls_output = gr.JSON(label="Tool Calls")
            submit.click(lambda feature_name: run_conversation(f"How does the {feature_name} variable affect the approval of my loan? Think step by step before answering the question"), feature_name, outputs=[output, toolcalls_output])
        
        with gr.TabItem("Specific Client"):
            with gr.Row():
                with gr.Column():
                    client_index = gr.Number(label="Client Index", minimum=0, maximum=999, step=1, info="Choose a client ID between 0 and 999 to analyze their outcome.")
                    submit = gr.Button(value="Submit")
                with gr.Column():
                    output = gr.Textbox(lines=5, label="Output")
                    toolcalls_output = gr.JSON(label="Tool Calls")
            submit.click(lambda client_index: run_conversation(f"I'm client number {client_index}. How was I classified by the system and what made the system classify me as it did? If I was classified as a bad customer, what could I do to change this outcome? Think step by step before answering the question"), client_index, outputs=[output, toolcalls_output])


if __name__ == "__main__":
    demo.launch(share=True)