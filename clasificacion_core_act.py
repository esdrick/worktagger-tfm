import logging
import pandas as pd
import numpy as np

from random import randrange

from skllm import MultiLabelZeroShotGPTClassifier
from skllm import ZeroShotGPTClassifier
from skllm.config import SKLLMConfig
from skllm.datasets import get_multilabel_classification_dataset
from skllm.preprocessing import GPTSummarizer
import streamlit as st
import io

from core_act import load_activities
from heuristic_rules import HEURISTIC_RULES

def apply_heuristic(app, title):
    app = str(app).lower()
    title = str(title).lower()
    for rule in HEURISTIC_RULES:
        for keyword in rule["keywords"]:
            if keyword in app or keyword in title:
                logging.debug(f"[DEBUG] Matched keyword '{keyword}' -> Subactivity: {rule['subactivity']}, Activity: {rule['activity']}")
                return rule["subactivity"], rule["activity"]
    return "Unclassified", None


def simple_load_file(loaded_file=None, url_link=None, default_classification="No work-related", dayfirst=False):
    if loaded_file is not None:
        uploaded_file = io.BytesIO(loaded_file.read())
        uploaded_file.seek(0)
    else:
        uploaded_file = url_link

    df = pd.read_csv(uploaded_file,sep=";")

    if "Subactivity" not in df.columns or df["Subactivity"].isna().all():
        df["Subactivity"], df["Activity"] = zip(*df.apply(lambda row: apply_heuristic(row.get("App", ""), row.get("Title", "")), axis=1))

    df['Begin'] = pd.to_datetime(df['Begin'], dayfirst=dayfirst, errors='coerce')
    df['End'] = pd.to_datetime(df['End'], dayfirst=dayfirst, errors='coerce')

    df = df.dropna(subset=['Begin', 'End'])

    if "Activity" in df.columns:
        if "Subactivity" in df.columns:
            result = df
        else:
            dicc_core, dicc_subact, dicc_map_subact, dicc_core_color = load_activities()
            df["Subactivity"] = df["Activity"]
            df["Activity"] = df["Subactivity"].map(dicc_map_subact)
            result = df
    elif "Zero_shot_classification" in df.columns:
        df["Activity"] = df["Zero_shot_classification"]
        result = df
    else: # Tockler import
        #Adds a new column that specifies whether the type of recording is work or computer break.
        df['Type'] = "Computer work"
        # Set 'Type' to 'NATIVE - NO_TITLE' where 'Title' is 'NATIVE - NO_TITLE'
        df['Type'] = np.where(df['Title'].str.contains('NO_TITLE'), 'NO_TITLE', df['Type'])
        df['Activity'] = default_classification

        result = df[df['Type'] == 'Computer work'].copy()

    if "Merged_titles" not in result.columns:
        result['Merged_titles'] = df['App'] +" - "+ df["Title"]

    if "Duration" not in result.columns:
        result['Duration'] = (result['End'] - result['Begin'])/pd.Timedelta('1s')

    if "Subactivity" not in result.columns:
        result["Subactivity"] = "Unspecified " + result['Activity']

    if "Case" not in result.columns:
        result["Case"] = None

    if "App" not in result.columns:
        splitted = df['Merged_titles'].str.split(' - ', n=1, expand=True)
        result["App"] = splitted[0]

    logging.info(result.columns)

    return result


def classify(df, openai_key=None, openai_org=None):
    groups = (df['End'] != df['Begin'].shift(1)).cumsum()
    groups.name = "groups"
    merged_titles = df['Merged_titles'].astype(str).groupby(groups).apply(lambda x: ';'.join(x))
    X = merged_titles.tolist()
    labels = gpt_predict_labels(X, openai_key, openai_org)
    result = pd.merge(left=groups, left_on="groups", right=pd.Series(labels, name="labels", index=pd.RangeIndex(start=1, stop=len(labels)+1)), right_index=True)
    return result['labels']


def prepare_for_classification(df):
    #We add new rows when there is a gap in the time between two rows, and add the rows to the data, with the category "Computer break".
    new_rows = []
    for i in range(len(df) - 1):
        current_end_time = df.loc[i, 'End']
        next_start_time = df.loc[i + 1, 'Begin']
        if current_end_time != next_start_time:
            new_row = {'App': "n.a.", 'Type': "n.a.", 'Title': "Computer break", 'Begin': current_end_time, 'End': next_start_time, 'Duration': next_start_time - current_end_time, 'Type': "Computer break"}
            new_rows.append(new_row)

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    #Merge all titles between computer breaks.
    df_merged_titles = df

    # Sort the DataFrame by 'Begin' to ensure the data is in chronological order
    df_merged_titles = df.sort_values(by='Begin')

    # Add a new column 'Previous_App' to represent the app in the previous row
    df_merged_titles['Previous_App'] = df_merged_titles['App'].shift(1)

    # Group by 'Type' and create a new column 'Group' to identify consecutive rows with the same 'Type'
    df_merged_titles['Group'] = (df_merged_titles['Type'] != df_merged_titles['Type'].shift(1)).cumsum()

    # Define a custom aggregation function for the 'agg' method
    def custom_aggregation(group):
        return pd.Series({
            'Merged_titles': ';'.join(map(str, group['Merged_titles'])),
            'Begin': group['Begin'].iloc[0],
            'End': group['End'].iloc[-1],
            'App': group['App'].iloc[0],
            'Type': group['Type'].iloc[0],
            'Duration': group['Duration'].sum()
        })

    # Group by 'Type' and 'Group', and apply the custom aggregation function
    result_df = df_merged_titles.groupby(['Type', 'Group']).apply(custom_aggregation).reset_index(drop=True)

    # Drop the temporary 'Group' column if it exists
    if 'Group' in result_df.columns:
        result_df.drop(columns=['Group'], inplace=True)

    # Sort the resulting DataFrame by the 'Begin' column
    result_df.sort_values(by='Begin', inplace=True)

    # Reset the index after sorting
    result_df.reset_index(drop=True, inplace=True)

    #Add a column with the most occurring title.

    # Define a custom function to find the most occurring title in a semicolon-separated string
    def find_most_occurring_title(merged_titles):
        titles = merged_titles.split(';')
        title_counts = pd.Series(titles).value_counts()
        most_occuring_title = title_counts.idxmax()
        return most_occuring_title

    # Apply the custom function to each row in the DataFrame and create a new column
    result_df['Most_occuring_title'] = result_df['Merged_titles'].apply(find_most_occurring_title)

    #Filter the dataframe to only contain rows where type is computer work, because I only want to classify those rows.

    filtered_df = result_df[result_df['Type'] == 'Computer work'].copy()

    return filtered_df



def load_uploaded_file(loaded_file):
    ficheros_subidos = io.BytesIO(loaded_file.read())
    ficheros_subidos.seek(0)

    df = pd.read_csv(ficheros_subidos,sep=";")
    #Concatenate the app and title.
    df['Title'] = df['App'] +" - "+ df["Title"]
    #Adds a new column that specifies whether the type of recording is work or computer break.
    df['Type'] = "Computer work"
    # Set 'Type' to 'NATIVE - NO_TITLE' where 'Title' is 'NATIVE - NO_TITLE'
    df['Type'] = np.where(df['Title'].str.contains('NO_TITLE'), 'NO_TITLE', df['Type'])

    #We add new rows when there is a gap in the time between two rows, and add the rows to the data, with the category "Computer break".
    new_rows = []
    for i in range(len(df) - 1):
        current_end_time = df.loc[i, 'End']
        next_start_time = df.loc[i + 1, 'Begin']
        if current_end_time != next_start_time:
            new_row = {'App': "n.a.", 'Type': "n.a.", 'Title': "Computer break", 'Begin': current_end_time, 'End': next_start_time, 'Type': "Computer break"}
            new_rows.append(new_row)

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    #Add a column for the duration between the start and end time and calculate the total duration of recorded data.

    df['Begin'] = pd.to_datetime(df['Begin'], errors='coerce')
    df['End'] = pd.to_datetime(df['End'], errors='coerce')
    df["Duration"] = df['End'] - df['Begin']

    #Merge all titles between computer breaks.
    df_merged_titles = df

    # Sort the DataFrame by 'Begin' to ensure the data is in chronological order
    df_merged_titles = df.sort_values(by='Begin')

    # Add a new column 'Previous_App' to represent the app in the previous row
    df_merged_titles['Previous_App'] = df_merged_titles['App'].shift(1)

    # Create a new column 'Merged_title' and initialize it with 'Title' column values
    df_merged_titles['Merged_titles'] = df_merged_titles['Title']

    # Group by 'Type' and create a new column 'Group' to identify consecutive rows with the same 'Type'
    df_merged_titles['Group'] = (df_merged_titles['Type'] != df_merged_titles['Type'].shift(1)).cumsum()

    # Define a custom aggregation function for the 'agg' method
    def custom_aggregation(group):
        return pd.Series({
            'Merged_titles': ';'.join(map(str, group['Merged_titles'])),
            'Begin': group['Begin'].iloc[0],
            'End': group['End'].iloc[-1],
            'App': group['App'].iloc[0],
            'Type': group['Type'].iloc[0],
            'Duration': group['Duration'].sum()  # Assuming you want to sum the durations
            # Add more columns as needed
        })

    # Group by 'Type' and 'Group', and apply the custom aggregation function
    result_df = df_merged_titles.groupby(['Type', 'Group']).apply(custom_aggregation).reset_index(drop=True)

    # Drop the temporary 'Group' column if it exists
    if 'Group' in result_df.columns:
        result_df.drop(columns=['Group'], inplace=True)

    # Sort the resulting DataFrame by the 'Begin' column
    result_df.sort_values(by='Begin', inplace=True)

    # Reset the index after sorting
    result_df.reset_index(drop=True, inplace=True)

    #Add a column with the most occurring title.

    # Define a custom function to find the most occurring title in a semicolon-separated string
    def find_most_occurring_title(merged_titles):
        titles = merged_titles.split(';')
        title_counts = pd.Series(titles).value_counts()
        most_occuring_title = title_counts.idxmax()
        return most_occuring_title

    # Apply the custom function to each row in the DataFrame and create a new column
    result_df['Most_occuring_title'] = result_df['Merged_titles'].apply(find_most_occurring_title)

    #Filter the dataframe to only contain rows where type is computer work, because I only want to classify those rows.

    filtered_df = result_df[result_df['Type'] == 'Computer work'].copy()

    return filtered_df


def gpt_classification(filtered_df, openai_key=None, openai_org=None):

    #Apply Zero-Shot Text Classification.

    # Assuming result_df is a DataFrame with a column "Merged_titles"
    X = filtered_df["Merged_titles"].tolist()

    labels = gpt_predict_labels(X, openai_key, openai_org)
    # Add the predicted labels to a new column
    filtered_df["Zero_shot_classification"] = labels

    return filtered_df


def gpt_predict_labels_fake(X, openai_key=None, openai_org=None):
    core_activities = [
        "Faculty plan/capacity group plan",
        "Management of education and research",
        "Human Resources policy",
        "Organizational matters",
        "Programme development" ,
        "Acquisition of contract teaching and research" ,
        "Accountability for contract teaching and research" ,
        "Advancing/communicating scientific knowledge and insight",
        "Working groups and committees",
        "Contribution to the research group or lab",
        "Organization of (series of) events",
        "Provision of education",
        "Student supervision" ,
        "PhD candidates" ,
        "Education development" ,
        "Testing" ,
        "Education evaluation" ,
        "Education coordination" ,
        "Research development" ,
        "Assessment of research" ,
        "Execution of research" ,
        "Publication of research" ,
        "Research coordination" ,
        "Research proposal" ,
        "Research plan" ,
        "Performing research" ,
        "Doctoral thesis"
    ]

    return [core_activities[randrange(len(core_activities))] for x in X]





def gpt_predict_labels(X, openai_key=None, openai_org=None):
    from core_act import load_activities
    dicc_core, dicc_subact, dicc_map_subact, dicc_core_color = load_activities()
    core_activities = list(set(dicc_map_subact.values()))

    prompt_template = (
        "You are a productivity assistant. Your task is to classify user computer activity based only on the application name and window title.\n\n"
        "Below are examples:\n"
        "Example 1:\n"
        "- WhatsApp - No title\n"
        "→ Label: Communication\n\n"
        "Example 2:\n"
        "- VSCode - tfm_analysis.py\n"
        "- Chrome - Scholar articles\n"
        "→ Label: Programming\n\n"
        "Example 3:\n"
        "- YouTube - Top 10 Netflix series\n"
        "→ Label: Distraction\n\n"
        "Example 4:\n"
        "- ChatGPT - Research question brainstorming\n"
        "→ Label: AI Consultation\n\n"
        "Now classify the following window titles:\n{}\n\n"
        "Choose only from these labels:\n{}\n"
        "Respond with ONE label only, and nothing else."
    )

    formatted_prompts = [
        prompt_template.format("\n".join(f"- {title}" for title in text.split(";")), ", ".join(core_activities))
        for text in X
    ]

    max_tokens = 4000
    truncated_prompts = [prompt[:max_tokens] for prompt in formatted_prompts]

    clf = ZeroShotGPTClassifier(openai_model="gpt-4-1106-preview", openai_key=openai_key, openai_org=openai_org)
    clf.fit(truncated_prompts, [core_activities])

    labels = clf.predict(truncated_prompts)

    return labels
