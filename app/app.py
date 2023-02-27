from flask import Flask, render_template
import pandas as pd
import pickle
import os 
import numpy as np

app = Flask(__name__)

# Get the path to the models & data folder in EC2
models_folder = "model/"
data_folder = "data/"

# Load the pickled models from EC2
with open(os.path.join(models_folder, 'XGB_Reg.pkl'), 'rb') as f:
    reg_model = pickle.load(f)
with open(os.path.join(models_folder, 'XGB_Clf.pkl'), 'rb') as f:
    clf_model = pickle.load(f)

# Load the data sample prepared
data = pd.read_csv(os.path.join(data_folder, 'scenario_1_test_data.csv'), index_col=0).reset_index(drop=True)

# Defining predictor columns
predictors =  ['most_frequent_category','device_model','device_brand','num_events_device','total_apps_active','most_common_hour','apps_installed',
              'events_per_day','most_frequent_day_Saturday','most_frequent_day_Monday','most_frequent_day_Thursday','most_frequent_day_Friday','most_frequent_day_Sunday',
              'most_frequent_day_Wednesday','most_frequent_day_Tuesday','most_common_binned_hour_8-12','most_common_binned_hour_4-8','most_common_binned_hour_16-20',
              'most_common_binned_hour_20-24','most_common_binned_hour_12-16','most_common_binned_hour_0-4','label_-1','label_0']

# Define a route to display the predictions
@app.route('/')
def home():
    # Make predictions
    gender_pred = clf_model.predict_proba(data[predictors])[:, 1]
    age_pred = reg_model.predict(data[predictors])
    
    # Create a table of predicted gender and age
    table = pd.DataFrame({'Gender_prob': pd.Series([round(i, 2) for i in gender_pred]),
                           'Age': pd.Series([round(i) for i in age_pred])})

    # Sort the gender probabilities and divide them into deciles
    sorted_probs = np.sort(table['Gender_prob'])
    decile_cutoffs = np.percentile(sorted_probs, np.arange(10, 101, 10))

    # Map gender probabilities to class labels based on deciles
    table['Gender'] = "Undefined (deciles 4, 5 and 6)"
    table.loc[table['Gender_prob'] >= decile_cutoffs[-3], 'Gender'] = "Male"
    table.loc[table['Gender_prob'] <= decile_cutoffs[2], 'Gender'] = "Female"

    # Order columns in the table
    table = table[["Gender_prob", "Gender", "Age"]]

    # Assign advertising campaigns based on gender and age
    table["Gender Based Campaign"] = "None"
    table.loc[table["Gender"] == "Female", "Gender Based Campaign"] = "Campaign 1 - Campaign 2"
    table.loc[table["Gender"] == "Male", "Gender Based Campaign"] = "Campaign 3"
    table["Age Based Campaign"] = "None"
    table.loc[table["Age"].isin(np.arange(0,25,1)), "Age Based Campaign"] = "Campaign 4"
    table.loc[table["Age"].isin(np.arange(25,33,1)), "Age Based Campaign"] = "Campaign 5"
    table.loc[table["Age"] >= 32, "Age Based Campaign"] = "Campaign 6"

    # Concatenate device IDs and predicted features
    table = pd.concat([data[["device_id"]], table], axis=1)
    
    # Select a random sample of 50 devices for display in Flask app
    sample_50 = table.sample(50).reset_index(drop=True)

    # Render the table as an HTML template
    return render_template('table.html', table=sample_50.to_html(index=True))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)


