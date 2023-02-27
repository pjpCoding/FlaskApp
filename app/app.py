from flask import Flask, render_template
import pandas as pd
import pickle
import os 
import numpy as np


app = Flask(__name__)

# Get the path to the models folder
models_folder = "../models/"

# Get the path to the models folder
data_folder = "../data/"

# Load the pickled regression model
with open(os.path.join(models_folder, 'xgb_reg.pkl'), 'rb') as f:
    reg_model = pickle.load(f)

# Load the pickled classification model
with open(os.path.join(models_folder, 'xgb_clf.pkl'), 'rb') as f:
    clf_model = pickle.load(f)

# Load the data
data = pd.read_csv(os.path.join(data_folder, 'scenario_1_test_data.csv'), index_col=0).reset_index(drop=True)

# Defining predictor columns
predictors = ['popular_day_Friday', 'popular_day_Monday', 'popular_day_Saturday',
       'popular_day_Sunday', 'popular_day_Thursday', 'popular_day_Tuesday',
       'popular_day_Wednesday', 'cluster_Cluster 0', 'cluster_Cluster 1',
       'cluster_Cluster 2', 'cluster_Cluster 3', 'cluster_Noise',
       'cluster_2_Cluster 0', 'cluster_2_Cluster 1', 'cluster_2_Cluster 2',
       'cluster_2_Cluster 3', 'cluster_2_Cluster 4', 'cluster_2_Noise',
       'phone_brand', 'device_model', 'number_of_events', 'popular_hour',
       'median_lat', 'median_long', 'total_apps_installed',
       'total_apps_active', 'n_categories', 'avg_events_hour',
       'popular_category', 'avg_events_day', 'percentage_of_active_apps']

# Define a route to display the predictions
@app.route('/')
def home():
    # Make predictions
    gender_pred = clf_model.predict_proba(data[predictors])[:, 1]
    age_pred = reg_model.predict(data[predictors])
    
    # Create a table of the predictions
    table = pd.DataFrame({'Gender_prob': pd.Series([round(i, 2) for i in gender_pred]),
                           'Age': pd.Series([round(i) for i in age_pred])})

    #Ordering probabilities to obtain deciles
    sorted_probs = np.sort(table['Gender_prob'])

    # Find the 10th, 20th, 30th, and so on percentiles to divide the sorted probabilities into deciles
    decile_cutoffs = np.percentile(sorted_probs, np.arange(10, 101, 10))

    # Map the probabilities to class labels based on the deciles
    table['Gender'] = "Undefined (deciles 4, 5 and 6)"
    
    # -3 position will be the 8th decile, so each value higher than that one will be MALE
    table.loc[table['Gender_prob'] >= decile_cutoffs[-3], 'Gender'] = "Male"

    # 2 position will be the 3th decile so each value lower than that will be Female
    table.loc[table['Gender_prob'] <= decile_cutoffs[2], 'Gender'] = "Female"

    #Ordering
    table = table[["Gender_prob", "Gender", "Age"]]

    #Mapping campaigns
    table["Gender Based Campaing"] = "None"

    table.loc[table["Gender"] == "Female", "Gender Based Campaing"] = "Campaign 1 - Campaing 2"
    table.loc[table["Gender"] == "Male", "Gender Based Campaing"] = "Campaign 3"


    table["Age Based Campaing"] = "None"

    table.loc[table["Age"].isin(np.arange(0,25,1)), "Age Based Campaing"] = "Campaing 4"
    table.loc[table["Age"].isin(np.arange(25,33,1)), "Age Based Campaing"] = "Campaing 5"
    table.loc[table["Age"] >= 32, "Age Based Campaing"] = "Campaing 6"


    table = pd.concat([data[["device_id"]], table], axis=1)
    

    # Creating 50 sample devices to show in the flask app
    sample_50 = table.sample(50).reset_index(drop=True)

    # Render the template with the table
    return render_template('table.html', table=sample_50.to_html(index=True))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)

