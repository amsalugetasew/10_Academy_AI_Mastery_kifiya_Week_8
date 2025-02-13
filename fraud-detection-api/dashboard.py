from flask import Flask, jsonify
import dash
from dash import dcc, html
import dash.dependencies as dd
import pandas as pd
import plotly.express as px

# Initialize Flask app
app = Flask(__name__)

# Load datasets
df1 = pd.read_csv('Fraud_Data.csv')
df2 = pd.read_csv('IpAddress_to_Country.csv')

# Sort dataframes before merging
df2 = df2.sort_values("lower_bound_ip_address")
df1 = df1.sort_values("ip_address")

# Merge fraud data with country data
df1 = pd.merge_asof(df1, df2, left_on="ip_address", right_on="lower_bound_ip_address")

# Drop unnecessary columns
df1 = df1.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"])

# Convert datetime columns
df1['signup_time'] = pd.to_datetime(df1['signup_time'])
df1['purchase_time'] = pd.to_datetime(df1['purchase_time'])
df1['day'] = df1['purchase_time'].dt.date  # Extract only date
df1['hour_of_day'] = df1['purchase_time'].dt.hour

# Fraud statistics API endpoint
@app.route('/fraud_summary')
def fraud_summary():
    total_transactions = len(df1)
    fraud_cases = df1['class'].sum()
    fraud_percentage = round((fraud_cases / total_transactions) * 100, 2)

    return jsonify({
        "total_transactions": int(total_transactions),
        "fraud_cases": int(fraud_cases),
        "fraud_percentage": float(fraud_percentage)
    })

# Initialize Dash app
dash_app = dash.Dash(__name__, server=app, routes_pathname_prefix='/dashboard/')

# Fraud trends over time
fraud_trend = df1.groupby('day')['class'].sum().reset_index()
fig_fraud_trend = px.line(fraud_trend, x='day', y='class', title='Fraud Cases Over Time')

# Fraud by country
fraud_by_country = df1[df1['class'] == 1]['country'].value_counts().reset_index()
fraud_by_country.columns = ['country', 'count']
fig_fraud_country = px.choropleth(fraud_by_country, locations='country', locationmode='country names',
                                  color='count', title='Fraud Cases by Country', color_continuous_scale='Reds')

# Fraud by device and browser
fraud_by_device = df1[df1['class'] == 1]['device_id'].value_counts().reset_index()
fraud_by_device.columns = ['device_id', 'count']
fig_fraud_device = px.bar(fraud_by_device, x='device_id', y='count', title='Fraud Cases by Device')

fraud_by_browser = df1[df1['class'] == 1]['browser'].value_counts().reset_index()
fraud_by_browser.columns = ['browser', 'count']
fig_fraud_browser = px.bar(fraud_by_browser, x='browser', y='count', title='Fraud Cases by Browser')

# Dashboard layout
dash_app.layout = html.Div(children=[
    html.H1("Fraud Detection Dashboard", style={'text-align': 'center'}),

    html.Div([
        html.Div([
            html.H3("Total Transactions"),
            html.P(id='total-transactions', style={'fontSize': '20px'})
        ], className="summary-box"),

        html.Div([
            html.H3("Fraud Cases"),
            html.P(id='fraud-cases', style={'fontSize': '20px'})
        ], className="summary-box"),

        html.Div([
            html.H3("Fraud Percentage"),
            html.P(id='fraud-percentage', style={'fontSize': '20px'})
        ], className="summary-box"),
    ], style={'display': 'flex', 'justify-content': 'space-around'}),

    dcc.Graph(figure=fig_fraud_trend),
    dcc.Graph(figure=fig_fraud_country),
    dcc.Graph(figure=fig_fraud_device),
    dcc.Graph(figure=fig_fraud_browser),
])

# Callback to update fraud summary stats
@dash_app.callback(
    [dd.Output('total-transactions', 'children'),
     dd.Output('fraud-cases', 'children'),
     dd.Output('fraud-percentage', 'children')],
    [dd.Input('total-transactions', 'id')]  # Just a placeholder trigger
)
def update_summary(_):
    response = fraud_summary().json
    return response["total_transactions"], response["fraud_cases"], f"{response['fraud_percentage']}%"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
