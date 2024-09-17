import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load the SpaceX data into a DataFrame
spacex_df = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv")
# Optionally clean data if necessary
# spacex_df['Launch Site'] = spacex_df['Launch Site'].str.strip()
print(spacex_df.head())
min_playload=spacex_df["Payload Mass (kg)"].min()
max_playload=spacex_df["Payload Mass (kg)"].max()
# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div(children=[
    html.H1('SpaceX Launch Records Dashboard',
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),

    # Dropdown for selecting launch sites
    dcc.Dropdown(id='site-dropdown',
                 options=[
                     {'label': 'All', 'value': 'ALL'},
                     {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                     {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
                     {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                     {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'}
                 ],
                 value='ALL',  # Default value
                 placeholder="Select a Launch Site here",
                 searchable=True),

    html.Br(),
    dcc.RangeSlider(id='payload-slider',
                min=0, max=10000, step=1000,
                marks={0:'0',2500:"2500",5000:"5000",7500:"7500",10000:"10000"
                       },
                value=[min_playload, max_playload]),

    # Graph for displaying success/failure pie chart
    dcc.Graph(id='success-pie-chart'),
    dcc.Graph(id='success-payload-scatter-chart'),
])

# Callback to update the pie chart based on selected launch site
@app.callback(
    [Output(component_id='success-pie-chart', component_property='figure'),
     Output(component_id='success-payload-scatter-chart', component_property='figure')],
    [Input(component_id='site-dropdown', component_property='value'),
     Input(component_id='payload-slider', component_property='value')]
)
def update_charts(entered_site, payload_range):
    # Filter the data based on payload range
    low, high = payload_range
    filtered_df = spacex_df[(spacex_df['Payload Mass (kg)'] >= low) & (spacex_df['Payload Mass (kg)'] <= high)]

    # Pie chart for success/failure
    if entered_site == 'ALL':
        pie_fig = px.pie(filtered_df, values='class',
                         names='Launch Site',
                         title='Success for All Launch Sites')
    else:
        filtered_df = filtered_df[filtered_df['Launch Site'] == entered_site]
        success_fail_counts = filtered_df.groupby('class').size().reset_index(name='count')
        success_fail_counts['class'] = success_fail_counts['class'].replace({0: 'Failure', 1: 'Success'})
        pie_fig = px.pie(success_fail_counts, values='count', names='class',
                         title=f'Success vs Failure for {entered_site}')

    # Scatter plot for Payload vs. Outcome
    scatter_fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class',
                             color='Booster Version Category',
                             title=f'Payload vs. Outcome for {entered_site if entered_site != "ALL" else "entered_site"}',
                             labels={'class': 'Mission Outcome'})

    return pie_fig, scatter_fig
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
