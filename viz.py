import pandas as pd
import numpy as np
import geopandas as gpd

df_blocks = pd.read_csv('Block_level_karnataka_new.csv')
df_district = pd.read_csv('District_level_karnataka_new.csv')
gdf= gpd.read_file('District/District.shp')

import pickle
with open('mapper.pkl', 'rb') as f:
    mapper = pickle.load(f)

df_blocks['District'].replace(mapper)

gdf = gpd.GeoDataFrame(df_district.merge(gdf[['geometry','KGISDist_1']], right_on='KGISDist_1', left_on='Updated_Name', how='left'), geometry='geometry')



import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)




# Layout of the app
app.layout = html.Div(style={'backgroundColor': '#f2f2f2'}, children=[
    # Top Row
    html.Div([
        # Fixed State (Karnataka)
        html.Div([
            html.Label("State State"),
            dcc.Dropdown(
                id='state-dropdown',
                options=['Karnataka'],
                multi=False,
                value='Karnataka'
            )
        ], className='two columns'),
        
        
        # Dropdown for district
        html.Div([
            html.Label("Select District:"),
            dcc.Dropdown(
                id='district-dropdown',
                options=[{'label': district, 'value': district} for district in sorted(df_district['Updated_Name'].unique())],
                multi=False
            )
        ], className='two columns'),
        
        # Dropdown for block
        html.Div([
            html.Label("Select Block:"),
            dcc.Dropdown(
                id='block-dropdown',
                multi=False
            )
        ], className='two columns'),
        
        # Dropdown for metric
        html.Div([
            html.Label("Select Metric:"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                        'Percentage of Households with Tap Connection',
                        'Perc HH Connection Reported',
                        'Perc HH Connection Certified',
                        'Percentage of Reported Households with Tap Connection',
                        'Percentage of Certified Households with Tap Connection',
                        'Percentage of Certified Households among Reported',
                    
                ],
                multi=False,
                value='Percentage of Households with Tap Connection'
            )
        ], className='two columns', style={'width': '30%', 'display': 'inline-block'}),
        
        # Dropdown for year
        html.Div([
            html.Label("Select Year:"),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': year, 'value': year} for year in df_district['Year'].unique()],
                multi=False,
                value=2023
            )
        ], className='two columns')
    ], className='row'),
    
    # Middle Row for the choropleth map
    html.Div([
        # Left side (choropleth map and bar chart)
        html.Div([
             html.Div([
                dcc.Graph(id='choropleth-map-2021')
            ], className='row', style={'width': '100%', 
                                       'display': 'inline-block', 
                                       'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff'}),
    
            html.Div([
                dcc.Graph(id='choropleth-map')
            ], className='row', style={'width': '100%', 
                                       'display': 'inline-block', 
                                       'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff'}),
           
        ], style={'width': '100%', 'display': 'flex','justify-content':'space-around'}),

        html.Div([
                ## meticsss left ones
            html.Div([
                 html.Div([
                    dcc.Graph(id='all-metrics-bar-chart')
                ], className='row', style={'width': '100%', 
                                           'display': 'inline-block', 
                                           'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff'}),
                html.Div([
                    dcc.Graph(id='all-metrics-bar-chart-block')
                ], className='row', style={'width': '100%', 
                                           'display': 'inline-block', 
                                           'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff'}),
            ], style={'width': '50%', 'display': 'flex','flex-direction':'column','justify-content':'space-around'}),
           
            
            # Right side (continuous bar graph)
            html.Div([
                dcc.Graph(id='continuous-y-chart')
            ], className='row', style={'width': '50%', 
                                       'display': 'inline-block', 
                                       'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff','margin-left': '1%'}),
        ], style={'width': '100%', 'display': 'flex'}),

        #block level long graph
        html.Div([
            dcc.Graph(id='bar-chart')
        ], className='row'),

        html.Div([
            html.Div([
                dcc.Graph(id='multibargraph')
            ], className='row', style={'width': '50%', 
                                       'display': 'inline-block', 
                                       'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff'}),
            html.Div([
                dcc.Graph(id='multibargraph2')
            ], className='row', style={'width': '50%', 
                                       'display': 'inline-block', 
                                       'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff'}),
        ], style={'width': '100%', 'display': 'flex','justify-content':'space-around'}),
    ]),
])

# Callback to update block dropdown based on selected district
@app.callback(
    Output('block-dropdown', 'options'),
    [Input('district-dropdown', 'value')]
)
def update_block_options(selected_district):
    if not selected_district:
        return []
    
    # Filter blocks based on selected district
    available_blocks = df_blocks[df_blocks['Updated_Name'] == selected_district]['Block'].unique()
    return [{'label': block, 'value': block} for block in available_blocks]

# Callback to update choropleth map based on selected options
# Callback to update choropleth map based on selected options
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('metric-dropdown', 'value')]
)
def update_choropleth_map(selected_metric):
    if not (selected_metric or selected_year):
        return px.choropleth()
    
    filtered_data = gdf[gdf['Year'] == 2023]
    filtered_data = filtered_data[['KGISDist_1','geometry',selected_metric]].to_crs('EPSG:4326')
    
    # Create and return the choropleth map using 'metric' column
    fig = px.choropleth(
        filtered_data,
        geojson=filtered_data.geometry,
        locations=filtered_data.index,
        color=selected_metric,
        color_continuous_scale='blues',
        hover_name='KGISDist_1',
        title=f"{selected_metric} ({2023})",
        range_color=[0, 100]
        
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            orientation='h', # Set orientation to 'h' for horizontal color bar
            # title=f'Choropleth Map for {selected_state} - {selected_metric}',
        )
    )
    return fig


@app.callback(
    Output('choropleth-map-2021', 'figure'),
    [Input('metric-dropdown', 'value'),]
)
def update_choropleth_map_2021(selected_metric):
    if not (selected_metric or selected_year):
        return px.choropleth()
    
    filtered_data = gdf[gdf['Year'] == 2021]
    filtered_data = filtered_data[['KGISDist_1','geometry',selected_metric]].to_crs('EPSG:4326')
    
    # Create and return the choropleth map using 'metric' column
    fig = px.choropleth(
        filtered_data,
        geojson=filtered_data.geometry,
        locations=filtered_data.index,
        color=selected_metric,
        color_continuous_scale='blues',
        hover_name='KGISDist_1',
        title=f"{selected_metric} ({2021})",
        range_color=[0, 100]
        
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            orientation='h', # Set orientation to 'h' for horizontal color bar
            # title=f'Choropleth Map for {selected_state} - {selected_metric}',
        )
    )
    return fig

# Callback to update bar graph based on selected options
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('district-dropdown', 'value'),
     Input('block-dropdown', 'value'),
     Input('metric-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_bar_chart(selected_district, selected_block, selected_metric, selected_year):
    if not selected_district:
        return px.bar()  #Empty bar chart
    
    # Filter data based on selected options
    filtered_data = df_blocks[
        (df_blocks['Updated_Name'] == selected_district) &
        (df_blocks['Year'] == selected_year)
    ]
    
    # Create and return the bar chart
    fig = px.bar(
        filtered_data,
        x='Block',
        y=selected_metric,
        title=f"{selected_metric} in {selected_district} ({selected_year})",
        labels={'Block': 'Block Name', selected_metric: selected_metric},
        text=selected_metric,
        height=500
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    return fig




@app.callback(
    Output('all-metrics-bar-chart', 'figure'),
    [Input('district-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_all_metrics_bar_chart(selected_district, selected_year):
    if not selected_district:
        return px.bar()  # Empty bar chart

    # Filter data based on selected options
    filtered_data = df_district[
        (df_district['Updated_Name'] == selected_district) &
        (df_district['Year'] == selected_year)
    ]

    # List of all metrics
    all_metrics = [
        'Percentage of Households with Tap Connection',
        'Perc HH Connection Reported',
        'Perc HH Connection Certified',
        'Percentage of Reported Households with Tap Connection',
        'Percentage of Certified Households with Tap Connection',
        'Percentage of Certified Households among Reported'
    ]

    # Aggregate data for the entire district
    aggregated_data = filtered_data.groupby('Year')[all_metrics].mean().reset_index()

    # Melt the DataFrame to have a long format suitable for a bar chart
    melted_data = pd.melt(
        aggregated_data,
        id_vars=['Year'],
        value_vars=all_metrics,
        var_name='Metric',
        value_name='Value'
    )

    # Create and return the horizontal bar chart with metrics on y-axis and values on x-axis
    fig = px.bar(
        melted_data,
        y='Metric',
        x='Value',
        title=f"All Metrics in {selected_district} ({selected_year})",
        labels={'Value': 'Metric Value'},
        orientation='h',
        height=500,
        text='Value',  # Enable labels on bars
        color_discrete_sequence=['green']  # Set the color to pink
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')  # Format labels as numbers
    return fig




@app.callback(
    Output('all-metrics-bar-chart-block', 'figure'),
    [Input('district-dropdown', 'value'),
     Input('block-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_all_metrics_bar_chart_block(selected_district, selected_block, selected_year):
    if  not selected_block or not selected_year:
        return px.bar()  # Empty bar chart

    # Filter data based on selected options
    filtered_data = df_blocks[
        (df_blocks['Updated_Name'] == selected_district) &
        (df_blocks['Block'] == selected_block) &
        (df_blocks['Year'] == selected_year)
    ]

    # List of all metrics
    all_metrics = [
        'Percentage of Households with Tap Connection',
        'Perc HH Connection Reported',
        'Perc HH Connection Certified',
        'Percentage of Reported Households with Tap Connection',
        'Percentage of Certified Households with Tap Connection',
        'Percentage of Certified Households among Reported'
    ]

    # Melting the DataFrame
    melted_data = pd.melt(
        filtered_data,
        id_vars=['Block'],
        value_vars=all_metrics,
        var_name='Metric',
        value_name='Metric Value'
    )

    # Create and return the horizontal bar chart
    fig = px.bar(
        melted_data,
        y='Metric',
        x='Metric Value',
        title=f"All Metrics in {selected_district} - {selected_block} ({selected_year})",
        labels={'Metric Value': 'Metric Value'},
        orientation='h',
        height=500,
        text='Metric Value',  # Enable labels on bars
        color_discrete_sequence=['green']  # Set the color to green
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')  # Format labels as numbers
    return fig







@app.callback(
    Output('continuous-y-chart', 'figure'),
    [Input('metric-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_continuous_y_chart(selected_metric, selected_year):
    if not selected_metric or not selected_year:
        return px.line()  # Empty line chart

    # Filter data based on selected options
    filtered_data = df_district[
        (df_district['Year'] == selected_year)
    ].sort_values(by='Percentage of Households with Tap Connection', ascending=True)

    # Create and return the line chart
    fig = px.bar(
        filtered_data,
        x=selected_metric,
        y='District',
        title=f"{selected_metric} in karnataka for ({selected_year})",
        labels={'District': 'District Name', selected_metric: selected_metric},
        height=800,
        text=selected_metric,  # Display values on top of bars
        color_discrete_sequence=['blue'],
    )

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        margin=dict(l=15,),
    )
    fig.update_xaxes(range=[0, 110])
    return fig
    
@app.callback(
    Output('multibargraph', 'figure'),
    [Input('district-dropdown', 'value')]
)
def multibargraph(selected_district):
    if not selected_district:
        return go.Figure()  # Empty figure
    
    # Filter data based on selected district
    filtered_data = df_blocks[df_blocks['Updated_Name'] == selected_district]

    # Sort data by the first variable
    sorted_data = filtered_data.sort_values(by='Percentage of Households with Tap Connection', ascending=True)

    # Create the multi-bar graph
    fig = go.Figure()

    # Add bars for each variable
    for variable in ['Percentage of Reported Households with Tap Connection', 'Percentage of Certified Households with Tap Connection','Percentage of Households with Tap Connection']:
        fig.add_trace(go.Bar(
            x=sorted_data[variable],
            y=sorted_data['Block'],
            name=variable,
            orientation='h',  # Set orientation to horizontal
            text=sorted_data[variable].apply(lambda x: f'{x:.1f}%'),  # Display data labels on the bars
            textposition='outside',
            marker=dict(line=dict(width=0)),
            # showlegend=False, 
        ))

    # Set x-axis limit from 0 to 100
    fig.update_xaxes(range=[0, 105])

    # Update layout to reduce bar gap and group gap
    fig.update_layout(
        # title=f'Tap Connection Statistics for {selected_district}',
        xaxis=dict(title='Percentage'),
        yaxis=dict(title='District Name'),
        barmode='group',
        bargap=0.2,        # Adjust the gap between bars
        bargroupgap=0.1,   # Adjust the gap between bar groups
        height=800,        # Increase height
        legend=dict(x=0.5, y=-0.15, orientation='h')
    )

    return fig

@app.callback(
    Output('multibargraph2', 'figure'),
    [Input('district-dropdown', 'value')]
)
def multibargraph(selected_district):
    if not selected_district:
        return go.Figure()  # Empty figure
    
    # Filter data based on selected district
    filtered_data = df_blocks[df_blocks['Updated_Name'] == selected_district]

    # Sort data by the first variable
    sorted_data = filtered_data.sort_values(by='Percentage of Households with Tap Connection', ascending=True)

    # Create the multi-bar graph
    fig = go.Figure()

    # Add bars for each variable
    for variable in ['Percentage of Certified Households among Reported','Percentage of Households with Tap Connection' ]:
        fig.add_trace(go.Bar(
            x=sorted_data[variable],
            y=sorted_data['Block'],
            name=variable,
            orientation='h',  # Set orientation to horizontal
            text=sorted_data[variable].apply(lambda x: f'{x:.1f}%'),  # Display data labels on the bars
            textposition='outside',
            marker=dict(line=dict(width=0)),
            # showlegend=False, 
        ))

    # Set x-axis limit from 0 to 100
    fig.update_xaxes(range=[0, 105])

    # Update layout to reduce bar gap and group gap
    fig.update_layout(
        # title=f'Tap Connection Statistics for {selected_district}',
        xaxis=dict(title='Percentage'),
        yaxis=dict(title='District Name'),
        barmode='group',
        bargap=0.2,        # Adjust the gap between bars
        bargroupgap=0.1,   # Adjust the gap between bar groups
        height=800,        # Increase height
        legend=dict(x=0.5, y=-0.15, orientation='h')
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True,host="0.0.0.0")
