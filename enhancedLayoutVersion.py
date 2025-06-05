import pandas as pd
import dash
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State, ALL
import io
import zipfile
from dash import dcc
import dash_daq as daq
import base64

# Load your data
# ctdis_data = pd.read_csv("Cts Dispensing pattern.csv", index_col=0).apply(pd.to_numeric, errors='coerce').fillna(-1.0)
# assay_data = pd.read_csv("Assay_dispensing.csv", index_col=0)
# sample_data = pd.read_csv("Sample_dispensing.csv", index_col=0)

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

ctdis_data = pd.DataFrame()
assay_data = pd.DataFrame()
sample_data = pd.DataFrame()

@app.callback(
    Output('ct-status', 'children'),
    Input('upload-ct', 'contents'),
    State('upload-ct', 'filename')
)
def upload_ct_data(contents, filename):
    global ctdis_data
    if contents is None: raise dash.exceptions.PreventUpdate
    try:
        decoded = base64.b64decode(contents.split(',')[1])
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        ctdis_data = df.set_index(df.columns[0]).apply(pd.to_numeric, errors='coerce').fillna(-1.0)
        return f"✅ CT data uploaded: {filename}"
    except Exception as e:
        return f"❌ Error: {e}"

@app.callback(
    Output('assay-status', 'children'),
    Input('upload-assay', 'contents'),
    State('upload-assay', 'filename')
)
def upload_assay_data(contents, filename):
    global assay_data
    if contents is None: raise dash.exceptions.PreventUpdate
    try:
        decoded = base64.b64decode(contents.split(',')[1])
        assay_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)
        return f"✅ Assay data uploaded: {filename}"
    except Exception as e:
        return f"❌ Error: {e}"

@app.callback(
    Output('sample-status', 'children'),
    Input('upload-sample', 'contents'),
    State('upload-sample', 'filename')
)
def upload_sample_data(contents, filename):
    global sample_data
    if contents is None: raise dash.exceptions.PreventUpdate
    try:
        decoded = base64.b64decode(contents.split(',')[1])
        sample_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)
        return f"✅ Sample data uploaded: {filename}"
    except Exception as e:
        return f"❌ Error: {e}"

def upload_style():
    return {
        'width': '60%',
        'height': '50px',
        'lineHeight': '50px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '10px',
        'textAlign': 'center',
        'margin-bottom': '10px'
    }

# Define helper functions (corrected)
def create_custom_colorscale(selected_color_ranges, zmin, zmax):
    colorscale = []
    sorted_ranges = sorted(selected_color_ranges.items(), key=lambda x: x[1][0])

    for color, (range_min, range_max) in sorted_ranges:
        norm_min = (range_min - zmin) / (zmax - zmin)
        norm_max = (range_max - zmin) / (zmax - zmin)
        norm_min, norm_max = np.clip([norm_min, norm_max], 0, 1)

        if norm_min >= norm_max:
            continue

        colorscale.append([norm_min, color])
        colorscale.append([norm_max, color])

    if colorscale and colorscale[0][0] > 0:
        colorscale.insert(0, [0, "white"])
    if colorscale and colorscale[-1][0] < 1:
        colorscale.append([1, "black"])

    if not colorscale:
        colorscale = [[0, "white"], [1, "black"]]

    return colorscale

def calculate_color_counts(data, color_ranges):
    counts = {color: 0 for color in color_ranges}
    for value in data.values.flatten():
        for color, (low, high) in color_ranges.items():
            if low <= value <= high:
                counts[color] += 1
                break
    return counts

app.layout = html.Div([
    # Corrected CSS Injection
    dcc.Markdown('''
        <style>
            * { font-family: Arial, sans-serif; box-sizing: border-box; }
            button {
                padding: 8px 14px; margin: 5px 0;
                border: none; border-radius: 5px; font-size: 16px;
                background-color: #007bff; color: white; cursor: pointer;
            }
            button:hover { background-color: #0056b3; }
            .range-container {
                display: flex; gap: 20px; align-items: center;
                padding: 10px; background-color: #fafafa;
                border-radius: 8px; margin-bottom: 15px;
            }
        </style>
    ''', dangerously_allow_html=True),

    # App title (left-aligned)
    html.H1("Plate Monkey Analysis", style={'textAlign': 'left', "font-family": "Arial", 'padding-left': '10px'}),

    # Upload file
    html.Div([
        html.H3("📤 Upload Your Datasets", style={'font-family': 'Arial'}),

        html.Div([
            html.Label("CT Dispensing Data:", style={'font-weight': 'bold'}),
            dcc.Upload(id='upload-ct', children=html.Div(['Drag or Select CT CSV']),
                    style=upload_style(), multiple=False),
            html.Div(id='ct-status', style={'color': 'green'})
        ]),

        html.Div([
            html.Label("Assay Dispensing Data:", style={'font-weight': 'bold'}),
            dcc.Upload(id='upload-assay', children=html.Div(['Drag or Select Assay CSV']),
                    style=upload_style(), multiple=False),
            html.Div(id='assay-status', style={'color': 'green'})
        ]),

        html.Div([
            html.Label("Sample Dispensing Data:", style={'font-weight': 'bold'}),
            dcc.Upload(id='upload-sample', children=html.Div(['Drag or Select Sample CSV']),
                    style=upload_style(), multiple=False),
            html.Div(id='sample-status', style={'color': 'green'})
        ]),
    ], style={'padding': '20px', 'font-family': 'Arial'}),

    # Add Color Range button
    html.Div([
        html.Button('➕ Add Color Range', id='add-range-btn', n_clicks=0),
    ], style={'textAlign': 'left', 'padding-left': '10px', "font-family": "Arial", 'margin-bottom': '20px'}),

    # Dynamic color-range container
    html.Div(id='color-range-container', children=[], style={'padding-left': '10px'}),

    # Update and Download buttons
    html.Div([
        html.Button("📊 Update Heatmap and Charts", id="update-btn"),
        # html.Button("📥 Download All Data (ZIP)", id="download-zip-btn"),
        # dcc.Download(id="download-zip")
    ], style={'textAlign': 'left', 'padding-left': '10px', "font-family": "Arial", 'margin-top': '20px', 'font-size': '16px'}),

    # Graphs layout: heatmap on top, bar + pie side by side
    html.Div([
        dcc.Graph(id="heatmap-plot", style={'width': '100%'}),

        html.Div([
            dcc.Graph(id="bar-chart", style={'width': '48%', 'height': '500px', 'display': 'inline-block', 'padding': '10px'}),
            dcc.Graph(id="pie-chart", style={'width': '48%', 'height': '300px', 'display': 'inline-block', 'padding': '10px'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'flexWrap': 'wrap'})
    ], style={'padding': '80px', 'margin': '0', "font-family": "Arial"})
])

@app.callback(
    Output('color-range-container', 'children'),
    Input('add-range-btn', 'n_clicks'),
    Input({'type': 'remove-range', 'index': ALL}, 'n_clicks'),
    State('color-range-container', 'children'),
    prevent_initial_call=True
)
def modify_color_ranges(add_clicks, remove_clicks, children):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Case 1: Add new range
    if triggered_id == 'add-range-btn':
        index = add_clicks  # Safe since n_clicks is always increasing
        new_child = html.Div([
            html.Div([
                html.Label("Min:", style={'font-weight': 'bold', "font-family": "Arial", 'font-size': '16px'}),
                dcc.Input(
                    id={'type': 'range-min', 'index': index},
                    type='number',
                    placeholder='Min',
                    style={'width': '80px', 'padding': '5px'}
                ),
            ], style={'display': 'flex', 'flexDirection': 'column'}),

            html.Div([
                html.Label("Max:", style={'font-weight': 'bold', "font-family": "Arial", 'font-size': '16px'}),
                dcc.Input(
                    id={'type': 'range-max', 'index': index},
                    type='number',
                    placeholder='Max',
                    style={'width': '80px', 'padding': '5px'}
                ),
            ], style={'display': 'flex', 'flexDirection': 'column'}),

            html.Div([
                html.Label("Pick Color:", style={'font-weight': 'bold', "font-family": "Arial", 'font-size': '16px'}),
                daq.ColorPicker(
                    id={'type': 'range-color', 'index': index},
                    value={'hex': '#119DFF'},
                    size=200,
                ),
                html.Button('Remove',
                            id={'type': 'remove-range', 'index': index},
                            n_clicks=0,
                            style={'margin-top': '10px', 'background-color': 'crimson', 'color': 'white'})
            ], style={
                'display': 'flex', 'flexDirection': 'column', 'align-items': 'center',
                'background-color': '#f5f5f5', 'padding': '10px', 'border-radius': '8px',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),
        ], id={'type': 'color-range-block', 'index': index},
           style={'display': 'inline-block', 'margin-right': '15px', 'margin-bottom': '10px'})

        children.append(new_child)
        return children

    # Case 2: Remove specific range
    elif 'remove-range' in triggered_id:
        remove_index = eval(triggered_id)['index']
        new_children = [
            child for child in children
            if not (isinstance(child, dict) and child.get('props', {}).get('id', {}).get('index') == remove_index)
        ]
        return new_children

    raise dash.exceptions.PreventUpdate

# Main callback for heatmap and charts
@app.callback(
    [Output("heatmap-plot", "figure"),
     Output("bar-chart", "figure"),
     Output("pie-chart", "figure")],
    Input("update-btn", "n_clicks"),
    State({'type': 'range-min', 'index': ALL}, 'value'),
    State({'type': 'range-max', 'index': ALL}, 'value'),
    State({'type': 'range-color', 'index': ALL}, 'value')
)
def update_visualizations(n_clicks, mins, maxes, colors):
    if not mins or not maxes or not colors:
        return go.Figure(), go.Figure(), go.Figure()

    selected_color_ranges = {
        color['hex']: (min_v, max_v)
        for min_v, max_v, color in zip(mins, maxes, colors)
        if None not in (min_v, max_v, color) and min_v < max_v
    }

    if not selected_color_ranges:
        return go.Figure(), go.Figure(), go.Figure()

    # Heatmap logic (your provided logic)
    z = ctdis_data.values
    combined_hover_text = (
        "Assay ID: " + assay_data.astype(str) + "<br>Sample ID: " + sample_data.astype(str)
    ).values

    zmin, zmax = np.nanmin(z), np.nanmax(z)
    colorscale = create_custom_colorscale(selected_color_ranges, zmin, zmax)

    heatmap_fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(i) for i in ctdis_data.columns],
        y=[str(i) for i in ctdis_data.index],
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        hoverinfo='text',
        text=combined_hover_text,
        colorbar=dict(title="Value Range"),
        xgap=1,
        ygap=1
    ))
    heatmap_fig.update_layout(title="Combined Heatmap of CT Dispensing", width=1000, height=700)

    # Bar chart logic
    color_counts = calculate_color_counts(ctdis_data, selected_color_ranges)
    bar_chart_fig = go.Figure(go.Bar(
        x=[f'{color}<br>{selected_color_ranges[color]}' for color in color_counts],
        y=list(color_counts.values()),
        marker_color=list(color_counts.keys())
    ))
    bar_chart_fig.update_layout(
        title="Count of Values in Color Ranges",
        xaxis_title="Color Ranges",
        yaxis_title="Count",
        width=None,  # Let Dash autosize based on container
        height=500,
        margin=dict(l=40, r=40, t=50, b=100),
        bargap=0.2,  # Shrink space between bars
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        xaxis_tickangle=-45  # Rotate labels if needed
    )

    # Pie chart (sample data distribution)
    # sample_counts = sample_data.fillna("NaN").stack().value_counts()
    N = 10
    sample_counts = sample_data.fillna("NaN").stack().value_counts()
    top_sample_counts = sample_counts.head(N)
    others = sample_counts.iloc[N:].sum()
    if others > 0:
        top_sample_counts["Other"] = others

    pie_chart_fig = go.Figure(go.Pie(
    labels=sample_counts.index,
    values=sample_counts.values,
    hole=0.5,
    hoverinfo="label+percent+value",  
    textinfo="none"))

    pie_chart_fig.update_layout(title="Distribution of Sample Types", width=500, height=500
)

    return heatmap_fig, bar_chart_fig, pie_chart_fig

@app.callback(
    Output("download-zip", "data"),
    Input("download-zip-btn", "n_clicks"),
    prevent_initial_call=True
)
def create_zip_file(n_clicks):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zf:
        for file_name in ['Assay_dispensing.csv', 'Sample_dispensing.csv', 'Cts Dispensing pattern.csv']:
            try:
                with open(file_name, 'rb') as f:
                    zf.writestr(file_name, f.read())
            except FileNotFoundError:
                print(f"File {file_name} not found.")

    buffer.seek(0)
    
    return dcc.send_bytes(buffer.getvalue(), filename="plate_monkey_template.zip")

if __name__ == "__main__":
    app.run_server(debug=True, port=4000)