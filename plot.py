# plot.py
import pandas as pd
import dash
from dash import dcc, html
import plotly.graph_objs as go
import os

# -----------------------------
# Load Data
# -----------------------------
CSV_PATH = "results.csv"  # path to your YOLO results
df = pd.read_csv(CSV_PATH)

# Ensure necessary columns exist
# Overall accuracy (mAP50) and class-wise accuracy are assumed in your CSV
# If class-wise accuracy columns are named differently, update here
class_columns = [
    "metrics/mAP50(B)",  # overall
    "metrics/mAP50_0", "metrics/mAP50_1", "metrics/mAP50_2",
    "metrics/mAP50_3", "metrics/mAP50_4", "metrics/mAP50_5",
    "metrics/mAP50_6", "metrics/mAP50_7", "metrics/mAP50_8",
    "metrics/mAP50_9"  # ambulance class
]

for col in class_columns:
    if col not in df.columns:
        df[col] = 0.0  # fallback if column missing

# -----------------------------
# Dash App Setup
# -----------------------------
app = dash.Dash(__name__)
app.title = "YOLO Training Dashboard"

# -----------------------------
# Figures
# -----------------------------
# Loss Curves
loss_fig = go.Figure()
loss_fig.add_trace(go.Scatter(x=df["epoch"], y=df["train/box_loss"], mode='lines+markers', name='Box Loss'))
loss_fig.add_trace(go.Scatter(x=df["epoch"], y=df["train/cls_loss"], mode='lines+markers', name='Cls Loss'))
loss_fig.add_trace(go.Scatter(x=df["epoch"], y=df["train/dfl_loss"], mode='lines+markers', name='DFL Loss'))
loss_fig.update_layout(title="Training Losses per Epoch", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_dark")

# Overall Accuracy Curve
acc_fig = go.Figure()
acc_fig.add_trace(go.Scatter(x=df["epoch"], y=df["metrics/mAP50(B)"], mode='lines+markers', name='Overall mAP50'))
acc_fig.update_layout(title="Overall Accuracy (mAP50)", xaxis_title="Epoch", yaxis_title="Accuracy", yaxis=dict(range=[0,1]), template="plotly_dark")

# Class-wise Accuracy
class_acc_fig = go.Figure()
for i, col in enumerate(class_columns[1:]):  # skip overall
    name = "Class " + str(i)
    if i == 9:
        name += " (Ambulance)"
    class_acc_fig.add_trace(go.Scatter(x=df["epoch"], y=df[col], mode='lines', name=name))
class_acc_fig.update_layout(title="Class-wise Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy", yaxis=dict(range=[0,1]), template="plotly_dark")

# -----------------------------
# Current Stats Meters
# -----------------------------
current_epoch = df["epoch"].iloc[-1]
current_map50 = df["metrics/mAP50(B)"].iloc[-1]
current_ambulance_acc = df["metrics/mAP50_9"].iloc[-1]  # ambulance class

def gauge(title, value):
    return dcc.Graph(
        figure=go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            gauge={'axis': {'range': [0, 1]}}
        )),
        style={'height': '250px'}
    )

# -----------------------------
# Layout
# -----------------------------
app.layout = html.Div(style={'backgroundColor':'#1e1e1e','color':'white','padding':'20px'}, children=[
    html.H1("YOLO Training Dashboard", style={'textAlign':'center'}),
    
    html.Div([
        html.Div([gauge("Current Epoch", current_epoch/90)], style={'width':'32%', 'display':'inline-block'}),
        html.Div([gauge("Current mAP50", current_map50)], style={'width':'32%', 'display':'inline-block'}),
        html.Div([gauge("Ambulance Accuracy", current_ambulance_acc)], style={'width':'32%', 'display':'inline-block'}),
    ], style={'textAlign':'center'}),
    
    html.Div([
        dcc.Graph(figure=loss_fig),
        dcc.Graph(figure=acc_fig),
        dcc.Graph(figure=class_acc_fig)
    ])
])

# -----------------------------
# Run App
# -----------------------------
port = int(os.environ.get("PORT", 8050))
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=port)
