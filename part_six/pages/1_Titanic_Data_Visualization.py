import streamlit as st
import pandas as pd
import numpy as np
from resources import get_train_data
from datetime import timedelta
from statistics import mean
import plotly.graph_objects as go  # pip install plotly==5.15.0
from plotly.subplots import make_subplots

@st.cache_data(ttl=timedelta(hours=24))
def get_viz_data():
    data = get_train_data()
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    age_bins = [0,10,20,30,40,50,60,70,80,90]
    data['Age_Binned'] = pd.cut(data['Age'], age_bins, ordered=True)

    fare_bins = [0,25,50,75,100,125,150]
    data['Fare_Binned'] = pd.cut(data['Fare'], fare_bins)

    return data

data = get_viz_data()

metrics = [x for x in data.columns.tolist() if '_Binned' not in x][2:]
metric = st.selectbox('Select Metric:', metrics)

data = data.sort_values(by=metric)

#table = st.table(data.head())

def prepare_data_for_visualization(data, metric):
    data = data.sort_values(by=metric)
    if metric in ['Age', 'Fare']:
        metric = metric+'_Binned'
    data = data[[metric, 'Survived']].dropna()
    x = list(set(data[metric]))
    y = [mean(data[data[metric] == j]['Survived']) for j in x]
    z = [len(data[data[metric] == j]) for j in x]

    if metric in ['Age_Binned', 'Fare_Binned']:
        x = [i.left for i in x]

    x = sorted(x)
    return {x: [y,z] for x, y, z in zip(x,y,z)}

data_dict = prepare_data_for_visualization(data=data, metric=metric)
x = data_dict.keys()


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
            go.Bar(
                x=list(x),
                y=[data_dict[key][1] for key in list(x)],
                name="Num. of Observations",
                marker_color="gray",
                opacity=0.15,
                width=0.6
            ),
            secondary_y=True
)

fig.add_trace(
            go.Scatter(
                x=list(x),
                y=[data_dict[key][0] for key in list(x)],
                name=f"Survival by {metric}",
                line=dict(color="#007459"),
                #mode="lines",
            ),
            secondary_y=False,
        )

fig.update_yaxes(rangemode='tozero')
fig.update_yaxes(showgrid=False, secondary_y=True)
fig.update_xaxes(type='category')

st.plotly_chart(fig)

code = """
import streamlit as st
import pandas as pd
import numpy as np
from resources import get_train_data
from datetime import timedelta
from statistics import mean
import plotly.graph_objects as go  # pip install plotly==5.15.0
from plotly.subplots import make_subplots

@st.cache_data(ttl=timedelta(hours=24))
def get_viz_data():
    data = get_train_data()
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    age_bins = [0,10,20,30,40,50,60,70,80,90]
    data['Age_Binned'] = pd.cut(data['Age'], age_bins, ordered=True)

    fare_bins = [0,25,50,75,100,125,150]
    data['Fare_Binned'] = pd.cut(data['Fare'], fare_bins)

    return data

data = get_viz_data()

metrics = [x for x in data.columns.tolist() if '_Binned' not in x][2:]
metric = st.selectbox('Select Metric:', metrics)

data = data.sort_values(by=metric)

table = st.table(data.head())

def prepare_data_for_visualization(data, metric):
    data = data.sort_values(by=metric)
    if metric in ['Age', 'Fare']:
        metric = metric+'_Binned'
    data = data[[metric, 'Survived']].dropna()
    x = list(set(data[metric]))
    y = [mean(data[data[metric] == j]['Survived']) for j in x]
    z = [len(data[data[metric] == j]) for j in x]

    if metric in ['Age_Binned', 'Fare_Binned']:
        x = [i.left for i in x]

    x = sorted(x)
    return {x: [y,z] for x, y, z in zip(x,y,z)}

data_dict = prepare_data_for_visualization(data=data, metric=metric)
x = data_dict.keys()

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
            go.Bar(
                x=list(x),
                y=[data_dict[key][1] for key in list(x)],
                name="Num. of Observations",
                marker_color="gray",
                opacity=0.15,
                width=0.6
            ),
            secondary_y=True
)

fig.add_trace(
            go.Scatter(
                x=list(x),
                y=[data_dict[key][0] for key in list(x)],
                name=f"Survival by {metric}",
                line=dict(color="#007459"),
                #mode="lines",
            ),
            secondary_y=False,
        )

fig.update_yaxes(rangemode='tozero')
fig.update_yaxes(showgrid=False, secondary_y=True)
fig.update_xaxes(type='category')

st.plotly_chart(fig)
"""

st.code(code, language="python", line_numbers=True)