#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly.express as px


# # Line plots with Plotly Express

# In[2]:


import numpy as np
t = np.linspace(0, 2*np.pi, 100)

fig = px.line(x = t, y = np.cos(t), labels = {"x":"t", "y": "cos(t)"})
fig.show()


# In[3]:


df = px.data.gapminder().query("continent == 'Oceania'")
fig = px.line(df, x = "year", y = "lifeExp", color = "country")
fig.show()


# In[4]:


fig = px.line(df, x = "year", y = "lifeExp", color = "country", markers = True)
fig.show()


# In[5]:


fig = px.line(df, x = "year", y = "lifeExp", color = "country", symbol = "country")
fig.show()


# # Line Plots on Date axes

# In[7]:


df =  px.data.stocks()
fig = px.line(df, x='date', y="GOOG")
fig.show()


# # Data Order in Scatter and Line Charts

# In[9]:


import pandas as pd
df =  pd.DataFrame(dict(
    x = [1, 3, 2, 4],
    y = [1, 2, 3, 4]
))
fig = px.line(df, x="x", y="y", title="Unsorted Input") 
fig.show()

df = df.sort_values(by="x")
fig = px.line(df, x="x", y="y", title="Sorted Input") 
fig.show()


# # Connected Scatterplots

# In[10]:


df =  px.data.gapminder().query("country in ['Canada', 'Botswana']")

fig = px.line(df, x="lifeExp", y="gdpPercap", color="country", text="year")
fig.update_traces(textposition="bottom right")
fig.show()


# # Scatter and line plots with go.Scatter

# # Simple Scatter Plot

# In[12]:


import plotly.graph_objects as go
N =  1000
t = np.linspace(0, 10, 100)
y = np.sin(t)

fig = go.Figure(data=go.Scatter(x=t, y=y, mode='markers'))

fig.show() 


# # Line and Scatter Plots

# In[14]:


#Create random data with numpy
import numpy as np
np.random.seed(1)

N = 100 
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N) + 5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N) - 5

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x =random_x, y=random_y0,
                    mode='markers',
                    name='markers'))
fig.add_trace(go.Scatter(x=random_x, y=random_y1,
                    mode='lines+markers',
                    name='lines+markers'))
fig.add_trace(go.Scatter(x=random_x, y=random_y2,
                    mode='lines',
                    name='lines'))

fig.show()


# # Bubble Scatter Plots

# In[15]:


fig = go.Figure(data=go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 11, 12, 13],
    mode='markers',
    marker=dict(size=[40, 60, 80, 100],
                color=[0, 1, 2, 3])
))

fig.show()


# # Style Scatter Plots

# In[16]:


t = np.linspace(0, 10, 100)

fig = go.Figure()

fig.add_trace(go.Scatter(
x = t, y = np.sin(t),
name = "sin",
mode = "markers",
marker_color = "rgba(152, 0, 0, .8)"))

fig.add_trace(go.Scatter(
    x=t, y=np.cos(t),
    name='cos',
    marker_color='rgba(255, 182, 193, .9)'
))

# Set options common to all traces with fig.update_traces
fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
fig.update_layout(title='Styled Scatter',
                  yaxis_zeroline=False, xaxis_zeroline=False)


fig.show()


# # Data Labels on Hover

# In[17]:


data =  pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv")

fig = go.Figure(data=go.Scatter(x=data['Postal'],
                                y=data['Population'],
                                mode='markers',
                                marker_color=data['Population'],
                                text=data['State'])) # hover text goes here

fig.update_layout(title='Population of USA States')
fig.show()


# # Scatter with a color Dimension

# In[19]:


fig = go.Figure(data=go.Scatter(
    y = np.random.randn(500),
    mode='markers',
    marker=dict(
        size=16,
        color=np.random.randn(500), #set color equal to a variable
        colorscale='Viridis', # one of plotly colorscales
        showscale=True
    )
))

fig.show()


# # Trace Zorder

# In[22]:


import plotly.data as data
df =  data.gapminder()

df_europe = df[df['continent'] == 'Europe']

trace1 = go.Scatter(x=df_europe[df_europe['country'] == 'France']['year'], 
                    y=df_europe[df_europe['country'] == 'France']['lifeExp'], 
                    mode='lines+markers', 
                    #zorder=3,
                    name='France',
                    marker=dict(size=15))

trace2 = go.Scatter(x=df_europe[df_europe['country'] == 'Germany']['year'], 
                    y=df_europe[df_europe['country'] == 'Germany']['lifeExp'], 
                    mode='lines+markers',
                    #zorder=1,
                    name='Germany',
                    marker=dict(size=15))

trace3 = go.Scatter(x=df_europe[df_europe['country'] == 'Spain']['year'], 
                    y=df_europe[df_europe['country'] == 'Spain']['lifeExp'], 
                    mode='lines+markers',
                    #zorder=2,
                    name='Spain',
                    marker=dict(size=15))

layout = go.Layout(title='Life Expectancy in Europe Over Time')

fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

fig.show()


# # Large Data Sets

# In[23]:


N = 100000
fig = go.Figure(data=go.Scattergl(
    x = np.random.randn(N),
    y = np.random.randn(N),
    mode='markers',
    marker=dict(
        color=np.random.randn(N),
        colorscale='Viridis',
        line_width=1
    )
))

fig.show()


# In[24]:


N =  100000
r = np.random.uniform(0, 1, N)
theta = np.random.uniform(0, 2*np.pi, N)

fig = go.Figure(data=go.Scattergl(
    x = r * np.cos(theta), # non-uniform distribution
    y = r * np.sin(theta), # zoom to see more points at the center
    mode='markers',
    marker=dict(
        color=np.random.randn(N),
        colorscale='Viridis',
        line_width=1
    )
))

fig.show()

