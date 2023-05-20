import dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc
import numpy as np
import plotly.figure_factory as ff
import pandas as pd
import plotly.express as px
from dash import dash_table, dcc, html
import flask
import pandas as pd
from sqlalchemy import create_engine
import os



#csv_path = '../data/combined.csv'
#csv_path = 'https://www.inf.u-szeged.hu/~vancsics/combined.zip'
csv_path = '../data/combined_without_Closure_34.zip'

# oszlopnev, amely szerinti a minimum elemeket tartjuk meg ha a filtered_aspect = True
aspect = 'Depth' #  'Tarantula-hit'
second_aspect = 'Tarantula-hit' if aspect == 'Depth' else 'Depth'

label_1 = "Csak a legkisebb " + aspect + " erteku eseteket vegyuk figyelembe (egy bug-on belul)"
label_2 = "A legkisebb " + aspect + " erteku eseteket kozul csak egyet vegyunk figyelembe (egy bug-on belul)"
label_3 = "Jeloljuk ki az alabbi projekte(ke)t"
label_4 = "A kijelolt projekte(ke)t..."


def get_projects():
    df = pd.read_csv(csv_path)
    return df['Project'].to_list()


def get_df():
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Tarantula_hit": "Tarantula-hit",
                            "Tarantula_freq_ef_ep_nf_np": "Tarantula-freq-ef_ep_nf_np"})
    return df


def drop_duplicates_in_df(df, drop_duplicated):
    if drop_duplicated:
        df = df.drop_duplicates(subset=['Project', 'Bug', aspect])
    return df


def filter_projects_in_df(df, selected_project_list, action_selected_project):
    if action_selected_project == 'kizarjuk':
        return df[~df['Project'].isin(selected_project_list)]
    return df[df['Project'].isin(selected_project_list)]


def drop_none_min_aspect_in_df(df, filtered_aspect):
    df['remove'] = True
    if filtered_aspect:
        min_aspect_df = df.groupby(['Project', 'Bug'])[aspect].agg('min').to_frame().reset_index()
        for _, row in min_aspect_df.iterrows():
            __ = df[(df['Project'] == row['Project']) & (df['Bug'] == row['Bug']) & (df[aspect] == row[aspect])]
            df.loc[__.index, 'remove'] = False
        df = df[~df['remove']]
    return df.drop(['remove'], axis=1)


def get_histo(df):
    hist_df = df[aspect].value_counts().to_frame().reset_index()
    hist_df = hist_df.sort_values(by=['index'])
    hist_df = hist_df.rename(columns={aspect: 'count'})
    hist_df = hist_df.rename(columns={"index": aspect})

    fig_histo = go.Figure()
    hist_vector = list()
    for _, row in hist_df.iterrows():
        hist_vector += [str(row[aspect]) for _ in range(int(row['count']))]
    fig_histo.add_trace(go.Histogram(x=hist_vector))
    fig_histo.update_layout(xaxis_title=aspect, yaxis_title="Number of buggy methods")
    return fig_histo


def get_histo_per_projects_1(df):
    fig_histo_per_projects = go.Figure()
    projects = sorted(list(set(list(df['Project'].tolist()))))
    for project in projects:
        _df = df[df['Project']==project]
        hist_df = _df[aspect].value_counts().to_frame().reset_index()
        hist_df = hist_df.sort_values(by=['index'])
        hist_df = hist_df.rename(columns={aspect: 'count'})
        hist_df = hist_df.rename(columns={"index": aspect})

        missing_depth = set([x for x in range(max(list(df['Depth'].to_list())))]) - set(list(hist_df['Depth'].to_list()))
        for x in missing_depth:
            hist_df.loc[len(hist_df)] = [x, 0]

        hist_df = hist_df.sort_values(by=['Depth'])
        hist_vector = list()
        for _, row in hist_df.iterrows():
            hist_vector += [str(row[aspect]) for _ in range(int(row['count']))]
        fig_histo_per_projects.add_trace(go.Histogram(x=hist_vector, name=project))
    fig_histo_per_projects.update_layout(xaxis_title=aspect, yaxis_title="Number of buggy methods")
    return fig_histo_per_projects


def get_histo_per_projects_2(df):
    fig_histo_per_projects = go.Figure()
    depths = max(list(df['Depth'].tolist()))
    for depth in range(depths + 1):
        _df = df[df['Depth'] == depth]
        hist_df = _df['Project'].value_counts().to_frame().reset_index()
        hist_df = hist_df.sort_values(by=['index'])
        hist_df = hist_df.rename(columns={'Project': 'count'})
        hist_df = hist_df.rename(columns={"index": 'Project'})

        hist_df = hist_df.sort_values(by=['Project'])
        hist_vector = list()
        for _, row in hist_df.iterrows():
            hist_vector += [str(row['Project']) for _ in range(int(row['count']))]
        fig_histo_per_projects.add_trace(go.Histogram(x=hist_vector, name=str(depth)))
    fig_histo_per_projects.update_layout(xaxis_title='Project', yaxis_title="Number of buggy methods")
    return fig_histo_per_projects


def get_strip(df):
    return px.strip(df, y=second_aspect, x=aspect, hover_data=df.columns)


def get_strip_2(df):
    return px.strip(df, y=second_aspect, x=aspect, color='Project', hover_data=df.columns)


def get_strip_3(df):
    return px.strip(df, y=second_aspect, x='Project', color=aspect, hover_data=df.columns)


def get_violin(df):
    return px.violin(df, y=second_aspect, x=aspect, color=aspect, box=True, points="all", hover_data=df.columns)


def get_violin_2(df):
    return px.violin(df, y=second_aspect, x=aspect, color='Project', box=True, points="all", hover_data=df.columns)


def get_violin_3(df):
    return px.violin(df, y=second_aspect, x='Project', color=aspect, box=True, points="all", hover_data=df.columns)

def get_heatmap(df):
    sorted_aspect_values = sorted(list(set(df[aspect].tolist())))
    sorted_second_aspect_values = sorted(list(set(df[second_aspect].tolist())))
    heatmap_matrix = np.zeros((len(sorted_aspect_values), len(sorted_second_aspect_values)))
    for index, row in df.iterrows():
        aspect_index = sorted_aspect_values.index(row[aspect])
        second_aspect_index = sorted_second_aspect_values.index(row[second_aspect])
        heatmap_matrix[aspect_index][second_aspect_index] += 1

    return px.imshow(heatmap_matrix, labels=dict(x=second_aspect, y=aspect, color="# methods"),
                     x=[str(x) for x in sorted_second_aspect_values], y=[str(y) for y in sorted_aspect_values],
                     text_auto=True)


def get_ternary(df):
    ternary_df = df.groupby(['Project', 'Bug', second_aspect])[aspect].agg('min').to_frame().reset_index()
    data = pd.DataFrame(columns=['Depth', 'Tarantula-hit', 'count'])
    for index, row in ternary_df.iterrows():
        depht = row['Depth']
        tar_hit = row['Tarantula-hit']
        count = df[(df['Depth'] == depht) & (df['Tarantula-hit'] == tar_hit)].shape[0]
        data.loc[len(data)] = [depht, tar_hit, count]
    return px.scatter_ternary(data, a="Depth", b="Tarantula-hit", c="count", size="count",
                              hover_data=['Depth', 'Tarantula-hit', 'count'])


def get_table(df):
    df['Bug'] = df['Bug'].astype(str)
    df['Depth'] = df['Depth'].astype(int)
    df['Tarantula-hit'] = df['Tarantula-hit'].astype(float)
    df['Tarantula-freq-ef_ep_nf_np'] = df['Tarantula-freq-ef_ep_nf_np'].astype(float)
    return dash_table.DataTable(id='market-level-table',
                                columns=[{"name": i, "id": i} for i in df.columns],
                                style_cell={'whiteSpace': 'pre-line'},
                                data=df.to_dict('records'),
                                filter_action="native",
                                sort_action="native",
                                page_action="native",
                                page_current=0,
                                page_size=10)


app = dash.Dash(__name__)
server = app.server

app.layout = dbc.Row([
    html.P([
        html.Label(label_1),
        dcc.Dropdown(['Igen', 'Nem'], placeholder='Igen / Nem', id='filtered_aspect')
    ], style={"width": "35%"}),

    html.P([
        html.Label(label_2),
        dcc.Dropdown(['Igen', 'Nem'], placeholder='Igen / Nem', id='drop_duplicated')
    ], style={"width": "35%"}),

    html.P([
        html.Label(label_3),
        dcc.Dropdown(sorted(get_projects()),
                     placeholder='projektek...', id='selected_project_list', multi=True),
    ], style={"width": "35%"}),

    html.P([
        html.Label(label_4),
        dcc.Dropdown(['kizarjuk', 'megjelenitjuk'], 'kizarjuk',
                     placeholder=label_4, id='action_selected_project'),
    ], style={"width": "35%"}),

    dbc.Spinner(children=[html.Br(),
                          dcc.Tabs(id='tabs-example-1', value='tab-1', children=[
                              dcc.Tab(label='Osszes', value='tab-1'),
                              dcc.Tab(label='Projektenkent', value='tab-2'),
                          ]),
                          html.Br(),
                          html.Div(id='output')],
                spinner_style={"width": "3rem", "height": "3rem"})
], className="g-0")




@app.callback(
    Output('output', 'children'),
    [Input('filtered_aspect', 'value'),
     Input('drop_duplicated', 'value'),
     Input('selected_project_list', 'value'),
     Input('action_selected_project', 'value'),
     Input('tabs-example-1', 'value')
     ]
)
def display_output(filtered_aspect, drop_duplicated, selected_project_list, action_selected_project, tab):
    if (filtered_aspect is None or filtered_aspect == '') and (drop_duplicated is None or drop_duplicated == ''):
        return ''

    if (selected_project_list is None or selected_project_list == '') and action_selected_project == 'megjelenitjuk':
        return ''

    if selected_project_list is None or selected_project_list == '':
        selected_project_list = []

    df = get_df()
    df['Tarantula-hit'] = df['Tarantula-hit'].astype(float)
    df['Depth'] = df['Depth'].astype(int)
    df = df.sort_values(by=['Project', 'Bug', 'Depth', 'Tarantula-hit'])

    df = drop_none_min_aspect_in_df(df, filtered_aspect)
    df = drop_duplicates_in_df(df, drop_duplicated)
    df = filter_projects_in_df(df, selected_project_list, action_selected_project)

    if tab == 'tab-1':
        return get_tab_figs(df)
    elif tab == 'tab-2':
        return get_tab_figs_per_project(df)



def get_tab_figs_per_project(df):
    return dcc.Tabs(
        id='tabs-per-project',
        children=[
            dcc.Tab(label=p, value='tab-' + str(p), children=get_tab_figs_2(df[df['Project'] == p])) for p in sorted(
                list(set(list(df['Project'].to_list()))))])


def get_tab_figs_2(df):
    return dcc.Tabs([
        dcc.Tab(label='Histograms', children=[
            dcc.Graph(figure=get_histo(df)),
        ]),
        dcc.Tab(label='Strips', children=[
            dcc.Graph(figure=get_strip(df)),
        ]),
        dcc.Tab(label='Violins', children=[
            dcc.Graph(figure=get_violin(df)),
        ]),
        dcc.Tab(label='Heatmap and table', children=[
            dcc.Graph(figure=get_heatmap(df)),
            get_table(df)])
    ])


def get_tab_figs(df):
    return dcc.Tabs([
        dcc.Tab(label='Histograms', children=[
            dcc.Graph(figure=get_histo(df)),
            dcc.Graph(figure=get_histo_per_projects_1(df)),
            dcc.Graph(figure=get_histo_per_projects_2(df))
        ]),
        dcc.Tab(label='Strips', children=[
            dcc.Graph(figure=get_strip(df)),
            dcc.Graph(figure=get_strip_2(df)),
            dcc.Graph(figure=get_strip_3(df))]),
        dcc.Tab(label='Violins', children=[
            dcc.Graph(figure=get_violin(df)),
            dcc.Graph(figure=get_violin_2(df)),
            dcc.Graph(figure=get_violin_3(df))]),
        dcc.Tab(label='Heatmap and table', children=[
            dcc.Graph(figure=get_heatmap(df)),
            get_table(df)])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)