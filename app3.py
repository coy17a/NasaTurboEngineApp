import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import dash_table
import plotly.graph_objects as go

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets=[dbc.themes.CERULEAN]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv('data/nasa_data.csv')
df['condition']=df['condition'].astype('category')
data_description = pd.read_csv('data/data_description.csv')
df_knn = pd.read_csv('data/knn_results.csv')
dt_results=pd.read_csv('data/dt_results.csv')

df_1 = pd.read_csv('data/nasa_data_clean.csv')
df_1.loc[:,'data_set'] = df_1['data_set'].astype('category')
fig_3dscatter = px.scatter_3d(df_1, x='Altitud', y='Mach Number', z='TRA',
                    color='data_set', title='Identification Operational Conditions',size_max=18)
df_1['o_condition']=0
condition1 = df_1['Altitud'] < 2
condition2 = (df_1['Altitud'] < 12) & (df_1['Altitud'] >2)
condition3 = (df_1['Altitud'] < 22) & (df_1['Altitud'] >12)
condition4 = (df_1['Altitud'] < 27) & (df_1['Altitud'] >22)
condition5 = (df_1['Altitud'] < 37) & (df_1['Altitud'] >27)
condition6 = (df_1['Altitud'] < 44) & (df_1['Altitud'] >37)
df_1.loc[condition1,'o_condition']=1
df_1.loc[condition2,'o_condition']=2
df_1.loc[condition3,'o_condition']=3
df_1.loc[condition4,'o_condition']=4
df_1.loc[condition5,'o_condition']=5
df_1.loc[condition6,'o_condition']=6
#df_1['o_condition']=df_1['o_condition'].astype('category')

fig_hist_lr=px.histogram(df_1,'target',title='RUL (Target) Histogram',nbins=20,marginal='box',color='o_condition')
fig_hist_lr.update_layout(yaxis_title="Count")

results_lr_base = pd.read_csv('data/lr_base_results.csv')
results_lr_base_viz= results_lr_base.query('alpha == 1.0000')
fig_lr_base=px.bar(results_lr_base_viz,x='variable',y='value',barmode='group',color='model'
                                        ,facet_row='variable',
                                        title='Linear Regression Base Models Performance in Test Data',
                                        height=800)
fig_lr_base.update_yaxes(matches=None)

w_results = pd.read_csv('data/lr_weights.csv')
df_over = pd.read_csv('data/nasa_data_final.csv')

categorical_options={'criterion':['gini','entropy'],
                      'splitter':['best','random']}
fe_results = pd.read_csv('data/fe_results.csv')
fe_results = fe_results[fe_results['index'].str.contains('test')]
fe_results1 = fe_results[(fe_results['model']== 'Original Data') | (fe_results['model']=='Feature_selection')]
fig_fe1=px.bar(fe_results1,x='variable',y='score',color='model',barmode='group',facet_col='index',title='Feature Engineering Resulst Using Decision Tree as Base Model')
fe_results2 = fe_results[(fe_results['model']== 'Original Data') | (fe_results['model']=='Feature_selection') | (fe_results['model']=='Feature_eng')]
fig_fe2=px.bar(fe_results2,x='variable',y='score',color='model',barmode='group',facet_col='index',title='Feature Engineering Resulst Using Decision Tree as Base Model')
fig_fe3=px.bar(fe_results,x='variable',y='score',color='model',barmode='group',title='Feature Engineering Resulst Using Decision Tree as Base Model')

fe_imp = pd.read_csv('data/feature_imp_fe.csv')
fe_imp.index=fe_imp['Unnamed: 0']
fe_imp= fe_imp.sort_values(by='Importance Value',ascending=False)
fig_fe_imp= px.bar(fe_imp,x='Importance Value', title='Feature Importance After Feature Engineering')

unit='2.0_2'
sensor='Ps30'
X_viz1 = pd.read_csv('data/sensor_viz.csv')
def plot_sensor(unit,sensor,template='plotly_dark'):
    plotdata=X_viz1[X_viz1['unit_number_id']==unit]
    plotdata=plotdata.iloc[10:,:]
    plotdata=plotdata.melt(id_vars=['unit_number_id','time_in_cycles'])
    plotdata= plotdata[plotdata['variable'].str.contains(sensor)]
    plotdata1 = plotdata[~plotdata['variable'].str.contains('std')]
    plotdata2 = plotdata[plotdata['variable'].str.contains('std')]
    fig = px.line(plotdata1,x='time_in_cycles',y='value',color='variable',title=f'Noise Reduction for Sensor {sensor},Unit:{unit}', template=template)
    return fig
fig_sensor=plot_sensor(unit,sensor,'plotly')
#----------------------------------------------------------------
import dash_bootstrap_components as dbc
#-----------------------------------------------------------------
def df_preprocess(value): 
    df_t=df
    target = value
    label_positive =df_t['target'] <= target 
    df_t['label_target']=0
    df_t.loc[label_positive,'label_target'] = 1
    dt= df_t.drop(columns=['max_cycles','target','unit_number','condition'])
    y=dt['label_target'].values
    X =dt.drop(columns=['label_target']).values
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42,stratify=y)
    return(dt,X_train, X_test, y_train, y_test )

def label_target_plot(df):
    y=df['label_target']
    positive_labels=y.sum()
    p_positive_labels2= positive_labels/len(y)*100
    p_negatives_labels2 = 100-p_positive_labels2
    negative_labels2 = len(y)-positive_labels
    label_distribution2=pd.DataFrame([[p_positive_labels2,p_negatives_labels2]],columns=['Positive','Negative'],index=['Percent'])
    label_distribution2= label_distribution2.T.reset_index()
    fig2= px.pie(label_distribution2, values='Percent',names='index',title='Label Distribution, Target={}. Total Samples={}'.format(target,len(y)))
    return fig2

def report(model,X_train, X_test, y_train, y_test):
    """ create score report for givin decission tree model"""
    #Fit Model
    model.fit(X_train,y_train)
    #Train Scores
    prediciton_train = model.predict(X_train)
    acc_train = accuracy_score(y_train,prediciton_train)
    p_score_train = precision_score(y_train,prediciton_train)
    r_score_train = recall_score(y_train,prediciton_train)
    f1_score_train = f1_score(y_train,prediciton_train)
    
    #Test Scores
    prediciton_test = model.predict(X_test)
    acc_test = accuracy_score(y_test,prediciton_test)
    p_score_test = precision_score(y_test,prediciton_test)
    r_score_test = recall_score(y_test,prediciton_test)
    f1_score_test = f1_score(y_test,prediciton_test)
    #create dataframe
    train_results = [acc_train,p_score_train,r_score_train,f1_score_train]
    test_results = [acc_test,p_score_test,r_score_test,f1_score_test]
    return train_results,test_results,model

def train_models(models,X_train, X_test, y_train, y_test):
    """List of models as input and prodcue a DF with the results for each model"""
    t_results =[]
    test_results=[]
    results_df = pd.DataFrame()
    for i,model in enumerate(models):
        t_results,test_results,model_clf = report(model,X_train, X_test, y_train, y_test)
        index1= 'train_'+str(i)
        index2 = 'test_'+str(i)
        data=pd.DataFrame([t_results,test_results],columns=['Accuracy','Precission','Recall','F1_score'],index=[index1,index2])
        results_df = pd.concat([results_df,data])
    results_df.reset_index(inplace=True)
    results_df=pd.melt(results_df,id_vars=['index'],value_name='score')
    results_df['index']= results_df['index'].astype('category')
    results_df['variable']= results_df['variable'].astype('category')
        
    return results_df

def train_models2(models,X_train, X_test, y_train, y_test,X_train2, X_test2):
    """List of models with process features as input and prodcue a DF with the results for each model"""
    t_results =[]
    test_results=[]
    results_df = pd.DataFrame()
    xtrain=[X_train,X_train2]
    xtest=[X_test,X_test2]
    for i,model in enumerate(models):
        t_results,test_results,model_clf = report(model,xtrain[i], xtest[i], y_train, y_test)
        index1= 'train_'+str(i)
        index2 = 'test_'+str(i)
        data=pd.DataFrame([t_results,test_results],columns=['Accuracy','Precission','Recall','F1_score'],index=[index1,index2])
        results_df = pd.concat([results_df,data])
    results_df.reset_index(inplace=True)
    results_df=pd.melt(results_df,id_vars=['index'],value_name='score')
    results_df['index']= results_df['index'].astype('category')
    results_df['variable']= results_df['variable'].astype('category')
        
    return results_df

#------------------------------------------------------------------------------------------------------------------------------------
#Correlation Matrix Plot
import plotly.io as pio
pio.templates.default = "none"
df_corr = df_1.copy()
df_corr.drop(columns=['max_cycles','unit_number_id'],inplace = True)
target = 25
label_positive =df_corr['target'] <= target 
df_corr['label_target']=0
df_corr.loc[label_positive,'label_target'] = 1

corr = df_corr.corr()

mask = np.zeros_like(corr, dtype = np.bool) #probably could do this on df… to check
mask[np.triu_indices_from(mask)] = True
corr1=corr.mask(mask)

X = corr1.columns.values
sns_colorscale = [[0.0, '#3f7f93'],
                    [0.071, '#5890a1'],
                    [0.143, '#72a1b0'],
                    [0.214, '#8cb3bf'],
                    [0.286, '#a7c5cf'],
                    [0.357, '#c0d6dd'],
                    [0.429, '#dae8ec'],
                    [0.5, '#f2f2f2'],
                    [0.571, '#f7d7d9'],
                    [0.643, '#f2bcc0'],
                    [0.714, '#eda3a9'],
                    [0.786, '#e8888f'],
                    [0.857, '#e36e76'],
                    [0.929, '#de535e'],
                    [1.0, '#d93a46']]
heat = go.Heatmap(z=corr1,
                    x=X, #check direct df column use
                    y=X, #check direct df row/index use
                    xgap=1, ygap=1,
                    colorscale=sns_colorscale, #“Magma”
                    colorbar_thickness=20,
                    colorbar_ticklen=3,
                    zmid=0, #added

#hovertext ='hovertext'
                       )
title = ""

layout = go.Layout(title_text=title, title_x=0.5,
                    width=800, height=800,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    yaxis_autorange='reversed')

corr_fig=go.Figure(data=[heat], layout=layout)


# dt_base= DecisionTreeClassifier()
# results=train_models([dt_base])
# df_new =results[results['index'].str.contains('test')]
# base = results['index'].str.contains('1')
# df_new.loc[:,'index'] = 'Base Model'
# #df_new.loc[base,'index'] = 'Best Model'
# fig_base = px.bar(df_new, x="variable", y="score",
#                  color='index', barmode='group',
#                  height=400,labels={"index": "Base Vs Best Model"})
#---------------------------------------------------------------------------------
# label distribution graph
df_l = df_1.copy()
target = 25
label_positive =df_l['target'] <= target 
df_l['label_target']=0
df_l.loc[label_positive,'label_target'] = 1
label_dist_original=label_target_plot(df_l)
label_dist_over=label_target_plot(df_over)
#---------------------------------------------------------------------------------
navbar = dbc.NavbarSimple(
    # children=[
    #     dbc.NavItem(dbc.NavLink("EDA", href="#")),
    #     dbc.DropdownMenu(
    #         children=[
    #             dbc.DropdownMenuItem("Bussiness Case", header=True),
    #             dbc.DropdownMenuItem("Modelling", href="#"),
    #             dbc.DropdownMenuItem("Documentation", href="#"),
    #         ],
    #         nav=True,
    #         in_navbar=True,
    #         label="More",
    #     ),
    # ],
    brand="AMII ML Technician I, TEAM 3",
    brand_href="#",
    color="dark",
    dark=True,
)
#------------------------------------------------------------------
jumbotron = dbc.Jumbotron(
    [
        html.H1("Nasa Turbofan Jet Engine Failure Study", className="display-3"),
        html.P(
            "QuAm Development "
            ,
            className="lead",
        ),
        html.Hr(className="my-2"),
        # #html.P(
        #     "Jumbotrons use utility classes for typography and "
        #     "spacing to suit the larger container."
        # ),
        # html.P(dbc.Button("Learn more", color="primary"), className="lead"),
    ]
)

##Figures 

#------------------------------------------------------------------
header_dt =dbc.Row(
           [dbc.Col(
                    html.H1('Clasification Problem Definition'),
                    width={'size':6,'offset':3})])

label_distribution = dbc.Row([ 
                            dbc.Col(
                                html.Div(className="",
                                children=[  
                                    dcc.Slider(id ='label_target',
                                       min=5,
                                       max=125,
                                       step=5,
                                       value=25,
                                    ),
                                       dcc.Graph( id='label_distribution',
                            #figure = fig)
                                       )]
                                ),
                                width={'size':6}
                            ),
                            dbc.Col(
                        html.Div(className="",
                        children=[
                            dcc.Graph(id='base_model')
                            #figure=fig_base)
                        ]
                        ),width={'size':6,'offset':0}
                    )
                        ])
#------------------------------------------------------------------
# base_model = dbc.Row([
#                     dbc.Col(
#                         html.Div(className="",
#                         children=[
#                             dcc.Graph(id='base_model')
#                             #figure=fig_base)
#                         ]
#                         ),width={'size':8,'offset':2}
#                     )
# ])

hyperparameters =dbc.Row(
           [dbc.Col(
                    html.H2('Hyper-Parameter Tunning'),
                    width={'size':6,'offset':3})])
hyperparameters_subtitle =dbc.Row(
           [dbc.Col(
                    html.H3('Numerical Parameters'),
                    width={'size':6,'offset':1})]) 
hyperparameters_subtitle2 =dbc.Row(
           [dbc.Col(
                    html.H3('Categorical Parameters'),
                    width={'size':6,'offset':1})])                     
param_dropdown= dbc.Row([ 
                            dbc.Col(
                                html.Div(className="",
                                children=[  
                                    dcc.Dropdown(id ='hyperparameter_option',
                                       options=[{'label':x,'value':x} for x in['min_samples_split','min_samples_leaf','max_depth']],
                                       value='min_samples_split',
                                       multi=False
                                    ),
                                    dcc.Dropdown(id ='score',
                                       options=[{'label':x,'value':x} for x in['Accuracy','Precission','Recall','F1_score']],
                                       value='Accuracy',
                                       multi=False
                                    ),
                                    dcc.Graph(id='hyperparameter',
                                    
                                    )
                            ]),width={'size':4,'offset':1}),
                        dbc.Col(
                        html.Div(className="",
                        children=[
                            dcc.Dropdown(id ='hyperparameter_option2',
                                       options=[{'label':x,'value':x} for x in['min_samples_split','min_samples_leaf','max_depth']],
                                       value='min_samples_split',
                                       multi=False
                                    ),
                                    dcc.Dropdown(id ='score2',
                                       options=[{'label':x,'value':x} for x in['Accuracy','Precission','Recall','F1_score']],
                                       value='Accuracy',
                                       multi=False
                                    ),
                                    dcc.Graph(id='hyperparameter2',
                                    
                                    )
                        ]
                        ),width={'size':4,'offset':1}
                    )
                        ])


param_dropdown2= dbc.Row([ 
                            dbc.Col(
                                html.Div(className="",
                                children=[  
                                    dcc.Dropdown(id ='categorical_parameter',
                                       options=[{'label':x,'value':x} for x in categorical_options.keys()],
                                       value='criterion',
                                       multi=False
                                    ),
                                    dcc.Dropdown(id ='categorical_value',
                                    value="",
                                    multi=False
                                    ),
                                    dcc.Graph(id='hyperparameter_categorical',
                                    
                                    )
                            ]),width={'size':8,'offset':2})
                    
                        ])
best_model_title =dbc.Row(
           [dbc.Col(
                    html.H3('Finding Best Model'),
                    width={'size':8,'offset':2})])  
best_model_options = dbc.Row([dbc.Col(
                        html.Div(className='',
                        children=[
                            html.Header('Criterion'),
                            dcc.Dropdown(id='best_criterion',
                            options=[{'label':x,'value':x} for x in ['gini','entropy']],
                            value='gini')
                        ]), width={'size':2,'offset':1}
                    ),
                    dbc.Col(
                        html.Div(className='',
                        children=[
                            html.Header('Splitter'),
                            dcc.Dropdown(id='best_splitter',
                            options=[{'label':x,'value':x} for x in ['best','random']],
                            value='best'
                            )
                        ]), width=2
                    ),
                    dbc.Col(
                        html.Div(className='',
                        children=[
                            html.Header('Minimum Sample Splits'),
                            dcc.Dropdown(id='best_min_sample',
                            options=[{'label':x,'value':x} for x in [5,10,25,50,75]],
                            value=5)
                        ]), width=2
                    ),
                    dbc.Col(
                        html.Div(className='',
                        children=[
                            html.Header('Minimum Sample Leaf'),
                            dcc.Dropdown(id='best_min_leaf',
                            options=[{'label':x,'value':x} for x in [5,10,25,50,75]],
                            value=5)
                        ]), width=2
                    ),
                    dbc.Col(
                        html.Div(className='',
                        children=[
                            html.Header('Maximum Depth'),
                            dcc.Dropdown(id='best_max_depth',
                            options=[{'label':x,'value':x} for x in [5,10,25,50,75]],
                            value=5)
                        ]), width=2
                    )
                    
                    ])
best_model = dbc.Col(
                html.Div(className='',
                children=[
                    dcc.Graph(id='best_model'
                    
                    )
                ]), width={'size':8,'offset':2}
    )
feature_importance = dbc.Col(
                html.Div(className='',
                children=[
                    dcc.Graph(id='feature_importance'
                    
                    )
                ]), width={'size':8,'offset':2}
    )
header_knn =dbc.Row(
           [dbc.Col(
                    html.H1('K-nearest Neighbourghs kNN'),
                    width={'size':6,'offset':3})])
label_distribution2 = dbc.Row([ 
                            dbc.Col(
                                html.Div(className="",
                                children=[  
                                    dcc.Slider(id ='label_target2',
                                       min=5,
                                       max=125,
                                       step=5,
                                       value=25,
                                    ),
                                       dcc.Graph( id='label_distribution2',
                            #figure = fig)
                                       )]
                                ),
                                width={'size':6}
                            ),
                            dbc.Col(
                        html.Div(className="",
                        children=[
                            dcc.Graph(id='base_model2')
                            #figure=fig_base)
                        ]
                        ),width={'size':6,'offset':0}
                    )
                        ])


best_model_title2 =dbc.Row(
           [dbc.Col(
                    html.H3('Finding Best Model'),
                    width={'size':8,'offset':2})]) 


best_model_options_knn = dbc.Row([dbc.Col(
                        html.Div(className='',
                        children=[
                            html.Header('Neighbours'),
                            dcc.Slider(id='n',
                            min=1,
                            max=20,
                            step=None,
                            value=5,
                            marks={ 
                                1:'1',
                                3:'3',
                                5:'5',
                                7:'7',
                                10:'10',
                                15:'15',
                                20:'20'
                            })

                        ]), width={'size':5,'offset':1}
                    ),
                    dbc.Col(
                        html.Div(className='',
                        children=[
                            html.Header('Weight'),
                            dcc.Dropdown(id='best_weight',
                            options=[{'label':x,'value':x} for x in ['uniform','distance']],
                            value='uniform'
                            )
                        ]), width={'size':2,'offset':1}
                    ),
                    dbc.Col(
                        html.Div(className='',
                        children=[
                            html.Header('Data Transformation'),
                            dcc.Dropdown(id='data_transformation',
                            options=[{'label':x,'value':x} for x in ['Original','Normalization(Max_min)','Standarization']],
                            value='Original'
                            )
                        ]), width={'size':2,'offset':0}
                    ),
                    ])
best_modelknn = dbc.Col(
                html.Div(className='',
                children=[
                    dcc.Graph(id='best_modelknn'
                    
                    )
                ]), width={'size':8,'offset':2}
    )
hyper_knn= dbc.Row([ 
                            dbc.Col(
                                html.Div(className="",
                                children=[  
                                    html.Header('Neighbourgh Effect and Preprocessing Effect'),
                                    dcc.Dropdown(id ='hyperparameter_option_knn', 
                                     options=[{'label':x,'value':x} for x in ['Accuracy','Precission','Recall','F1_score','All']],
                                    value='Accuracy'
                                            ),
                                   dcc.Graph(id='hyperparameter_knn',
                                    )
                            ]),width={'size':10,'offset':1})
])
#--------------------------------------------------------------------------------------------------------
#EDA
eda_1= dbc.Row(
           [dbc.Col(
                    html.H1('Exploratory Data Analysys'),
                    width={'size':8,'offset':4})])
eda_2=dbc.Row([
                 dbc.Col(html.Div(className='',
                              children=[
                
                                # html.H3(children='''
                                #   Histogram Visualization
                                # '''),
                                html.Header(' Select the Experiment Condition'),
                                dcc.Dropdown(id = 'conditions',
                                    options=[{'label':condition,'value':condition} for condition in df_1.data_set.unique()],
                                    value=df_1.data_set.unique(),
                                    multi=True
                                ),
                                html.Header('Bins Selection'),
                                dcc.Slider(
                                            id='slider_histogram1',
                                            min=5,
                                            max=100,
                                            step=1,
                                            value=50,
                                        ),
                                html.Header(' Select one of the Features for Histrogam 1'),
                                        dcc.Dropdown(id = 'feature',
                                        options=[{'label':feature,'value':feature} for feature in df_1.columns],
                                        value='target',
                                        multi=False),
                                dcc.Graph(
                                    id='histogram',
                                    # figure=fig
                                )
                      ]
                    ), width=5),
                 
              dbc.Col(html.Div(className='',
                        children=[  
                        # html.H3(children='''
                        #           Histogram Visualization
                        #         '''),                                      
                        html.Header(' Select the Experiment Condition'),
                        dcc.Dropdown(id = 'conditions2',
                                    options=[{'label':condition,'value':condition} for condition in df_1.data_set.unique()],
                                    value=df_1.data_set.unique(),
                                    multi=True
                        ),
                        html.Header('Bins Selection'),
                        dcc.Slider(
                                    id='slider_histogram2',
                                    min=5,
                                    max=100,
                                    step=1,
                                    value=50,
                                        ),
                        html.Header(' Select one of the Features for Histrogam 2'),
                                        html.Div(),
                                        dcc.Dropdown(id = 'feature2',
                                        options=[{'label':feature,'value':feature} for feature in df_1.columns],
                                        value='target',
                                        multi=False),
                                      
                        dcc.Graph(
                                    id='histogram2',
                            # figure=fig
                        )
                      ]
                   ), width=5),
           ], justify='center')
eda_3= dbc.Row([dbc.Col(className = '',
                        children=[
                            html.H2('Operational Conditions'),
                            dcc.Graph(id='3d_scatter',
                            figure=fig_3dscatter)
                        ],width={'size':8,'offset':2})
])

eda_4= dbc.Row([ dbc.Col(className='',
                        children=[html.H2('Scatter Matrix'),
                        html.Header(' Select the Experiment Condition'),
                        dcc.Dropdown(id = 'conditions3',
                                    options=[{'label':condition,'value':condition} for condition in df_1.o_condition.unique()],
                                    value=df_1.o_condition.unique(),
                                    multi=True
                        ),
                        html.Header(' Select the Feature to Compare'),
                            dcc.Dropdown(id = 'matrix_feature',
                                        options=[{'label':f,'value':f} for f in df.columns],
                                        value=['target','Ps30','T30'],
                                        multi=True
                            ),
                            
                            dcc.Graph(
                                    id='scatter_matrix',
                                    style={'display':'block'}
                                     )
                                ],
                           width=10,
                           
                           )
                  ],justify='center')
eda_5 =  dbc.Row([ dbc.Col(className='',
                        children=[html.H2('Correlation Matrix'),
                            
                            dcc.Graph(
                                    id='corr_matrix',
                                    style={'display':'block'},
                                    figure=corr_fig 
                                     )
                                ],
                           width={'size':8,'offset':2},                           
                           )
                  ],justify='center')

eda_6 = dbc.Row([dbc.Col(className='',
                         children=[html.H2('Sensor Trend'),
                         dcc.Dropdown(id='sensor_selection',
                         options = [{'label':condition,'value':condition} for condition in df_1.o_condition.unique()],
                         value= df_1.o_condition.unique()[0],
                         multi= False
                         )
                         ]) 
                         ])
     
eda_0 = html.Div(children=[eda_1,eda_2,eda_3,eda_4,eda_5])  


data_description = dbc.Col(className='',
            children=[ html.H2('Data Description'),
            dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in data_description.columns],
            data=data_description.to_dict('records'),
            style_cell={'fontSize':16, 'font-family':'sans-serif'}
            ) 
            ],width={'size':4,'offset':4} )

header_lr =dbc.Row(
           [dbc.Col(
                    html.H1('Linear Regression'),
                    width={'size':8,'offset':4})])

header_od =dbc.Row(
           [dbc.Col(
                    html.H1('Turbo Engine Status'),
                    width={'size':6,'offset':4})])
#--------------------------------------------------------------------------------------------------------
# Linear Regression
l_histogram = dbc.Col(html.Div(className='',
                              children=[
                
                                # html.H3(children='''
                                #   Histogram Visualization
                                # '''),
                                html.Header(' Select the Experiment Condition'),
                                dcc.Dropdown(id = 'o_conditions',
                                    options=[{'label':condition,'value':condition} for condition in df_1.o_condition.unique()],
                                    value=df_1.o_condition.unique(),
                                    multi=True
                                ),
                                html.Header('Bins Selection'),
                                dcc.Slider(
                                            id='lr_hist_bins',
                                            min=5,
                                            max=100,
                                            step=5,
                                            value=20,
                                        ),
                                
                                dcc.Graph(
                                    id='histogram_lr',
                                    #figure=fig_hist_lr
                                )
                      ]
                    ), width={'size':10,'offset':1})
lr_base_models = dbc.Row([
                 dbc.Col(html.Div(className='',
                              children=[
                
                                # html.H3(children='''
                                #   Histogram Visualization
                                # '''),
                                html.Header(''),
                                
                                dcc.Graph(
                                    id='lr_base',
                                    figure=fig_lr_base
                                )
                      ]
                    ), width=10),
                    
           ], justify='center')  
hyper_lr =dbc.Row(
           [dbc.Col(
                    html.H1('Hyper Parameter Tunning'),
                    width={'size':8,'offset':4})])



hyper_lr_metrics= dbc.Row([dbc.Col(
                        html.Div(className='',
                        children=[ 
                        html.Header('LR Metric Selection'),    
                        dcc.Dropdown(id='metric_lr_L1',
                        options=[{'label':x,'value':x} for x in ['MAE','MSE','RMSE','R2']],
                        value='RMSE')
                        ])
                       ,
                         width={'size':3,'offset':1}   
                        ),
                        dbc.Col(
                        html.Div(className='',
                        children=[     
                        html.Header('LR Metric Selection'),    
                        dcc.Dropdown(id='metric_lr_L2',
                        options=[{'label':x,'value':x} for x in ['MAE','MSE','RMSE','R2']],
                        value='RMSE')
                        ])
                        , width={'size':3,'offset':3}   
                        )
                        ]) 


hyper_lr_graph= dbc.Row([dbc.Col(
                        html.Div(className='',
                        children=[
                            html.Header(''),
                            dcc.Graph(id='l1_graph',                            
                            )

                        ]), width={'size':5,'offset':1}
                    ),
                    dbc.Col(
                        html.Div(className='',
                        children=[
                            html.Header(''),
                            dcc.Graph(id='l2_graph',)
                            ]), width={'size':5,'offset':1}
                    )
                ])
feature_selection_lr =dbc.Row(
           [dbc.Col(
                    html.H1('Feature Importance LR '),
                    width={'size':8,'offset':4})])
feature_selection_lr_graph= dbc.Row([ 
                            dbc.Col(
                                html.Div(className="",
                                children=[  
                                    html.Header('Feature Importance'),
                                    dcc.Dropdown(id ='lr_regularization', 
                                     options=[{'label':x,'value':x} for x in ['L1','L2']],
                                    value='L2'
                                            ),
                                    dcc.Dropdown(id ='lr_regularization_alpha', 
                                     options=[{'label':x,'value':x} for x in [0.0001,0.001,0.01,0.1,0.5,1]],
                                    value=0.0001
                                            ),
                                   dcc.Graph(id='lr_feature_importance',
                                    )
                            ]),width={'size':10,'offset':1})
])
#--------------------------------------------------------------------------------------------------------------------------------------------------
#Feature Engineering
header_r =dbc.Row(
           [dbc.Col(
                    html.H1('Feature Selection'),
                    width={'size':6,'offset':3})])
 
feature_importance_fe = dbc.Row([
            dbc.Col(
                html.Div(className='',
                children=[
                    dcc.Graph(id='feature_importance_fe'
                    
                    )
                ]), width={'size':5,'offset':1}
    ),
    dbc.Col(
                html.Div(className='',
                children=[
                    dcc.Graph(id='feature_importance_fegraph',
                    figure=fig_fe1
                    
                    )
                ]), width={'size':4,'offset':1}
    )
    ])
header_fe =dbc.Row(
           [dbc.Col(
                    html.H1('Feature Engineering'),
                    width={'size':6,'offset':3})])
feature_eng_fe = dbc.Row([
            dbc.Col(
                html.Div(className='',
                children=[
                    dcc.Graph(id='noise_reduction',
                    figure=fig_sensor
                    
                    )
                ]), width={'size':5,'offset':1}
    ),
    dbc.Col(
                html.Div(className='',
                children=[
                    dcc.Graph(id='feature_eng_performance',
                    figure=fig_fe2
                    
                    )
                ]), width={'size':4,'offset':1}
    )
    ])
fe_graph= dbc.Row([
            dbc.Col(
                html.Div(className='',
                children=[
                    dcc.Graph(id='feature_eng',
                    figure=fig_fe_imp
                    
                    )
                ]), width={'size':10,'offset':1}
    ) ])
oversampling_1 = dbc.Row([
            dbc.Col(
                html.Div(className='',
                children=[
                    dcc.Graph(id='label_dist_original',
                    figure=label_dist_original
                    
                    )
                ]), width={'size':5,'offset':1}
    ),
    dbc.Col(
                html.Div(className='',
                children=[
                    dcc.Graph(id='label_dist_over',
                    figure=label_dist_over
                    
                    )
                ]), width={'size':4,'offset':1}
    )
    ])
fe_results_graph= dbc.Row([
            dbc.Col(
                html.Div(className='',
                children=[
                    dcc.Graph(id='feature_eng_results',
                    figure=fig_fe3
                    
                    )
                ]), width={'size':10,'offset':1}
    ) ])
 #----------------------------------------------------------------------------------------   
card_content0 = [
    dbc.CardHeader("Ps30"),
    dbc.CardBody(
        [
            html.H3("47.2 psi", className="card-title"),
            html.P(
                "Static pressure at HPC outlet (psi)",
                className="card-text",
            ),
        ]
    ),
]
card_content1 = [
    dbc.CardHeader("Altitud"),
    dbc.CardBody(
        [
            html.H3("41.2 kft", className="card-title"),
            html.P(
                "Altitud Test",
                className="card-text",
            ),
        ]
    ),
]
card_content2 = [
    dbc.CardHeader("TRA"),
    dbc.CardBody(
        [
            html.H3("100 º", className="card-title"),
            html.P(
                "Throttle resolver angle (Degrees)",
                className="card-text",
            ),
        ]
    ),
]
card_content3 = [
    dbc.CardHeader("T2"),
    dbc.CardBody(
        [
            html.H3("518.67 Rº", className="card-title"),
            html.P(
                "Total temperature at fan inlet (Rº)",
                className="card-text",
            ),
        ]
    ),
]
card_content4 = [
    dbc.CardHeader("T50"),
    dbc.CardBody(
        [
            html.H3("1120.5 Rº", className="card-title"),
            html.P(
                "Total temperature at LPT outlet (Rº)",
                className="card-text",
            ),
        ]
    ),
]
card_content5 = [
    dbc.CardHeader("Phi"),
    dbc.CardBody(
        [
            html.H3("530 (pps/psi)", className="card-title"),
            html.P(
                "Ratio of fuel flow to Ps30 ",
                className="card-text",
            ),
        ]
    ),
]
card_content6 = [
    dbc.CardHeader("P2"),
    dbc.CardBody(
        [
            html.H3("14.62 psi", className="card-title"),
            html.P(
                "Pressure at fan inlet",
                className="card-text",
            ),
        ]
    ),
]
card_content7 = [
    dbc.CardHeader("Nf"),
    dbc.CardBody(
        [
            html.H3("2215 rpm", className="card-title"),
            html.P(
                "Physical Fan Speed",
                className="card-text",
            ),
        ]
    ),
]
card_content8 = [
    dbc.CardHeader("BPR"),
    dbc.CardBody(
        [
            html.H3("9.3", className="card-title"),
            html.P(
                "By Pass Ratio",
                className="card-text",
            ),
        ]
    ),
]
card_content9 = [
    dbc.CardHeader("RUL Prediciton"),
    dbc.CardBody(
        [
            html.H3("13 Cycles", className="card-title"),
            html.P(
                "Remaining Useful Life",
                className="card-text",
            ),
        ]
    ),
]
card_content10 = [
    dbc.CardHeader("Engine Status"),
    dbc.CardBody(
        [
            html.H3("Maintance Required", className="card-title"),
            html.P(
                "RUL below 25 cycles",
                className="card-text",
            ),
        ]
    ),
]

cards = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Card(card_content0, color="info", inverse=True)),
                dbc.Col(
                    dbc.Card(card_content1, color="info", inverse=True)
                ),
                dbc.Col(dbc.Card(card_content2, color="info", inverse=True)),
                dbc.Col(dbc.Card(card_content3, color="info", inverse=True))
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(card_content4, color="info", inverse=True)),
                dbc.Col(dbc.Card(card_content5, color="info", inverse=True)),
                dbc.Col(dbc.Card(card_content6, color="info", inverse=True)),
                dbc.Col(dbc.Card(card_content7, color="info", inverse=True))
            ],
            className="mb-3",
        ),
        dbc.Row([
            dbc.Col(
                html.Div(className='',
                children=[
                    html.Header('Sensor Selection'),
                    dcc.Dropdown(id ='sensor_operation_option',
                                       options=[{'label':x,'value':x} for x in X_viz1.columns[0:10]],
                                       value='Ps30',
                                       multi=False
                                    ),
                    dcc.Graph(id='noise_reduction2',
                    
                    )
                ]), width={'size':12,'offset':0}
            )
        ]),
        dbc.Row(
            [   
                dbc.Col(html.H3('Prediciton of Engine Status'))
            ]
        ),
        dbc.Row(
            [   
                dbc.Col(dbc.Card(card_content9, color="light"),width={'size':4,'offset':1}),
                dbc.Col(dbc.Card(card_content10, color="danger", inverse=True),width={'size':4,'offset':1}),
            ]
        ),
    ]
)

#--------------------------------------------------------------------------------------------------------------------------------------------------
tab_eda=[data_description,
        eda_0]


tab_dt = [header_dt,
            label_distribution,
            hyperparameters,
            hyperparameters_subtitle,
            param_dropdown,
            hyperparameters_subtitle2,
            param_dropdown2,
            best_model_title,
            best_model_options, 
            best_model,
            feature_importance]
tab_knn=[header_knn,
        label_distribution2,
         hyperparameters,
         hyper_knn,
        best_model_title2,
        best_model_options_knn,
        best_modelknn]

tab_lr=[header_lr,l_histogram,
        lr_base_models,
        hyper_lr,
        hyper_lr_metrics,
        hyper_lr_graph,
        feature_selection_lr,
        feature_selection_lr_graph,
        ]


tab_r=[header_r,
    feature_importance_fe,
    header_fe,
    feature_eng_fe,
    fe_graph,
    oversampling_1,
    fe_results_graph]

tab_od = [header_od,cards
         ]

tabs = dbc.Tabs(
    [ 
        dbc.Tab(tab_eda,label='EDA'),
        dbc.Tab(tab_dt,label='Decission Trees'),
        dbc.Tab(tab_knn,label='kNN'),
        dbc.Tab(tab_lr, label='Linear Regression'),
        dbc.Tab(tab_r,label= 'Feature Engineering'),
        dbc.Tab(tab_od,label= 'Operations')
    ]
)
#--------------------------------------------------------------------
app.layout = html.Div(children=[navbar,
                                jumbotron,
                                tabs
                                ])


#----------------------------------------------------------------------
@app.callback(
    [Output('label_distribution', 'figure'),
    Output('base_model', 'figure')],
    [Input('label_target', 'value')])
def update_output(value):
    df_t=df
    target = value
    label_positive =df_t['target'] <= target 
    df_t['label_target']=0
    df_t.loc[label_positive,'label_target'] = 1
    
    positive_labels=df_t['label_target'].sum()
    p_positive_labels= positive_labels/len(df_t)*100
    p_negatives_labels = 100-p_positive_labels
    negative_labels = len(df_t)-positive_labels
    label_distribution=pd.DataFrame([[p_positive_labels,p_negatives_labels]],columns=['Positive','Negative'],index=['Percent'])
    label_distribution= label_distribution.T.reset_index()
    fig= px.pie(label_distribution, values='Percent',names='index',title='Label Distribution, Target={}. Total Samples={}'.format(value,len(df_t)))
    #fig=px.bar(label_distribution,x='Percent',y='index',title='Label Distribution, Target={}. Total Samples={}'.format(value,len(df_t)))
    
    dt,X_train, X_test, y_train, y_test = df_preprocess(value)
    dt_base= DecisionTreeClassifier()
    results=train_models([dt_base],X_train, X_test, y_train, y_test)
    df_new =results[results['index'].str.contains('test')]
    base = results['index'].str.contains('1')
    df_new.loc[:,'index'] = 'Base Model'
    #df_new.loc[base,'index'] = 'Best Model'
    fig_base = px.bar(df_new, x="variable", y="score",
                    color='variable',
                    height=500,title='Decision Tree Scores on Test Data for Base Model')


    return fig,fig_base

@app.callback(
Output('hyperparameter', 'figure'),
[Input('hyperparameter_option', 'value'),
Input('score','value')])

def hyper_update(parameter,score):
    
    dt_viz=dt_results[dt_results.metric==parameter]
    fig=px.scatter(dt_viz,x=parameter,y=score,color='set')

    
    return fig

@app.callback(
Output('hyperparameter2', 'figure'),
[
Input('hyperparameter_option2', 'value'),
Input('score2','value')])
def hyper_update2(parameter,score):
    dt_viz=dt_results[dt_results.metric==parameter]
    fig=px.scatter(dt_viz,x=parameter,y=score,color='set')
    return fig
@app.callback(
    Output('categorical_value', 'options'),
    [Input('categorical_parameter', 'value')])
def categorical_slection(value):
    categorical_options={'criterion':['gini','entropy'],
                      'splitter':['best','random']}
    return [{'label':x,'value':x} for x in categorical_options[value]]

# @app.callback(
# Output('score3', 'value'),
# [Input('score3', 'opttions')])

# def categorical_options0(options):
#     return options['value']      

@app.callback(
Output('hyperparameter_categorical', 'figure'),
[Input('label_target','value'),
Input('categorical_parameter', 'value'),
Input('categorical_value','value')])
def categorical_options1(value,hyper1,hyper_option):
    # df_t=df
    # target = value
    # label_positive =df_t['target'] <= target 
    # df_t['label_target']=0
    # df_t.loc[label_positive,'label_target'] = 1
      
    # dt= df_t.drop(columns=['max_cycles','target','unit_number','condition'])
    # y=dt['label_target'].values
    # X =dt.drop(columns=['label_target']).values
    # X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42,stratify=y)
    dt,X_train, X_test, y_train, y_test = df_preprocess(value)
    dt_base= DecisionTreeClassifier(random_state=42)
    if hyper_option != '':
        if hyper1 =='criterion':
                if hyper_option in ['gini','entropy']:
                    dt_new=DecisionTreeClassifier(criterion=hyper_option,random_state=42)
                else:
                    dt_new=dt_base

        elif hyper1=='splitter':
                if hyper_option in ['best','random']:
                    dt_new=DecisionTreeClassifier(splitter=hyper_option,random_state=42)
                else:
                    dt_new=dt_base
    else: 
        dt_new=dt_base
    results=train_models([dt_base,dt_new],X_train, X_test, y_train, y_test)
    df_new =results[results['index'].str.contains('test')]
    base = results['index'].str.contains('1')
    df_new.loc[:,'index'] = 'Base Model'
    df_new.loc[base,'index'] = 'New Model'
    fig = px.bar(df_new, x="variable", y="score",
                    color='index',
                    barmode='group',
                    height=500,title='Decision Tree Categorical Hyperparameter Effect',
                    labels={"index": f"Base Vs Model {hyper1} = {hyper_option}"})

    return fig            

@app.callback(
[Output('best_model', 'figure'),
Output('feature_importance', 'figure'),
Output('feature_importance_fe','figure')],
[Input('label_target','value'),
Input('best_criterion','value'),
Input('best_splitter', 'value'),
Input('best_min_sample','value'),
Input('best_min_leaf','value'),
Input('best_max_depth','value')
])  

def update_best_model(value,criterion,splitter,min_sample,min_leaf,max_depth):
    # df_t=df
    # target = value
    # label_positive =df_t['target'] <= target 
    # df_t['label_target']=0
    # df_t.loc[label_positive,'label_target'] = 1
      
    # dt= df_t.drop(columns=['max_cycles','target','unit_number','condition'])
    # y=dt['label_target'].values
    # X =dt.drop(columns=['label_target']).values
    # X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42,stratify=y)
    dt,X_train, X_test, y_train, y_test = df_preprocess(value)
    dt_base= DecisionTreeClassifier(random_state=42)
    dt_new = DecisionTreeClassifier(criterion=criterion,
                                    splitter=splitter,
                                    min_samples_split=min_sample,
                                    min_samples_leaf=min_leaf,
                                    max_depth=max_depth,
                                    random_state=42)
    results=train_models([dt_base,dt_new],X_train, X_test, y_train, y_test)
    df_new =results[results['index'].str.contains('test')]
    base = results['index'].str.contains('1')
    df_new.loc[:,'index'] = 'Base Model'
    df_new.loc[base,'index'] = 'New Model'
    fig = px.bar(df_new, x="variable", y="score",
                    color='index',
                    barmode='group',
                    height=500,title='Best Decision Tree ',
                    labels={"index": f"Base Vs Best Model: {min_leaf},{min_sample}"}
                    #template='ggplot2'
                    #color_discrete_sequence =['pink','blue']
                    )
    columns_names=dt.drop(columns=['label_target']).columns
    df_f= pd.DataFrame(dt_new.feature_importances_,index=columns_names,columns=['Importance Value'])
    df_f= df_f.sort_values(by='Importance Value',ascending=False)
    fig2=px.bar(df_f[:10],x='Importance Value',title='Feature Importance')
    fig3=px.bar(df_f[:10],x='Importance Value',title='Feature Selection Form Decision Trees')

    return fig,fig2,fig3

@app.callback(
    [Output('label_distribution2', 'figure'),
    Output('base_model2', 'figure')],
    [Input('label_target2', 'value')])
def update_knn_base_model(value):
    df_t=df
    target = value
    label_positive =df_t['target'] <= target 
    df_t['label_target']=0
    df_t.loc[label_positive,'label_target'] = 1
    
    positive_labels=df_t['label_target'].sum()
    p_positive_labels= positive_labels/len(df_t)*100
    p_negatives_labels = 100-p_positive_labels
    negative_labels = len(df_t)-positive_labels
    label_distribution=pd.DataFrame([[p_positive_labels,p_negatives_labels]],columns=['Positive','Negative'],index=['Percent'])
    label_distribution= label_distribution.T.reset_index()
    fig= px.pie(label_distribution, values='Percent',names='index',title='Label Distribution, Target={}. Total Samples={}'.format(value,len(df_t)))
    #fig=px.bar(label_distribution,x='Percent',y='index',title='Label Distribution, Target={}. Total Samples={}'.format(value,len(df_t)))
    
    dt,X_train, X_test, y_train, y_test = df_preprocess(value)
    dt_base= KNeighborsClassifier()
    results=train_models([dt_base],X_train, X_test, y_train, y_test)
    df_new =results[results['index'].str.contains('test')]
    df_new.loc[:,'index'] = 'Base Model'
    #df_new.loc[base,'index'] = 'Best Model'
    fig_base = px.bar(df_new, x="variable", y="score",
                    color='variable',
                    height=500,title='kNN scores in Test Data for Base Model')
    return fig,fig_base

@app.callback(
Output('best_modelknn', 'figure'),
[Input('label_target','value'),
Input('n','value'),
Input('best_weight', 'value'),
Input('data_transformation','value')
])  

def update_best_model2(value,n,weight,data_transform):
    knn_df=df_knn
    base_results = knn_df[(knn_df['k']==5) & (knn_df['weight']=='uniform') & (knn_df['data']==0)]

    base_results.loc[:,'index']='Base Model'

    if data_transform == 'Original':
        data_transform = 0
    elif data_transform == 'Normalization(Max_min)':
        data_transform = 1
    else :
        data_transform = 2 

    new_results = knn_df[(knn_df['k']==n) & (knn_df['weight']==weight) & (knn_df['data']==data_transform)]
    new_results.loc[:,'index']='New model'
    df_new = pd.concat([base_results,new_results])

    # dt,X_train, X_test, y_train, y_test = df_preprocess(value)
    # if data_transform == 'Normalization(Max_min)':
    #     MinMax_norm = MinMaxScaler().fit(X_train)
    #     X_train2= MinMax_norm.transform(X_train)
    #     X_test2=MinMax_norm.transform(X_test)
    # elif data_transform == 'Standarization':
    #     ss_std = StandardScaler().fit(X_train)
    #     X_train2= ss_std.transform(X_train) 
    #     X_test2 =ss_std.transform(X_test) 
    # else:
    #     X_train2= X_train
    #     X_test2 = X_test

    # model_base= KNeighborsClassifier()
    # model_new = KNeighborsClassifier(n_neighbors=n,weights=weight)
    # results=train_models2([model_base,model_new],X_train, X_test, y_train, y_test,X_train2,X_test2)
    # df_new =results[results['index'].str.contains('test')]
    # base = results['index'].str.contains('1')
    # df_new.loc[:,'index'] = 'Base Model'
    # df_new.loc[base,'index'] = 'New Model'

    
    fig = px.bar(df_new, x="variable", y="score",
                    color='index',
                    barmode='group',
                    height=500,title='Best kNN Model',
                    labels={"index": f"Base Vs Best Model: {n},{weight}"}
                    #template='ggplot2'
                    #color_discrete_sequence =['pink','blue']
                    )

    return fig
@app.callback(
    Output('hyperparameter_knn', 'figure'),
    [Input('hyperparameter_option_knn', 'value')])
def udatate_hyper_knn(value):
    knn_df=df_knn
    knn_df['data2']=knn_df['data'].map({0: 'Original', 1: 'Normalize',2:'Standarize'})
    df_knn_viz=knn_df.query('weight=="uniform"')
    if value == 'All':
        fig= px.line(df_knn_viz,x='k',y='score',color='data2',facet_col='variable')
    else:
        df_knn_viz=df_knn_viz[df_knn_viz['variable']==value]
        fig= px.line(df_knn_viz,x='k',y='score',color='data2')

    return fig

#--------------------------------------------------------------------------
# EDA callbacks 

@app.callback(
  Output('histogram','figure'),
  [Input('feature','value'),
  Input ('conditions','value'),
  Input('slider_histogram1','value')]
)
def update_graph(feature,conditions,bins):
  dff =[]
  if len(conditions)>0:
    for condition in conditions:
      dff.append(df_1.query('data_set=={}'.format(condition)))  
    dff = pd.concat(dff)
  else:
    dff = df_1
  fig=px.histogram(dff,feature,title='{} Histogram'.format(feature),nbins=bins,marginal='box')#color='condition')
  fig.update_layout(
      yaxis_title="Count")
  return fig

@app.callback(
Output('histogram2','figure'),
[Input('feature2','value'),
Input ('conditions2','value'),
 Input('slider_histogram2','value')]
)
def update_graph2(feature,conditions,bins):
  dff =[]
  if len(conditions)>0:
    for condition in conditions:
      dff.append(df_1.query('data_set=={}'.format(condition)))  
    dff = pd.concat(dff)
  else:
    dff = df_1
  fig=px.histogram(dff,feature,title='{} Histogram'.format(feature),nbins=bins,marginal='box',color='data_set')
  fig.update_layout(
      yaxis_title="Count")
  return fig 

@app.callback(
Output('scatter_matrix','figure'),
[Input('matrix_feature','value'),
Input ('conditions3','value')]
)
def update_graph3(feature,conditions):
  dff =[]
  if len(conditions)>0:
    for condition in conditions:
      dff.append(df_1.query('o_condition=={}'.format(condition)))  
    dff = pd.concat(dff)
  else:
    dff = df_1

  if len(feature)>1:
    fig=px.scatter_matrix(dff, dimensions=feature,color='o_condition',opacity=0.4)
    fig.update_traces(diagonal_visible=True)#)
  else:
    pass
  return fig


@app.callback(
  Output('histogram_lr','figure'),
  [Input('lr_hist_bins','value'),
  Input ('o_conditions','value')]
)
def update_histogram_lr(bins,o_conditions):
  dff =[]
  if len(o_conditions)>0:
    for condition in o_conditions:
      dff.append(df_1.query('data_set=={}'.format(condition)))  
    dff = pd.concat(dff)
  else:
    dff = df_1
  fig=px.histogram(dff,'target',title='RUL ("target") Histogram',nbins=bins,marginal='box',color='o_condition',template='plotly_white')
  fig.update_layout(
      yaxis_title="Count")
  return fig

@app.callback(
  Output('l1_graph','figure'),
  [Input('metric_lr_L1','value')]
)
def update_lr_l1(metric):
    df_viz=results_lr_base.copy()
    df_viz=df_viz[(df_viz.model=="L1") & (df_viz.variable==metric)]
    fig= px.line(df_viz,x='alpha',y='value',color='model',title='Lasso Regularization')
    return fig

@app.callback(
  Output('l2_graph','figure'),
  [Input('metric_lr_L2','value')]
)
def update_lr_l2(metric):
    df_viz=results_lr_base.copy()
    df_viz=df_viz[(df_viz.model=="L2") & (df_viz.variable==metric)]
    fig= px.line(df_viz,x='alpha',y='value',color='model',title='Ridge Regularizaiton')
    return fig

@app.callback(
Output('lr_feature_importance','figure'),
[Input('lr_regularization','value'),
Input('lr_regularization_alpha','value')]
)

def update_lr_features(model,alpha):
    w_plot=w_results[(w_results.model==model) & (w_results.alpha ==alpha)]
    w_plot=w_plot.sort_values(by='weight')
    fig=px.bar(w_plot,x='feature',y='weight',title=f'Feature Weights, Model:{model}, alpha:{alpha}')
    return fig   

@app.callback(
    Output('noise_reduction2','figure'),
    [Input('sensor_operation_option','value')]
)
def update_senor_op(sensor):
    unit='2.0_2'
    fig=plot_sensor(unit,sensor)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
