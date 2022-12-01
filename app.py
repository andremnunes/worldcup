import googletrans
from googletrans import Translator
translator = Translator()

# ------------------------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import missingno as msno
import re
import sys, tempfile, urllib, os
import networkx as nx
import altair as alt
import nx_altair as nxa
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
def novaExibicaoDataFrame(data):
  gb = GridOptionsBuilder.from_dataframe(data)
  gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
  gb.configure_side_bar() #Add a sidebar
  gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
  gridOptions = gb.build()

  grid_response = AgGrid(
      data,
      gridOptions=gridOptions,
      data_return_mode='AS_INPUT', 
      update_mode='MODEL_CHANGED', 
      fit_columns_on_grid_load=False,
      theme='streamlit', #Add theme color to the table
      enable_enterprise_modules=True,
      height=350, 
      width='100%',
      reload_data=True
  )

  data = grid_response['data']
  selected = grid_response['selected_rows'] 
  df = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df

  return df

def ajustaCodeCountryISO(df, campo):
  df[campo + 'ISO'] = df[campo] 
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Ivory Coast','Côte d\'Ivoire')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Zaire', 'Congo, The Democratic Republic of the')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Bolivia','Bolivia, Plurinational State of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Iran','Iran, Islamic Republic of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Russian','Russian Federation')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Scotland','United Kingdom')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('England','United Kingdom')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Wales','United Kingdom')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Northern Ireland','United Kingdom')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('USA','United States')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Korea','Korea, Republic of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Germany FR','Germany')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('East Germany','Germany')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('West Germany','Germany')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Czechoslovakia', 'Czechia')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Soviet Union', 'Russian Federation')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Yugoslavia', 'Serbia')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Federal Republic of Yugoslavia', 'Serbia')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Kingdom of Yugoslavia', 'Serbia')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Socialist Federal Republic of Yugoslavia', 'Serbia')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('Korea Republic','Korea, Republic of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('South Korea','Korea, Republic of')
  df[campo + 'ISO'] = df[campo + 'ISO'].replace('North Korea','Korea, Democratic People\'s Republic of')

  df[campo + '_Code'] = [countries.get(country, 'Unknown code') for country in df[campo + 'ISO']]

  return(df)

def buscar_resultados(partidas):
    if partidas['Home Team Goals'] > partidas['Away Team Goals']:
        return 'Home Team Vencedor'
    elif partidas['Home Team Goals'] < partidas['Away Team Goals']:
        return 'Away Team Vencedor'
    elif partidas['Home Team Goals'] == partidas['Away Team Goals']:
        return 'Empate'
    else:
        return ''

def makeUniqueCategories(categoriesGames):
    categories = list() 
    categoriesGames = categoriesGames.to_numpy() 
    for categoriesString in categoriesGames:
        #categoriesString = categoriesString.replace(" ", "")
        listOfCategories = categoriesString.split(",")
        for category in listOfCategories:
            if len(categories) == 0:
                categories.append(category)
            else:
                if not (category in categories):
                    categories.append(category)
    return categories

# ------------------------------------------------------------------------------

import pycountry

countries = {}
for Country in pycountry.countries:
  countries[Country.name] = Country.alpha_3

# ------------------------------------------------------------------------------

# https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/

import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

# ------------------------------------------------------------------------------

def top_entries(df):
    mat = df.corr().abs()
    
    # Remove duplicate and identity entries
    mat.loc[:,:] = np.tril(mat.values, k=-1)
    mat = mat[mat>0]

    # Unstack, sort ascending, and reset the index, so features are in columns
    # instead of indexes (allowing e.g. a pretty print in Jupyter).
    # Also rename these it for good measure.
    return (mat.unstack()
             .sort_values(ascending=False)
             .reset_index()
             .rename(columns={
                 "level_0": "Atributo X",
                 "level_1": "Atributo Y",
                 0: "Correlação"
             }))


# ------------------------------------------------------------------------------

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud2(data, imagem):
    mask = np.array(Image.open(imagem))
    wordcloud = WordCloud(width=800, height=400, stopwords=stopwords, background_color="white", mode="RGBA", max_words=1000, mask=mask).generate(data)
    # create coloring from image
    image_colors = ImageColorGenerator(mask)
    fig = plt.figure(figsize=[20,10])
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    plt.show()

    st.pyplot(fig)

# ------------------------------------------------------------------------------

# https://discuss.streamlit.io/t/how-do-i-use-a-background-image-on-streamlit/5067/5

import base64

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(main_bg);
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack('https://raw.githubusercontent.com/andremnunes/worldcup/c0848f0c80b075ad6f84ba1b927d6e7487ef90ea/logo.png')

# -------------------

#image = Image.open('logo.png')
#st.image(image, caption='FIFA World Cup 2022')

# https://docs.streamlit.io/library/api-reference/text
st.title("Copa do Mundo da FIFA")

st.header("Storytelling")

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------

nome_dataset = st.selectbox('Qual o conjunto de dados gostaria de analisar?', 
['Resultados das Copas do Mundo',
'Resultados das Partidas',
'Jogadores',
'Clube de Origem',
'Convocações'
]
, key = "nome_dataset")

if nome_dataset == "Resultados das Copas do Mundo":
  option = "ResultadoCopa"
  arquivo_csv = "https://raw.githubusercontent.com/andremnunes/worldcup/c0848f0c80b075ad6f84ba1b927d6e7487ef90ea/dataset/WorldCupMatches.csv"
  nome_csv = "WorldCups.csv"
elif nome_dataset == "Resultados das Partidas":
  option = "ResultadoPartidas"
  arquivo_csv = "https://raw.githubusercontent.com/andremnunes/worldcup/c0848f0c80b075ad6f84ba1b927d6e7487ef90ea/dataset/WorldCupMatches.csv"
  nome_csv = "WorldCupMatches.csv"
elif nome_dataset == "Jogadores":
  option = "Jogadores"
  arquivo_csv = "https://raw.githubusercontent.com/andremnunes/worldcup/c0848f0c80b075ad6f84ba1b927d6e7487ef90ea/dataset/WorldCupPlayers.csv"
  nome_csv = "WorldCupPlayers.csv"
elif nome_dataset == "Clube de Origem":
  option = "ClubeOrigem"
  arquivo_csv = "https://raw.githubusercontent.com/andremnunes/worldcup/c0848f0c80b075ad6f84ba1b927d6e7487ef90ea/dataset/worldcup_squads.csv"
  nome_csv = "worldcup_squads.csv"
elif nome_dataset == "Convocações":
  option = "Convocacoes"
  arquivo_csv = "https://raw.githubusercontent.com/andremnunes/worldcup/c0848f0c80b075ad6f84ba1b927d6e7487ef90ea/dataset/worldcup_squads.csv"
  nome_csv = "worldcup_squads.csv"
else:
  option = ""

st.write(f'Você escolheu: {nome_dataset} ({nome_csv})')

# ------------------------------------------------------------------------------

if option == "ResultadoCopa":
  df_csv = pd.read_csv(arquivo_csv, sep=',', thousands='.')

  # ----------------------------------------------------------------------------

  # Pré-processamento
  # =====================

  # https://en.wikipedia.org/wiki/2018_FIFA_World_Cup
  new_row = {'Year':2018, 
           'Country':'Russian Federation', 
           'Winner':'France', 
           'Runners-Up':'Croatia', 
           'Third':'Belgium', 
           'Fourth':'England', 
           'GoalsScored':169, 
           'QualifiedTeams':32, 
           'MatchesPlayed':64, 
           'Attendance':3031768}
  df_csv = df_csv.append(new_row, ignore_index=True)

  # https://en.wikipedia.org/wiki/2022_FIFA_World_Cup - 30/11/2022
  new_row = {'Year':2022, 
           'Country':'Qatar', 
           'Winner':'', 
           'Runners-Up':'', 
           'Third':'', 
           'Fourth':'', 
           'GoalsScored':92, 
           'QualifiedTeams':32, 
           'MatchesPlayed':38, 
           'Attendance':1914090}
  df_csv = df_csv.append(new_row, ignore_index=True)

  # https://stackoverflow.com/questions/41486867/duplicate-row-if-a-column-contains-any-special-character-in-python-pandas
  df_csv = df_csv.drop('Country', axis=1) \
        .join(df_csv['Country'] \
        .str \
        .split('/', expand=True) \
        .stack() \
        .reset_index(level=1, drop=True).rename('Country')) \
        .reset_index(drop=True)

  df_csv.replace(to_replace='Germany FR', value='Germany', inplace=True)
  df_csv['Ref_Country'] = df_csv['Country']
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('England','United Kingdom')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('USA','United States')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('Korea','Korea, Republic of')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('Germany FR','Germany')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('Czechoslovakia', 'Czechia')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('Soviet Union', 'Russian Federation')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('Yugoslavia', 'Serbia')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('Korea Republic','Korea, Republic of')
  df_csv['Country_Code'] = [countries.get(country, 'Unknown code') for country in df_csv.Ref_Country]

  # ----------------------------------------------------------------------------

elif option == "ResultadoPartidas":
  df_csv = pd.read_csv(arquivo_csv)
  df_csv.drop_duplicates(inplace=True)

  # ----------------------------------------------------------------------------

  # Pré-processamento
  # =====================

  df_csv["Stadium"].replace(to_replace="Maracan� - Est�dio Jornalista M�rio Filho", value="Estadio do Maracana", inplace=True)
  df_csv.replace(to_replace='Germany FR', value='Germany', inplace=True)
  df_csv.replace(to_replace='rn">Republic of Ireland', value='Republic of Ireland', inplace=True)
  df_csv.replace(to_replace='rn">United Arab Emirates', value='United Arab Emirates', inplace=True)
  df_csv.replace(to_replace='rn">Trinidad and Tobago', value='Trinidad and Tobago', inplace=True)
  df_csv.replace(to_replace='rn">Serbia and Montenegro', value='Serbia and Montenegro', inplace=True)
  df_csv.replace(to_replace='rn">Bosnia and Herzegovina', value='Bosnia and Herzegovina', inplace=True)
  df_csv.replace(to_replace="C�te d'Ivoire", value="Côte d'Ivoire", inplace=True)
  df_csv.fillna('', inplace=True)
  df_csv['Stage'] = df_csv['Stage'].replace(to_replace='Group [A-Za-z0-9]+', value = 'Fase de grupos', regex = True)
  df_csv['Stage'] = df_csv['Stage'].replace(to_replace='Semi-finals', value = 'Semi-finais')
  df_csv['Stage'] = df_csv['Stage'].replace(to_replace='Preliminary round', value = 'Rodada preliminar')
  df_csv['Stage'] = df_csv['Stage'].replace(to_replace='Quarter-finals', value = 'Quartas de final')
  df_csv['Stage'] = df_csv['Stage'].replace(to_replace='Match for third place', value = 'Jogo pelo terceiro lugar')
  df_csv['Stage'] = df_csv['Stage'].replace(to_replace='First round', value = 'Primeira Rodada')
  df_csv['Stage'] = df_csv['Stage'].replace(to_replace='Third place', value = 'Terceiro lugar')
  df_csv['Stage'] = df_csv['Stage'].replace(to_replace='Round of 16', value = 'Oitavas de final')
  df_csv['Stage'] = df_csv['Stage'].replace(to_replace='Play-off for third place', value = 'Disputa pelo terceiro lugar')
  df_csv['Stage'] = df_csv['Stage'].replace(to_replace='Third place play-off', value = 'Disputa pelo terceiro lugar')

elif option == "Jogadores":
  df_csv = pd.read_csv(arquivo_csv)
  df_csv.drop_duplicates(inplace=True)

  # ----------------------------------------------------------------------------

  # Pré-processamento
  # =====================

  df_csv.replace(to_replace='Germany FR', value='Germany', inplace=True)

elif option == "ClubeOrigem":
  df_csv = pd.read_csv(arquivo_csv)
  df_csv.drop_duplicates(inplace=True)

elif option == "Convocacoes":
  df_csv = pd.read_csv(arquivo_csv)
  df_csv.drop_duplicates(inplace=True)


  # ----------------------------------------------------------------------------
  # Pré-processamento
  # =====================

  df_csv.replace(to_replace='Germany FR', value='Germany', inplace=True)

  # Idade
  df_csv['Age'] = df_csv['DOB/Age'].str.extract('\(aged (.+)\)', expand=False).str.strip()


# ------------------------------------------------------------------------------  
if option != "ClubeOrigem":
  df = filter_dataframe(df_csv)
  st.dataframe(df)

# ----------------------------

# ------------------------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------------------------

if st.checkbox('GRÁFICOS'):

# ------------------------------------------------------------------------------

  # MARCELO

  if option == "ResultadoCopa":
  
    # --------------------------------------------------------------------------

    st.write("O que queremos visualizar neste gráfico:")
    st.write(" - Quais os países que já sediaram a Copa do Mundo FIFA?")
    st.write(" - Quais os países que nunca sediaram a Copa do Mundo FIFA?")

    fig = px.choropleth(
    df.sort_values('Year'), 
    locations="Country_Code",
    color="Country", 
    hover_name="Country",
    hover_data = df.columns.tolist(),
    title = "Países"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("O que queremos visualizar neste gráfico:")
    st.write(" - Quais os países que já sediaram a Copa do Mundo FIFA mais de uma vez?")

    fig = px.choropleth(
      df.sort_values('Year'), 
      locations="Country_Code",
      color="Country", 
      hover_name="Country",
      hover_data = df.columns.tolist(),
      title = "País",
      color_continuous_scale=px.colors.sequential.Burg,
      animation_frame='Year')

    fig.update_geos(fitbounds = "locations", visible = True)

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------

    st.write("O que queremos visualizar neste gráfico:")
    st.write(" - Como foi o comportamento das partidas jogadas (MatchesPlayed)?")
    st.write(" - Como foi o comportamento dos gols marcados (GoalsScored)?")
    st.write(" - Como foi o comportamento do comparecimento do público (Attendance)?")

    y_column = st.selectbox('Selecione a coluna do eixo y', df.drop('Year', axis=1).select_dtypes('number').columns, key = "y_grafico_de_linha")
    st.line_chart(data=df, x="Year", y=y_column, width=0, height=0, use_container_width=True)  

    # --------------------------------------------------------------------------

    newdf = df[['Year', 'GoalsScored','QualifiedTeams', 'MatchesPlayed', 'Attendance']].drop_duplicates()
    mychart1 = alt.Chart(newdf).encode(
        alt.X('Year:O'),
        alt.Y('GoalsScored:Q', sort='-y'),
        alt.Color('GoalsScored:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=newdf.columns.tolist(),
    )

    mychart2 = alt.Chart(newdf).encode(
        alt.X('Year:O'),
        alt.Y('QualifiedTeams:Q', sort='-y'),
        alt.Color('QualifiedTeams:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=newdf.columns.tolist(),
    )

    mychart3 = alt.Chart(newdf).encode(
        alt.X('Year:O'),
        alt.Y('MatchesPlayed:Q', sort='-y'),
        alt.Color('MatchesPlayed:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=newdf.columns.tolist(),
    )

    mychart4 = alt.Chart(newdf).encode(
        alt.X('Year:O'),
        alt.Y('Attendance:Q', sort='-y'),
        alt.Color('Attendance:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=newdf.columns.tolist(),
    )

    st.write("Observe que em todos os atributos houve tendência de crescimento com exceção do ano de 2022 que ainda não temos os dados atualizados.")

    data_container = st.container()
    with data_container:
      plot1, plot2 = st.columns(2)
      with plot1:
        st.altair_chart(mychart1.mark_bar() + mychart1.mark_point() + mychart1.mark_line(), use_container_width=True)
        st.write("No atributo GoalsScored, a Copa de 1998 está em nível idêntico a Copa de 2014 que foi o maior nível alcançado de 171 gols no total.")

        st.altair_chart(mychart3.mark_bar() + mychart3.mark_point() + mychart3.mark_line(), use_container_width=True)
        st.write("Com o aumento do número de equipes qualificadas também aumentou proporcionalmente o número de partidas jogadas.")

      with plot2:
        st.altair_chart(mychart2.mark_bar() + mychart2.mark_point() + mychart2.mark_line(), use_container_width=True)
        st.write("As equipes qualificadas se mantiveram no nível de 13 equipes de 1930 até 1950.")
        st.write("A partir da Copa de 1954 as equipes qualificadas passaram a 16 e se mantiveram neste nível até a Copa de 1978.")
        st.write("Novos aumentos nas Copas de 1982 e 1998, passando para 24 e 32 equipes qualificadas respectivamente.")
        st.write("O número de equipes qualificadas permaneceu constante ao longo dos anos.")

        st.altair_chart(mychart4.mark_bar() + mychart4.mark_point() + mychart4.mark_line(), use_container_width=True)
        st.write("Com o aumento do número de equipes qualificadas e o consequente aumento de países participantes, também aumentou o número de torcedores.")
        st.write("Houve um salto de 1962 a 1966 muito provavelmente em função do desenvolvimento da tecnologia de transmissão dos jogos (rádio, televisão, etc).")
        st.write("O maior público foi atingido na Copa do Mundo de 1994 que foi sediada nos Estados Unidos.")
        st.write("Houve uma queda brusca na audiência em 1998 e a partir daí observa-se uma ligeira tendência de queda.")

    # --------------------------------------------------------------------------

    st.write("O que queremos visualizar neste gráfico:")
    st.write(" - Quais países ganharam a Copa do Mundo FIFA?")
    st.write(" - Quais países ganharam ficaram em segundo lugar?")
    st.write(" - Quais países ganharam ficaram em terceiro lugar?")
    st.write(" - Quais países ganharam ficaram em quarto lugar?")

    newdf = df[['Year','Winner','Runners-Up','Third','Fourth','GoalsScored','QualifiedTeams','MatchesPlayed','Attendance']]
    mychart1 = alt.Chart(newdf.drop_duplicates()).mark_bar().encode(
    x='Winner',
    y='count()',
    color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis', reverse=True)),    
    tooltip=newdf.columns.tolist()
    )

    mychart2 = alt.Chart(newdf.drop_duplicates()).mark_bar().encode(
    x='Runners-Up',
    y='count()',
    color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis', reverse=True)),  
    tooltip=newdf.columns.tolist()
    )

    mychart3 = alt.Chart(newdf.drop_duplicates()).mark_bar().encode(
    x='Third',
    y='count()',
    color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis', reverse=True)),
    tooltip=newdf.columns.tolist()
    )

    mychart4 = alt.Chart(newdf.drop_duplicates()).mark_bar().encode(
    x='Fourth',
    y='count()',
    color=alt.Color('Year:O', scale=alt.Scale(scheme='viridis', reverse=True)),
    tooltip=newdf.columns.tolist()
    )

    data_container = st.container()
    with data_container:
      plot1, plot2 = st.columns(2)
      with plot1:
        st.altair_chart(mychart1, use_container_width=True)
        st.write("Os países ganhadores da Copa do Mundo FIFA: Argentina, Brasil, Inglaterra, França, Alemanha, Itália, Espanha e Uruguai.")
        st.write("O Brasil tem o maior número de vitórias em Copas do Mundo.")

        st.altair_chart(mychart3, use_container_width=True)
      with plot2:
        st.altair_chart(mychart2, use_container_width=True)
        st.altair_chart(mychart4, use_container_width=True)       

  # ------------------------------------------------------------------------------

  # MARCELO

  if option == "ResultadoPartidas":

    # --------------------------------------------------------------------------

    if st.checkbox('RESULTADOS DOS JOGOS'):

    # --------------------------------------------------------------------------

      df['Resultado'] = df.apply(lambda x: buscar_resultados(x), axis=1)
      df1 = df[['Year', 'Stage', 'Home Team Name', 'Resultado']].loc[df['Resultado'] == "Home Team Vencedor"].rename(columns={'Home Team Name': 'Nome'})
      df2 = df[['Year', 'Stage', 'Away Team Name', 'Resultado']].loc[df['Resultado'] == "Away Team Vencedor"].rename(columns={'Away Team Name': 'Nome'})
      df11 = df[['Year', 'Stage', 'Away Team Name', 'Resultado']].loc[df['Resultado'] == "Home Team Vencedor"].rename(columns={'Away Team Name': 'Nome'})
      df11.replace(to_replace='Home Team Vencedor', value='Derrota', inplace=True)
      df22 = df[['Year', 'Stage', 'Home Team Name', 'Resultado']].loc[df['Resultado'] == "Away Team Vencedor"].rename(columns={'Home Team Name': 'Nome'})
      df22.replace(to_replace='Away Team Vencedor', value='Derrota', inplace=True)
      df3 = df[['Year', 'Stage', 'Home Team Name', 'Resultado']].loc[df['Resultado'] == "Empate"].rename(columns={'Home Team Name': 'Nome'})
      df4 = df[['Year', 'Stage', 'Away Team Name', 'Resultado']].loc[df['Resultado'] == "Empate"].rename(columns={'Away Team Name': 'Nome'})

      st.write("O que queremos visualizar neste gráfico:")
      st.write(" - Qual foi o resultado das partidas ao longo dos anos da Copa do Mundo FIFA?")
      st.write("Observação: Para cada vitória há uma derrota por isso os números estão pareados.")

      if st.checkbox('Excluir as derrotas?'):
        df5 = pd.concat([df1, df2, df3, df4])
      else:
        df5 = pd.concat([df1, df2, df3, df4, df11, df22])        

      df5.replace(to_replace='Germany FR', value='Germany', inplace=True)
      df5.replace(to_replace='Home Team Vencedor', value='Vitória', inplace=True)
      df5.replace(to_replace='Away Team Vencedor', value='Vitória', inplace=True)

      mychart0 = alt.Chart(df5).encode(    
          x='Year:O',    
          y='count(Resultado):Q',    
          color=alt.Color('Resultado:N', scale=alt.Scale(scheme="viridis")),
          tooltip=['Year','Resultado','count(Resultado):Q'],
      )
      st.altair_chart(mychart0.mark_bar(), use_container_width=True)

      st.write("Observe que curiosamente na primeira Copa realizada em 1930, não houve empates.")
      st.write("Proporcionalmente ao aumento do número de equipes qualificadas há um aumento do número de partidas jogadas.")
      st.write("Com o aumento do número de partidas jogadas também aumentou proporcionalmente o número de vitórias, derrotas e empates.")

      st.write("O que queremos visualizar neste gráfico:")
      st.write(" - A mesma informação obtida no gráfico acima, mas particionada pelo Stage: Final, fase de grupos, oitavas de final, quartas de final e etc.")

      mychart1 = alt.Chart(df5).encode(    
          x='Year:O',    
          y='count(Resultado):Q',    
          color=alt.Color('Resultado:N', scale=alt.Scale(scheme="viridis")),
          tooltip=['Year','Stage','Resultado','count(Resultado):Q'],
      )
      st.altair_chart(mychart1.mark_bar(), use_container_width=True)

      st.write("O que queremos visualizar neste gráfico:")
      st.write(" - Uma visão geral dos resultados dos jogos por Stage: Final, fase de grupos, oitavas de final, quartas de final e etc.")

      uniqueCategories2 = makeUniqueCategories(df5['Stage'])
      input_dropdown2 = alt.binding_select(options=uniqueCategories2, name="Stage: ")
      selection2 = alt.selection_single(fields=['Stage'], bind=input_dropdown2)
      
      uniqueCategories3 = makeUniqueCategories(df5['Nome'])
      input_dropdown3 = alt.binding_select(options=uniqueCategories3, name="Nome: ")
      selection3 = alt.selection_single(fields=['Nome'], bind=input_dropdown3)
      
      mychart2 = alt.Chart(df5).encode(
          x='Year:O',    
          y='count(Resultado):Q',    
          color=alt.Color('Resultado:O', scale=alt.Scale(scheme="viridis")),
          tooltip=['Year','Resultado','count(Resultado):Q'],
          column=alt.Column('Stage:N')
      )
      st.altair_chart(mychart2.mark_bar().add_selection(selection2).transform_filter(selection2).interactive(), use_container_width=True)

      st.write("Observe a Final das Copas de 1994 e 2006: Estas Copas foram decididas nos pênaltis.")
      st.write("Após a cobrança de 9 penalidades, o Brasil venceu a Itália por 3–2 nos pênaltis, e se tornou a primeira seleção tetracampeã do mundo.")
      
      link = '[Wikipédia](https://pt.wikipedia.org/wiki/Final_da_Copa_do_Mundo_FIFA_de_1994)'
      st.markdown(link, unsafe_allow_html=True)

      st.write("A partida final da Copa do Mundo FIFA de 2006 foi disputada em 9 de julho no Olympiastadion, na cidade de Berlim na Alemanha. A Itália venceu a França nos pênaltis após empate por 1–1 em 120 minutos de jogo. Um dos momentos mais comentados da partida foi a expulsão de Zinedine Zidane após agredir o jogador Marco Materazzi com uma cabeçada.")
      link = '[Wikipédia](https://pt.wikipedia.org/wiki/Final_da_Copa_do_Mundo_FIFA_de_2006)'
      st.markdown(link, unsafe_allow_html=True)

      st.write("O que queremos visualizar neste gráfico:")
      st.write(" - Como foi o desempenho da seleção brasileira de futebol ao longo das Copas do Mundo FIFA?")

      df_resultado = df5.copy()
      df_resultado["Total"]=1
      df_resultado = df_resultado.groupby(["Year","Stage","Nome","Resultado"])["Total"].sum().reset_index()
      df_resultado.columns = ["Year","Stage","Nome","Resultado","Total"]
      df_resultado = df_resultado.groupby(["Year","Stage","Nome","Resultado"])["Total"].sum().reset_index()
      df_resultado = df_resultado.sort_values(by="Total",ascending =False)
      df_resultado["Total"] = df_resultado["Total"].astype(int)

      mychart3 = alt.Chart(df_resultado).encode(     
          x='Year:O',    
          y='Total:Q',  
          color=alt.Color('Resultado:N', scale=alt.Scale(scheme="viridis")),
          tooltip=['Year','Stage','Resultado','Total:Q'],
          column=alt.Column('Nome:N')
      )
      st.altair_chart(mychart3.mark_bar().add_selection(selection3).transform_filter(selection3).interactive(), use_container_width=True)
      
      st.write("Observe o desempenho do Brasil ao longo das Copas do Mundo.")

      st.write("É a seleção mais bem-sucedida da história do futebol mundial, sendo a recordista em conquistas em Copas do Mundo, com cinco títulos invictos (1958, 1962, 1970, 1994 e 2002).")
      link = '[Wikipédia](https://pt.wikipedia.org/wiki/Sele%C3%A7%C3%A3o_Brasileira_de_Futebol)'
      st.markdown(link, unsafe_allow_html=True)

      st.write("A final da Copa do Mundo FIFA de 2002 foi disputada em 30 de junho no International Stadium, na cidade de Yokohama no Japão, entre a Seleção Brasileira e a Seleção Alemã. Foi a primeira vez que as duas equipes se confrontaram numa Copa do Mundo[1], a sétima vez que a Alemanha disputava uma final de Copa do Mundo, e a sexta do Brasil (sendo a terceira de forma consecutiva), já que em 1950 não houve uma final propriamente dita, e sim uma rodada final de um quadrangular.")
      st.write("Ao final de 90 minutos, o Brasil venceu a Alemanha por 2–0, com Ronaldo Nazário sendo o autor de ambos os gols. Com isso, o Brasil se tornou pentacampeão mundial, feito até hoje não igualado por mais nenhuma seleção. A Alemanha perdeu a final da Copa do Mundo pela quarta vez, mais um recorde no torneio.")
      st.write("O Brasil se tornou, também, a primeira equipe a conquistar a Copa do Mundo em 3 continentes diferentes (Europa, Ásia e Américas), feito igualado em 2014 pela Alemanha.")

      link = '[Wikipédia](https://pt.wikipedia.org/wiki/Final_da_Copa_do_Mundo_FIFA_de_2002)'
      st.markdown(link, unsafe_allow_html=True)

      st.write("O que queremos visualizar neste gráfico:")
      st.write(" - Como foi o desempenho da seleção brasileira de futebol na Final da Copa do Mundo FIFA?")

      mychart4 = alt.Chart(df_resultado).encode(    
          x='Year:O',    
          y='Total:Q',
          color=alt.Color('Resultado:O', scale=alt.Scale(scheme="viridis")),
          tooltip=['Year','Resultado','Total:Q'],
          column=alt.Column('Nome:N')
      )
      st.altair_chart(mychart4.mark_bar().add_selection(selection3).add_selection(selection2).transform_filter(selection3).transform_filter(selection2).interactive(), use_container_width=True)
      
      st.write("Observe o desempenho do Brasil na Final.")
      st.write("Em 17 de julho de 1994, há exatos 28 anos, a Seleção brasileira se tornou tetracampeã da Copa do Mundo de Futebol em uma partida emocionante contra a Itália, decidida nos pênaltis após o empate em 0 a 0 no tempo normal e na prorrogação.")

      link = '[ayrtonsenna.com.br](https://www.ayrtonsenna.com.br/28-anos-da-homenagem-na-final-da-copa-senna-aceleramos-juntos-o-tetra-e-nosso/)'
      st.markdown(link, unsafe_allow_html=True)

      st.write("A final da Copa do Mundo FIFA de 1998 foi disputada em 12 de julho no Stade de France, na cidade de Saint-Denis na França. A França derrotou o Brasil por 3-0 e sagrou-se campeã pela primeira vez.")
      link = '[Wikipédia](https://pt.wikipedia.org/wiki/Final_da_Copa_do_Mundo_FIFA_de_1998)'
      st.markdown(link, unsafe_allow_html=True)

    # --------------------------------------------------------------------------

    if st.checkbox('ANÁLISE DE GOLS'):

    # --------------------------------------------------------------------------

      # Total de gols por Nome
      df_gols_home = df.groupby(["Year","Stage","Home Team Name"])["Home Team Goals"].sum().reset_index()
      df_gols_home.columns = ["Year","Stage","Nome","Gols"]
      df_gols_away = df.groupby(["Year","Stage","Away Team Name"])["Away Team Goals"].sum().reset_index()
      df_gols_away.columns = ["Year","Stage","Nome","Gols"]
      df_gols = pd.concat([df_gols_home, df_gols_away],axis=0)
      df_gols = df_gols.groupby(["Year","Stage","Nome"])["Gols"].sum().reset_index()
      df_gols = df_gols.sort_values(by="Gols",ascending =False)
      df_gols["Gols"] = df_gols["Gols"].astype(int)
      
      uniqueCategories2 = makeUniqueCategories(df_gols['Stage'])
      input_dropdown2 = alt.binding_select(options=uniqueCategories2, name="Stage: ")
      selection2 = alt.selection_single(fields=['Stage'], bind=input_dropdown2)

      uniqueCategories3 = makeUniqueCategories(df_gols['Nome'])
      input_dropdown3 = alt.binding_select(options=uniqueCategories3, name="Nome: ")
      selection3 = alt.selection_single(fields=['Nome'], bind=input_dropdown3)

      st.write("O que queremos visualizar neste gráfico:")
      st.write(" - Quantos gols cada equipe marcou nas diferentes fases da Copa do Mundo FIFA?")
      st.write(" - Quantos gols cada equipe marcou na Final da Copa do Mundo FIFA?")

      mychart1 = alt.Chart(df_gols).encode(
        alt.X('Nome:N'),
        alt.Y('Gols:Q', sort='y'),
        alt.Color('Year:O', scale=alt.Scale(scheme='viridis')),
        tooltip=df_gols.columns.tolist(),
        column=alt.Column('Stage:N')
      )
     
      st.altair_chart(mychart1.mark_bar().add_selection(selection2).transform_filter(selection2).interactive(), use_container_width=True)

      st.write("O que queremos visualizar neste gráfico:")
      st.write(" - Quantos gols a seleção brasileira marcou nas diferentes fases da Copa do Mundo FIFA?")
      st.write(" - Quantos gols a seleção brasileira marcou na Final da Copa do Mundo FIFA?")

      mychart2 = alt.Chart(df_gols).encode(
        alt.X('Year:N'),
        alt.Y('Gols:Q', sort='y'),
        alt.Color('Year:O', scale=alt.Scale(scheme='viridis')),
        tooltip=df_gols.columns.tolist(),
        column=alt.Column('Stage:N')
      )
      
      st.altair_chart(mychart2.mark_bar().add_selection(selection3).add_selection(selection2).transform_filter(selection3).transform_filter(selection2).interactive(), use_container_width=True)

      arquivo_csv = "https://raw.githubusercontent.com/andremnunes/worldcup/c0848f0c80b075ad6f84ba1b927d6e7487ef90ea/dataset/WorldCups.csv"
      df_world_cups = pd.read_csv(arquivo_csv, sep=',', thousands='.')
      anos_vitoria=df_world_cups.loc[(df_world_cups['Winner'] == "Brazil")]['Year'].to_list()
      import datetime
      anos_vitoria.append(int(datetime.date.today().strftime("%Y")))

      st.write("O que queremos visualizar neste gráfico:")
      st.write(" - Quantos gols a seleção brasileira marcou nos anos em que foi campeã nas diferentes fases da Copa do Mundo FIFA?")
      st.write(" - Quantos gols a seleção brasileira marcou nos anos em que foi campeã na Final da Copa do Mundo FIFA?")

      mychart3 = alt.Chart(df_gols[df_gols['Year'].isin(anos_vitoria)]).encode(
        alt.X('Year:N'),
        alt.Y('Gols:Q', sort='y'),
        alt.Color('Year:O', scale=alt.Scale(scheme='viridis')),
        tooltip=df_gols.columns.tolist(),
        column=alt.Column('Stage:N', sort=alt.SortField("Stage", order="ascending"))  
      )
      st.altair_chart(mychart3.mark_bar().add_selection(selection2).add_selection(selection3).transform_filter(selection2).transform_filter(selection3).interactive(), use_container_width=True)

    # --------------------------------------------------------------------------

    if st.checkbox('GRÁFICO DE REDE'):

    # --------------------------------------------------------------------------

      st.write("O que queremos visualizar neste gráfico:")
      st.write(" - Quais as seleções que mais jogaram partidas?")
      st.write(" - Quais os times que tiveram uma curta participação em Copas do Mundo?")
      
      # https://github.com/Zsailer/nx_altair/blob/master/examples/nx_altair-tutorial.ipynb
      # https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_pandas_edgelist.html
      # https://towardsdatascience.com/customizing-networkx-graphs-f80b4e69bedf
      # https://infovis.fh-potsdam.de/tutorials/infovis7networks.html

      #x_column = st.selectbox('Selecione o atributo x', df.select_dtypes(include=['object']).columns, key = "x_grafico_de_rede")
      #y_column = st.selectbox('Selecione o atributo y', df.select_dtypes(include=['object']).columns, key = "y_grafico_de_rede")
      x_column = "Home Team Name"
      y_column = "Away Team Name"

      layout = st.selectbox('Layout', ['Layout Kamada Kawai','Layout Circular','Layout Aleatório','Layout Concha','Layout Primavera','Layout Spectral','Layout Espiral'], key = "layout_grafico_de_rede")

      G = nx.from_pandas_edgelist(df, source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())  

      density = nx.density(G)
      st.write(f"Informações: {translator.translate(nx.info(G), src='en', dest='pt').text}")
      st.write("Densidade da rede:", density)

      if layout == "Layout Circular":
        pos = nx.circular_layout(G)

      elif layout == "Layout Kamada Kawai":
        pos = nx.kamada_kawai_layout(G)

      elif layout == "Layout Aleatório":
        pos = nx.random_layout(G)

      elif layout == "Layout Concha":
        pos = nx.shell_layout(G)

      elif layout == "Layout Primavera":
        pos = nx.spring_layout(G)

      elif layout == "Layout Spectral":
        pos = nx.spectral_layout(G)

      elif layout == "Layout Espiral":
        pos = nx.spiral_layout(G)

      else:
        pos = nx.spring_layout(G)

      d = {v:v for v in list(G.nodes())}
      degrees = dict(G.degree(G.nodes()))
      between = nx.betweenness_centrality(G)
      nx.set_node_attributes(G, d, 'name')
      nx.set_node_attributes(G, degrees, 'degree') # save the degrees as a node attribute
      nx.set_node_attributes(G, between, 'between') # save the between as a node attribute
      mychart = nxa.draw_networkx(G=G, pos=pos, node_size='degree:Q', node_color='degree:N', node_tooltip=["name", "degree", "between"], cmap='viridis', linewidths=0,).properties(width=500, height=500)
      st.altair_chart(mychart, use_container_width=True)

      st.write("Os países ganhadores da Copa do Mundo FIFA: Argentina, Brasil, Inglaterra, França, Alemanha, Itália, Espanha e Uruguai, são também os que mais possuem partidas jogadas entre todos os times da Copa do Mundo.")
      st.write("Entre eles, os mais importantes são Brasil e Alemanha.")

      # ------------------------------------------------------------------------

      sorted_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True) # reverse sort of the degrees
      sorted_between = sorted(between.items(), key=lambda x: x[1], reverse=True) # reverse sort of the between

      # ------------------------------------------------------------------------

      df_sorted_degree = pd.DataFrame(sorted_degree, columns=['Nó','N_Vizinhos'])
    
      mychart = alt.Chart(df_sorted_degree.head(10)).mark_bar().encode(
      x='N_Vizinhos:Q',
      y=alt.Y('Nó:N', sort='-x'),
      color=alt.Color('N_Vizinhos:Q', scale=alt.Scale(scheme='viridis')),
      tooltip=df_sorted_degree.columns.tolist()
      ).interactive()

      #https://discuss.streamlit.io/t/how-to-display-a-table-and-its-plot-side-by-side-with-an-adjusted-height/30214
      data_container = st.container()
      with data_container:
        table, plot = st.columns(2)
        with table:
          st.table(df_sorted_degree.head(10))
        with plot:
          st.altair_chart(mychart, use_container_width=True)

      # ------------------------------------------------------------------------

      df_sorted_between = pd.DataFrame(sorted_between, columns=['Nó','N_Between'])

      mychart = alt.Chart(df_sorted_between.head(10)).mark_bar().encode(
      x='N_Between:Q',
      y=alt.Y('Nó:N', sort='-x'),
      color=alt.Color('N_Between:Q', scale=alt.Scale(scheme='viridis')),
      tooltip=df_sorted_between.columns.tolist()
      ).interactive()

      #https://discuss.streamlit.io/t/how-to-display-a-table-and-its-plot-side-by-side-with-an-adjusted-height/30214
      data_container = st.container()
      with data_container:
        table, plot = st.columns(2)
        with table:
          st.table(df_sorted_between.head(10))
        with plot:
          st.altair_chart(mychart, use_container_width=True)

      # ------------------------------------------------------------------------

      st.write("O que queremos visualizar neste gráfico:")
      st.write(" - Quais as seleções que já jogaram a Final da Copa do Mundo entre si?")

      G = nx.from_pandas_edgelist(df.loc[(df['Stage'] == "Final")], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      pos = nx.kamada_kawai_layout(G)
      d = {v:v for v in list(G.nodes())}
      degrees = dict(G.degree(G.nodes()))
      between = nx.betweenness_centrality(G)
      nx.set_node_attributes(G, d, 'name')
      nx.set_node_attributes(G, degrees, 'degree') # save the degrees as a node attribute
      nx.set_node_attributes(G, between, 'between') # save the between as a node attribute
      mychart = nxa.draw_networkx(G=G, pos=pos, node_size='degree:Q', node_color='degree:N', node_tooltip=["name", "degree", "between"], cmap='viridis', linewidths=0,).properties(title='Final', width=500, height=500)
      st.altair_chart(mychart, use_container_width=True)

      st.write("Jogar a Final da Copa do Mundo não necessariamente garante a vitória, mas ao contrário do senso comum que sugere o Brasil como maior jogador de Finais da Copa do Mundo, é a Alemanha que tem maior importância, seguida de Brasil e Itália.")

      # ------------------------------------------------------------------------

      st.write("O que queremos visualizar neste gráfico:")
      st.write(" - Quais as seleções que jogaram entre si nos anos em que o Brasil ganhou a Copa do Mundo FIFA?")
      st.write(" - Quão longe cada equipe chegou no campeonato?")

      #G = nx.from_pandas_edgelist(df.loc[(df['Stage'] == "Fase de grupos") & (df['Year'] == 1958)], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      G = nx.from_pandas_edgelist(df.loc[(df['Year'] == 1958)], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      pos = nx.circular_layout(G, scale=1, center=None, dim=2)
      d = {v:v for v in list(G.nodes())}
      degrees = dict(G.degree(G.nodes()))
      between = nx.betweenness_centrality(G)
      nx.set_node_attributes(G, d, 'name')
      nx.set_node_attributes(G, degrees, 'degree') # save the degrees as a node attribute
      nx.set_node_attributes(G, between, 'between') # save the between as a node attribute
      mychart1 = nxa.draw_networkx(G=G, pos=pos, node_size='degree:Q', node_color='degree:N', node_tooltip=["name", "degree", "between"], cmap='viridis', linewidths=0,).properties(title='1958', width=250, height=250)

      #G = nx.from_pandas_edgelist(df.loc[(df['Stage'] == "Fase de grupos") & (df['Year'] == 1962)], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      G = nx.from_pandas_edgelist(df.loc[(df['Year'] == 1962)], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      pos = nx.circular_layout(G, scale=1, center=None, dim=2)
      d = {v:v for v in list(G.nodes())}
      degrees = dict(G.degree(G.nodes()))
      between = nx.betweenness_centrality(G)
      nx.set_node_attributes(G, d, 'name')
      nx.set_node_attributes(G, degrees, 'degree') # save the degrees as a node attribute
      nx.set_node_attributes(G, between, 'between') # save the between as a node attribute
      mychart2 = nxa.draw_networkx(G=G, pos=pos, node_size='degree:Q', node_color='degree:N', node_tooltip=["name", "degree", "between"], cmap='viridis', linewidths=0,).properties(title='1962', width=250, height=250)

      #G = nx.from_pandas_edgelist(df.loc[(df['Stage'] == "Fase de grupos") & (df['Year'] == 1970)], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      G = nx.from_pandas_edgelist(df.loc[(df['Year'] == 1970)], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      pos = nx.circular_layout(G, scale=1, center=None, dim=2)
      d = {v:v for v in list(G.nodes())}
      degrees = dict(G.degree(G.nodes()))
      between = nx.betweenness_centrality(G)
      nx.set_node_attributes(G, d, 'name')
      nx.set_node_attributes(G, degrees, 'degree') # save the degrees as a node attribute
      nx.set_node_attributes(G, between, 'between') # save the between as a node attribute
      mychart3 = nxa.draw_networkx(G=G, pos=pos, node_size='degree:Q', node_color='degree:N', node_tooltip=["name", "degree", "between"], cmap='viridis', linewidths=0,).properties(title='1970', width=250, height=250)

      #G = nx.from_pandas_edgelist(df.loc[(df['Stage'] == "Fase de grupos") & (df['Year'] == 1994)], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      G = nx.from_pandas_edgelist(df.loc[(df['Year'] == 1994)], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      pos = nx.circular_layout(G, scale=1, center=None, dim=2)
      d = {v:v for v in list(G.nodes())}
      degrees = dict(G.degree(G.nodes()))
      between = nx.betweenness_centrality(G)
      nx.set_node_attributes(G, d, 'name')
      nx.set_node_attributes(G, degrees, 'degree') # save the degrees as a node attribute
      nx.set_node_attributes(G, between, 'between') # save the between as a node attribute
      mychart4 = nxa.draw_networkx(G=G, pos=pos, node_size='degree:Q', node_color='degree:N', node_tooltip=["name", "degree", "between"], cmap='viridis', linewidths=0,).properties(title='1994', width=250, height=250)

      #G = nx.from_pandas_edgelist(df.loc[(df['Stage'] == "Fase de grupos") & (df['Year'] == 2002)], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      G = nx.from_pandas_edgelist(df.loc[(df['Year'] == 2002)], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      pos = nx.circular_layout(G, scale=1, center=None, dim=2)
      d = {v:v for v in list(G.nodes())}
      degrees = dict(G.degree(G.nodes()))
      between = nx.betweenness_centrality(G)
      nx.set_node_attributes(G, d, 'name')
      nx.set_node_attributes(G, degrees, 'degree') # save the degrees as a node attribute
      nx.set_node_attributes(G, between, 'between') # save the between as a node attribute
      mychart5 = nxa.draw_networkx(G=G, pos=pos, node_size='degree:Q', node_color='degree:N', node_tooltip=["name", "degree", "between"], cmap='viridis', linewidths=0,).properties(title='2002', width=250, height=250)

      #G = nx.from_pandas_edgelist(df.loc[(df['Stage'] == "Fase de grupos") & (df['Year'] == 2022)], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      G = nx.from_pandas_edgelist(df.loc[(df['Year'] == 2022)], source=x_column, target=y_column, edge_attr=True, create_using=nx.Graph())
      pos = nx.circular_layout(G, scale=1, center=None, dim=2)
      d = {v:v for v in list(G.nodes())}
      degrees = dict(G.degree(G.nodes()))
      between = nx.betweenness_centrality(G)
      nx.set_node_attributes(G, d, 'name')
      nx.set_node_attributes(G, degrees, 'degree') # save the degrees as a node attribute
      nx.set_node_attributes(G, between, 'between') # save the between as a node attribute
      mychart6 = nxa.draw_networkx(G=G, pos=pos, node_size='degree:Q', node_color='degree:N', node_tooltip=["name", "degree", "between"], cmap='viridis', linewidths=0,).properties(title='2022', width=250, height=250)

      data_container = st.container()
      with data_container:
        plot1, plot2 = st.columns(2)
        with plot1:
          st.altair_chart(mychart1, use_container_width=True)
          st.write("Seleções mais importantes: Brasil, Alemanha, Suécia e França.")

          st.altair_chart(mychart3, use_container_width=True)
          st.write("Seleções mais importantes: Brasil, Alemanha, Itália e Uruguai.")

          st.altair_chart(mychart5, use_container_width=True)
          st.write("Seleções mais importantes: Alemanha, Coréia do Sul, Turquia e Brasil.")

        with plot2:
          st.altair_chart(mychart2, use_container_width=True)
          st.write("Seleções mais importantes: Yugoslavia, Chile e Brasil.")

          st.altair_chart(mychart4, use_container_width=True)
          st.write("Seleções mais importantes: Itália, Bulgária, Suécia e Brasil.")

          st.altair_chart(mychart6, use_container_width=True)
          st.write("Seleções mais importantes: Brasil ???")

      st.write("Os gráficos acima apenas confirmam quais são os países favoritos para jogar novamente uma final da Copa do Mundo e/ou avançar no torneio.")

      st.write("Quanto mais partidas jogadas maior o degree. Desta forma, o degree define o quão longe uma equipe chegou no campeonato.")

      # ------------------------------------------------------------------------

# ------------------------------------------------------------------------------

  if option == "Jogadores":
    
    if st.checkbox('WORD CLOUD'):
      categorical = [var for var in df.columns if df[var].dtype=='O']
      column = st.selectbox('Selecione a coluna', categorical)
      imagem = st.selectbox('Selecione a imagem', ['bola.png'])
      x = column.strip()
      try:
        text = " ".join(var for var in df[x])
        st.write("Existem {} palavras na combinação de todos os valores do atributo {}.".format(len(text), x))
        show_wordcloud2(text, imagem)
      except:
        st.write("Não foi possível gerar o gráfico. Tente outro atributo ou outra imagem.")
        pass  

# ------------------------------------------------------------------------------
## JULIAN - Código que entra direto na tela de Squads
#if option == "worldcup_squads.csv" and nome_dataset == "Convocações":
if option == "Convocacoes":

  # INICIO JULIAN - IDADE
  st.subheader("Pergunta: A média de idade dos jogadores nos times das seleções tem alguma influência nas primeiras 4 posições da competição?")

  st.write("A primeira coisa que precisamos são dos dados relacionados aos jogadores e suas idades (já mostrado no dataset acima), assim como precisamos das colocações das seleções nas Copas realizadas (dataset que será adicionado adiante).")

  # Preenche com 0 quando não tem valor de idade
  # df['Age'].dropna(how = 'all', inplace = True)  
  df['Age'].fillna(0, inplace = True)

  dfMeanAge = df
  dfMeanAge['Year'] = dfMeanAge['Year'].astype(str)
  dfMeanAge['Age'] = dfMeanAge['Age'].astype(int)

  dfMeanAge['SquadCountryName'] = dfMeanAge['ClubCountry']
  dfMeanAge['SquadCountryName'] = dfMeanAge['SquadCountryName']
  dfMeanAge['SquadCountryName'] = dfMeanAge['SquadCountryName'].replace('England','United Kingdom')
  dfMeanAge['SquadCountryName'] = dfMeanAge['SquadCountryName'].replace('USA','United States')
  dfMeanAge['SquadCountryName'] = dfMeanAge['SquadCountryName'].replace('Korea','Korea, Republic of')
  dfMeanAge['SquadCountryName'] = dfMeanAge['SquadCountryName'].replace('Germany FR','Germany')
  dfMeanAge['SquadCountryName'] = dfMeanAge['SquadCountryName'].replace('West Germany','Germany')  
  dfMeanAge['SquadCountryName'] = dfMeanAge['SquadCountryName'].replace('Czechoslovakia', 'Czechia')
  dfMeanAge['SquadCountryName'] = dfMeanAge['SquadCountryName'].replace('Soviet Union', 'Russian Federation')
  dfMeanAge['SquadCountryName'] = dfMeanAge['SquadCountryName'].replace('Russia', 'Russian Federation')
  dfMeanAge['SquadCountryName'] = dfMeanAge['SquadCountryName'].replace('Yugoslavia', 'Serbia')
  dfMeanAge['SquadCountryName'] = dfMeanAge['SquadCountryName'].replace('Korea Republic','Korea, Republic of')
  dfMeanAge['ClubCountry'] = [countries.get(country, 'U/C') for country in df_csv.SquadCountryName]

  # Data frame com idades médias para cada squad    
  dfMeanAge = dfMeanAge.groupby(['ClubCountry', 'Year', 'SquadCountryName'], as_index=False)['Age'].mean()

  st.dataframe(dfMeanAge)

  ####### PEGA OS GANHADORES ######
  ####### TODO: Precisa rever isso pois já existe o código e precisa externalizar em uma função
  ######################################################################

  arquivo_csv = ""https://raw.githubusercontent.com/andremnunes/worldcup/c0848f0c80b075ad6f84ba1b927d6e7487ef90ea/dataset/WorldCups.csv"
  df_csv = pd.read_csv(arquivo_csv, sep=',', thousands='.')    
  # Pré-processamento
  # =====================
  # https://en.wikipedia.org/wiki/2018_FIFA_World_Cup
  new_row = {'Year':2018, 
          'Country':'Russian Federation', 
          'Winner':'France', 
          'Runners-Up':'Croatia', 
          'Third':'Belgium', 
          'Fourth':'England', 
          'GoalsScored':169, 
          'QualifiedTeams':32, 
          'MatchesPlayed':64, 
          'Attendance':3031768}
  df_csv = df_csv.append(new_row, ignore_index=True)
 
  # https://stackoverflow.com/questions/41486867/duplicate-row-if-a-column-contains-any-special-character-in-python-pandas
  df_csv = df_csv.drop('Country', axis=1) \
        .join(df_csv['Country'] \
        .str \
        .split('/', expand=True) \
        .stack() \
        .reset_index(level=1, drop=True).rename('Country')) \
        .reset_index(drop=True)

  # Retira 1 2002
  df_csv = df_csv.drop_duplicates('Year')

  df_csv.replace(to_replace='Germany FR', value='Germany', inplace=True)
  df_csv['Ref_Country'] = df_csv['Country']
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('England','United Kingdom')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('USA','United States')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('Korea','Korea, Republic of')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('Germany FR','Germany')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('West Germany','Germany')  
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('Czechoslovakia', 'Czechia')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('Soviet Union', 'Russian Federation')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('Yugoslavia', 'Serbia')
  df_csv['Ref_Country'] = df_csv['Ref_Country'].replace('Korea Republic','Korea, Republic of')
  df_csv['Country_Code'] = [countries.get(country, 'U/C') for country in df_csv.Ref_Country]

  # PRIMEIRO LUGAR
  df_csv['Winner_Country'] = df_csv['Winner']
  df_csv['Winner_Country'] = df_csv['Winner_Country'].replace('England','United Kingdom')
  df_csv['Winner_Country'] = df_csv['Winner_Country'].replace('USA','United States')
  df_csv['Winner_Country'] = df_csv['Winner_Country'].replace('Korea','Korea, Republic of')
  df_csv['Winner_Country'] = df_csv['Winner_Country'].replace('Germany FR','Germany')
  df_csv['Winner_Country'] = df_csv['Winner_Country'].replace('West Germany','Germany')  
  df_csv['Winner_Country'] = df_csv['Winner_Country'].replace('Czechoslovakia', 'Czechia')
  df_csv['Winner_Country'] = df_csv['Winner_Country'].replace('Soviet Union', 'Russian Federation')
  df_csv['Winner_Country'] = df_csv['Winner_Country'].replace('Yugoslavia', 'Serbia')
  df_csv['Winner_Country'] = df_csv['Winner_Country'].replace('Korea Republic','Korea, Republic of')
  df_csv['Winner_Country'] = [countries.get(country, 'U/C') for country in df_csv.Winner_Country]
  
  # SEGUNDO LUGAR
  df_csv['Second_Country'] = df_csv['Runners-Up']
  df_csv['Second_Country'] = df_csv['Second_Country'].replace('England','United Kingdom')
  df_csv['Second_Country'] = df_csv['Second_Country'].replace('USA','United States')
  df_csv['Second_Country'] = df_csv['Second_Country'].replace('Korea','Korea, Republic of')
  df_csv['Second_Country'] = df_csv['Second_Country'].replace('Germany FR','Germany')
  df_csv['Second_Country'] = df_csv['Second_Country'].replace('West Germany','Germany')  
  df_csv['Second_Country'] = df_csv['Second_Country'].replace('Czechoslovakia', 'Czechia')
  df_csv['Second_Country'] = df_csv['Second_Country'].replace('Soviet Union', 'Russian Federation')
  df_csv['Second_Country'] = df_csv['Second_Country'].replace('Yugoslavia', 'Serbia')
  df_csv['Second_Country'] = df_csv['Second_Country'].replace('Korea Republic','Korea, Republic of')
  df_csv['Second_Country'] = [countries.get(country, 'U/C') for country in df_csv.Second_Country]

  # TERCEIRO LUGAR
  df_csv['Third_Country'] = df_csv['Third']
  df_csv['Third_Country'] = df_csv['Third_Country'].replace('England','United Kingdom')
  df_csv['Third_Country'] = df_csv['Third_Country'].replace('USA','United States')
  df_csv['Third_Country'] = df_csv['Third_Country'].replace('Korea','Korea, Republic of')
  df_csv['Third_Country'] = df_csv['Third_Country'].replace('Germany FR','Germany')
  df_csv['Third_Country'] = df_csv['Third_Country'].replace('West Germany','Germany')  
  df_csv['Third_Country'] = df_csv['Third_Country'].replace('Czechoslovakia', 'Czechia')
  df_csv['Third_Country'] = df_csv['Third_Country'].replace('Soviet Union', 'Russian Federation')
  df_csv['Third_Country'] = df_csv['Third_Country'].replace('Yugoslavia', 'Serbia')
  df_csv['Third_Country'] = df_csv['Third_Country'].replace('Korea Republic','Korea, Republic of')
  df_csv['Third_Country'] = [countries.get(country, 'U/C') for country in df_csv.Third_Country]

  # QUARTO LUGAR
  df_csv['Fourth_Country'] = df_csv['Fourth']
  df_csv['Fourth_Country'] = df_csv['Fourth_Country'].replace('England','United Kingdom')
  df_csv['Fourth_Country'] = df_csv['Fourth_Country'].replace('USA','United States')
  df_csv['Fourth_Country'] = df_csv['Fourth_Country'].replace('Korea','Korea, Republic of')
  df_csv['Fourth_Country'] = df_csv['Fourth_Country'].replace('Germany FR','Germany')
  df_csv['Fourth_Country'] = df_csv['Fourth_Country'].replace('West Germany','Germany')  
  df_csv['Fourth_Country'] = df_csv['Fourth_Country'].replace('Czechoslovakia', 'Czechia')
  df_csv['Fourth_Country'] = df_csv['Fourth_Country'].replace('Soviet Union', 'Russian Federation')
  df_csv['Fourth_Country'] = df_csv['Fourth_Country'].replace('Yugoslavia', 'Serbia')
  df_csv['Fourth_Country'] = df_csv['Fourth_Country'].replace('Korea Republic','Korea, Republic of')
  df_csv['Fourth_Country'] = [countries.get(country, 'U/C') for country in df_csv.Fourth_Country]

  df_csv['Year'] = df_csv['Year'].astype(str)
  df_csv = df_csv.dropna(how='all')
  df_wc = df_csv

  dfMerged = pd.merge(dfMeanAge, df_wc, on=["Year"], how='left')

  dfMerged.drop(['Country', 'Runners-Up', 'Third', 'Fourth', 'Winner', 'GoalsScored', 'QualifiedTeams', 'MatchesPlayed', 'Attendance', 'Ref_Country', 'Country_Code'], axis=1, inplace=True)
  dfMerged.rename(columns = {'ClubCountry':'SquadCountry'}, inplace = True)
  dfMerged['SquadCountry_Winner'] = np.where((dfMerged['SquadCountry'] != 'U/C') & (dfMerged['SquadCountry'] == dfMerged['Winner_Country']), True, False)
  dfMerged['SquadCountry_Second'] = np.where((dfMerged['SquadCountry'] != 'U/C') & (dfMerged['SquadCountry'] == dfMerged['Second_Country']), True, False)
  dfMerged['SquadCountry_Third'] = np.where((dfMerged['SquadCountry'] != 'U/C') & (dfMerged['SquadCountry'] == dfMerged['Third_Country']), True, False)
  dfMerged['SquadCountry_Fourth'] = np.where((dfMerged['SquadCountry'] != 'U/C') & (dfMerged['SquadCountry'] == dfMerged['Fourth_Country']), True, False)

  dfMerged['SquadCountry_Position'] = np.where((dfMerged['SquadCountry'] != 'U/C') & (dfMerged['SquadCountry_Winner'] == True), 1, 
    np.where((dfMerged['SquadCountry'] != 'U/C') & (dfMerged['SquadCountry_Second'] == True), 2, 
      np.where((dfMerged['SquadCountry'] != 'U/C') & (dfMerged['SquadCountry_Third'] == True), 3, 
        np.where((dfMerged['SquadCountry'] != 'U/C') & (dfMerged['SquadCountry_Fourth'] == True), 4, 0)
      )
    )
  )

  st.write("Abaixo está o dataset completo para a realização da análise que possui todas as informações necessárias, principalmente a ideia média de cada uma dos times das seleções em cada uma das copas, assim como a colocação de cada uma delas representada.")  
  st.dataframe(dfMerged)

  st.write("No gráfico abaixo foram plotados os boxplots de cada um dos anos para termos alguma dimensão da variação das idades médias em cada ano.")
  
  mychart0 = alt.Chart(dfMerged).mark_boxplot(extent='min-max').encode(
    x='Year:O',
    y='Age:Q',
  ).properties(width=300)
  
  st.altair_chart(mychart0, use_container_width=True)
  
  dfMerged4 = dfMerged[(dfMerged['SquadCountry_Winner'] == True) | (dfMerged['SquadCountry_Second'] == True) | (dfMerged['SquadCountry_Third'] == True) | (dfMerged['SquadCountry_Fourth'] == True)]
  dfMerged4['Age'] = dfMerged4['Age'].round(decimals = 1)
  dfMerged4['Color'] = np.where((dfMerged4['SquadCountry'] != 'U/C') & (dfMerged4['SquadCountry_Winner'] == True), '#FFD700', 
    np.where((dfMerged4['SquadCountry'] != 'U/C') & (dfMerged4['SquadCountry_Second'] == True), '#C0C0C0', 
      np.where((dfMerged4['SquadCountry'] != 'U/C') & (dfMerged4['SquadCountry_Third'] == True), '#CD7F32', 
        np.where((dfMerged4['SquadCountry'] != 'U/C') & (dfMerged4['SquadCountry_Fourth'] == True), 'orange', 0)
      )
    )
  )
  dfMerged4['Colocacao'] = np.where((dfMerged4['SquadCountry'] != 'U/C') & (dfMerged4['SquadCountry_Winner'] == True), '1° Lugar', 
    np.where((dfMerged4['SquadCountry'] != 'U/C') & (dfMerged4['SquadCountry_Second'] == True), '2° Lugar', 
      np.where((dfMerged4['SquadCountry'] != 'U/C') & (dfMerged4['SquadCountry_Third'] == True), '3° Lugar', 
        np.where((dfMerged4['SquadCountry'] != 'U/C') & (dfMerged4['SquadCountry_Fourth'] == True), '4° Lugar', 0)
      )
    )
  )

  st.write("Geramos um dataset que contém as informações mais detalhadas das colocações do primeiro ao quarto time em cada copa.")

  st.dataframe(dfMerged4)

  st.write("No gráfico abaixo foram pegos somente do primeiro ao quarto colocado para tentermos entender se nestas posições a idade média tem alguma influencia. Lembrando que o comparativo de valores é apenas entre os primeiros 4 colocados.")

  mychart1 = alt.Chart(dfMerged4).encode(
      alt.X('Year:O', axis=alt.Axis(tickCount=10, grid=True)),
      alt.Y('Age:O', sort='-y', axis=alt.Axis(tickCount=10, grid=True)),
      tooltip = dfMerged4.columns.tolist(),
      color = alt.Color('Colocacao', scale=alt.Scale(scheme='dark2')),
  )
  text = alt.Chart(dfMerged4).mark_text(dy=-10).encode(
      alt.X('Year:O'),
      alt.Y('Age:O', sort='-y'),
      text='Colocacao'
  )

  st.altair_chart(mychart1.mark_point(filled=True,size=100) + text, use_container_width=True)

  st.write("Abaixo outra maneira de visualizar os 4 primeiros colocados com relação a média de idade de seus jogadores.")

  mychart2 = alt.Chart(dfMerged4).encode(
      alt.X('Year:O', axis=alt.Axis(tickCount=10, grid=True)),
      alt.Y('Colocacao:O', sort='-y', axis=alt.Axis(tickCount=10, grid=True)),
      tooltip = dfMerged4.columns.tolist(),
      color = alt.Color('Colocacao', legend=None),
      size = alt.Size('Age:O', legend=None)
  )
  st.altair_chart(mychart2.mark_point(filled=True), use_container_width=True)
  
  st.write("Para tentar chegar a conclusão tentamos compilar os dados das seleções com suas idades junto com a variação de idade de cada ano comparativamente com o campeão de cada ano.")

  # deixa somente o campeão!
  dfMerged['Winner'] = np.where((dfMerged['SquadCountry'] != 'U/C') & (dfMerged['SquadCountry'] == dfMerged['Winner_Country']), '🏆' + dfMerged['SquadCountry'], '')
  dfMerged['Second'] = np.where((dfMerged['SquadCountry'] != 'U/C') & (dfMerged['SquadCountry'] == dfMerged['Second_Country']), '🥈' + dfMerged['SquadCountry'], '')
  dfMerged['Third'] = np.where((dfMerged['SquadCountry'] != 'U/C') & (dfMerged['SquadCountry'] == dfMerged['Third_Country']), '🥉' + dfMerged['SquadCountry'], '')
  dfMerged['BR'] = np.where((dfMerged['SquadCountry'] == 'BRA') & (dfMerged['SquadCountry'] != dfMerged['Winner_Country']) & (dfMerged['SquadCountry'] != dfMerged['Second_Country']) & (dfMerged['SquadCountry'] != dfMerged['Third_Country']), '🎯BR' , '')

  st.dataframe(dfMerged)

  mychartFinal = alt.Chart(dfMerged).encode(
    alt.X('Year:O', axis=alt.Axis(title='Ano da Copa do Mundo')),
    alt.Y('Age:Q', sort='-y', axis=alt.Axis(title='Média de Idade das Seleçoes'), scale=alt.Scale(domain=[6, 39])),
    # color=alt.value('#9A1032'),
    color=alt.value('#A5A2A0')
  )

  textFinalBR = alt.Chart(dfMerged).encode(
      alt.X('Year:O'),
      alt.Y('Age:Q', sort='-y', scale=alt.Scale(domain=[6, 39])),
      text='BR:N',
      color=alt.value('#FF2A23'),
  )

  textSecondFinal = alt.Chart(dfMerged).encode(
      alt.X('Year:O'),
      alt.Y('Age:Q', sort='-y', scale=alt.Scale(domain=[6, 39])),
      text='Second:N',
      color=alt.value('#FFA828'),
  )

  textThirdFinal = alt.Chart(dfMerged).encode(
      alt.X('Year:O'),
      alt.Y('Age:Q', sort='-y', scale=alt.Scale(domain=[6, 39])),
      text='Third:N',
      color=alt.value('#FFA828'),
  )

  textFinal = alt.Chart(dfMerged).encode(
      alt.X('Year:O'),
      alt.Y('Age:Q', sort='-y', scale=alt.Scale(domain=[6, 39])),
      text='Winner:N',
      color=alt.value('#FFA828'),
  )

  dotFinal = alt.Chart(dfMerged).encode(
      alt.X('Year:O'),
      alt.Y('Age:Q', sort='-y', scale=alt.Scale(domain=[6, 39])),
      tooltip = dfMerged.columns.tolist(),
      color=alt.value('#bc123a'),
  ) 
  
  st.altair_chart(
    mychartFinal.mark_boxplot(extent='min-max', size=25) 
    + textFinalBR.mark_text(dx=15, align='left', fontSize=9, fontWeight=400, opacity=0.8)
    + textSecondFinal.mark_text(dx=15, align='left', fontSize=11, fontWeight=600, opacity=0.8)  
    + textThirdFinal.mark_text(dx=15, align='left', fontSize=10, fontWeight=500, opacity=0.8)  
    + textFinal.mark_text(dx=15, align='left', fontSize=12, fontWeight=700, opacity=0.8)    
    + dotFinal.mark_point(filled=False,size=100)
    , use_container_width=True
  )

  st.write("No gráfico abaixo foi retirado a ano de 1930 por causa de falta de dados e para melhorar a visualização no eixo y (Idade).")

  # Deleta 1930
  dfMergerWithout1930 = dfMerged[dfMerged['Year'] != '1930']

  # st.dataframe(dfMergerWithout1930)

  mychartFinal = alt.Chart(dfMergerWithout1930).encode(
    alt.X('Year:O', axis=alt.Axis(title='Ano da Copa do Mundo')),
    alt.Y('Age:Q', sort='-y', axis=alt.Axis(title='Média de Idade das Seleçoes'), scale=alt.Scale(domain=[12, 39])),
    # color=alt.value('#9A1032'),
    color=alt.value('#A5A2A0')
  )

  textFinalBR = alt.Chart(dfMergerWithout1930).encode(
      alt.X('Year:O'),
      alt.Y('Age:Q', sort='-y', scale=alt.Scale(domain=[12, 39])),
      text='BR:N',
      color=alt.value('#FF2A23'),
  )

  textSecondFinal = alt.Chart(dfMergerWithout1930).encode(
      alt.X('Year:O'),
      alt.Y('Age:Q', sort='-y', scale=alt.Scale(domain=[12, 39])),
      text='Second:N',
      color=alt.value('#FFA828'),
  )

  textThirdFinal = alt.Chart(dfMergerWithout1930).encode(
      alt.X('Year:O'),
      alt.Y('Age:Q', sort='-y', scale=alt.Scale(domain=[12, 39])),
      text='Third:N',
      color=alt.value('#FFA828'),
  )

  textFinal = alt.Chart(dfMergerWithout1930).encode(
      alt.X('Year:O'),
      alt.Y('Age:Q', sort='-y', scale=alt.Scale(domain=[12, 39])),
      text='Winner:N',
      color=alt.value('#FFA828'),
  )

  dotFinal = alt.Chart(dfMergerWithout1930).encode(
      alt.X('Year:O'),
      alt.Y('Age:Q', sort='-y', scale=alt.Scale(domain=[12, 39])),
      tooltip = dfMergerWithout1930.columns.tolist(),
      color=alt.value('#bc123a'),
  ) 
  
  st.altair_chart(
    mychartFinal.mark_boxplot(extent='min-max', size=25) 
    + textFinalBR.mark_text(dx=15, align='left', fontSize=9, fontWeight=400, opacity=0.8)
    + textSecondFinal.mark_text(dx=15, align='left', fontSize=11, fontWeight=600, opacity=0.8)  
    + textThirdFinal.mark_text(dx=15, align='left', fontSize=10, fontWeight=500, opacity=0.8)  
    + textFinal.mark_text(dx=15, align='left', fontSize=12, fontWeight=700, opacity=0.8)    
    + dotFinal.mark_point(filled=False,size=100)
    , use_container_width=True
  )

  st.subheader("Pontos Interessantes")

  st.write("- Se utilizamos os dados de médias de idade das seleções que foram campeãs na Copa do Mundo de Futebol é muito pouco provável que Argentina seja campeã este ano, já que ela é praticamente um outlier desta ano com uma média 33.25 anos, sendo a maior média de idade do campeonato, e visualizando o histórico das copas é possível observar que a regra é que a média de idade da seleção campeão esteja próximo ao intervalo interquartílico desde 1978.")
  st.write("- Somente nas 2 primeras copas (1930 e 1934) uma seleção com o máximo de média de idade foi campeão, sendo que desde então isso nunca mais aconteceu.")
  st.write("- Nunca um time com a menor média de idade dos jogadores ficou entre os 3 primeros, já o time mais velho teve 2 primeiros lugares (1930, 1934), 2 segundo (1938, 1958) e 1 terceiro lugar (1966), mas interessante mencionar que após 1966 os extremos de idade média não ficaram entre os 4 primeiros.")
  st.write("- Importante mencionar que desde 1978 somente a ARGENTINA foi campeão (1986) com uma média bem fora do intervalo interquartílico.")

  st.subheader("Conclusão")
  st.write("De maneita geral é possível afirmar que as seleções mais bem posicionadas tendem a ficar com a média de idade mais próxima a média de idade de todas as seleções participantes, portante nem as seleções com médias mais baixas ou altas parecem ter mais chances de vencer, por isso acreditamos que a palavra chave para a média de idade é EQUILIBRIO, tendo como ideia central a mistura de jogadores mais experientes com jogadores que podem estar começando.")

  # FIM JULIAN - IDADE

if option == "ClubeOrigem":

    # INICIO DO CÓDIGO DO ANDRÉ
    st.subheader("Análise da evolução dos clubes de origem dos jogadores convocados")
    st.write("Como se deu a evolução ao longo das Copas do percentual dos jogadores que jogam em clubes do seu próprio país ou em clubes estrangeiros?")

    data_complete = ajustaCodeCountryISO(df_csv, 'Country')
    data_complete['joga_proprio_pais'] = data_complete.apply(lambda x : 'Yes' if x['Country'] == x['ClubCountry'] else 'No', axis=1)
    
    data_complete['total_players'] = 0
    dados_final = data_complete.groupby(["Country", "Country_Code", "Year", "joga_proprio_pais"])["total_players"].count().reset_index()
    dados_final['percent']= dados_final['total_players'] / dados_final.groupby(["Country", "Country_Code", "Year"])['total_players'].transform('sum')

    st.write("Tabela com os dados que estão exibidos no gráfico abaixo")
    df_joga_dentro_proprio_pais = dados_final.loc[dados_final['joga_proprio_pais'] == 'Yes']
    #st.dataframe(df_joga_dentro_proprio_pais)
    novaExibicaoDataFrame(data_complete)

    fig = px.choropleth(df_joga_dentro_proprio_pais.sort_values('Year'), locations="Country_Code",
                        color="percent",
                        hover_name="Country",
                        color_continuous_scale=px.colors.sequential.Plasma[::-1],
                        animation_frame='Year')

    fig.show()

    fig.update_geos(fitbounds = "locations", visible = True)
    st.plotly_chart(fig, use_container_width=True)

    st.write("Onde estão concentrados os grandes jogadores do planeta? Esta concentração mudou ao longo das Copas?")
    data_complete2 = df_csv.copy()

    data_complete2 = ajustaCodeCountryISO(data_complete2, 'ClubCountry')

    st.write("Dados agrupados por países do clube de origem do jogador ")
    dados_final2 = data_complete2.groupby(["ClubCountry", "ClubCountry_Code", "Year"])["total_players"].count().reset_index()
    novaExibicaoDataFrame(dados_final2)

    #st.write("Dados percentuais")
    #dados_final2['%'] = 100 * dados_final2.groupby(["ClubCountry", "Year"])['total_players'].transform('sum')/dados_final2.groupby(["Year"])["total_players"].transform('sum') 

    st.write("O gráfico da evolução ao longo das copas da quantidade de jogadores por país do clube de origem")
    fig = px.choropleth(dados_final2.sort_values('Year'), locations="ClubCountry_Code",
                        color="total_players",
                        hover_name="ClubCountry",
                        color_continuous_scale=px.colors.sequential.Plasma[::-1],
                        animation_frame='Year')

    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    
    if st.checkbox('Marque para exibir uma nuvem de palavra dos clubes que tiveram mais jogadores convocados ao longo das Copas'):
      #categorical = [var for var in df_csv.columns if df_csv[var].dtype=='O']
      #column = st.selectbox('Selecione a coluna', categorical)
      #imagem = st.selectbox('Selecione a imagem', ['bola.png'])

      filtro = ['Mundo', 'Brasil']
      selecao_filtro = st.selectbox('Selecione o filtro', filtro)
      column = 'Club'

      if selecao_filtro == 'Brasil':
        imagem = 'mapa_brasil.png'
        imagem = 'bola.png'
        df_cloud = df_csv.loc[df_csv['ClubCountry'] == 'Brazil']
      else:
        imagem = 'bola.png'
        df_cloud = df_csv


      x = column.strip()

      try:
        text = " ".join(var for var in df_cloud[x])
        st.write("Existem {} palavras na combinação de todos os valores do atributo {}.".format(len(text), x))
        show_wordcloud2(text, imagem)
      except:
        st.write("Não foi possível gerar o gráfico. Tente outro atributo ou outra imagem.")
        pass


    # FIM DO CÓDIGO DO ANDRÉ
