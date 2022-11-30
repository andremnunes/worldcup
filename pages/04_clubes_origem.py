import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import pandas as pd
import numpy as np
import pycountry
from pathlib import Path
import streamlit as st
import streamlit_book as stb

stb.set_chapter_config(path='pages/01_resultado_copa_mundo.py')

current_path = Path(__file__).parent.absolute()
st.write(current_path)
df_csv = pd.read_csv('dataset/worldcup_squads.csv', sep=',', engine='python', encoding='utf8')

countries = {}
for Country in pycountry.countries:
    countries[Country.name] = Country.alpha_3
        
def ajustaCodeCountryISO(df, campo, countries):
  
    df[campo + 'ISO'] = df[campo]
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Ivory Coast', 'Côte d\'Ivoire')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Zaire', 'Congo, The Democratic Republic of the')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Bolivia', 'Bolivia, Plurinational State of')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Iran', 'Iran, Islamic Republic of')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Russian', 'Russian Federation')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Scotland', 'United Kingdom')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('England', 'United Kingdom')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Wales', 'United Kingdom')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Northern Ireland', 'United Kingdom')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('USA', 'United States')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Korea', 'Korea, Republic of')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Germany FR', 'Germany')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('East Germany', 'Germany')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('West Germany', 'Germany')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Czechoslovakia', 'Czechia')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Soviet Union', 'Russian Federation')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Yugoslavia', 'Serbia')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Federal Republic of Yugoslavia', 'Serbia')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Kingdom of Yugoslavia', 'Serbia')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Socialist Federal Republic of Yugoslavia', 'Serbia')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('Korea Republic', 'Korea, Republic of')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('South Korea', 'Korea, Republic of')
    df[campo + 'ISO'] = df[campo + 'ISO'].replace('North Korea', 'Korea, Democratic People\'s Republic of')

    df[campo + '_Code'] = [countries.get(country, 'Unknown code') for country in df[campo + 'ISO']]

    return (df)


st.subheader("Análise da evolução dos clubes de origem dos jogadores convocados")
st.write(
    "Como se deu a evolução ao longo das Copas do percentual dos jogadores que jogam em clubes do seu próprio país ou em clubes estrangeiros?")

data_complete = ajustaCodeCountryISO(df_csv, 'Country', countries)
data_complete['joga_proprio_pais'] = data_complete.apply(lambda x: 'Yes' if x['Country'] == x['ClubCountry'] else 'No',
                                                         axis=1)
st.dataframe(data_complete)

data_complete['total_players'] = 0
dados_final = data_complete.groupby(["Country", "Country_Code", "Year", "joga_proprio_pais"])[
    "total_players"].count().reset_index()
dados_final['percent'] = dados_final['total_players'] / dados_final.groupby(["Country", "Country_Code", "Year"])[
    'total_players'].transform('sum')

st.write("Dados percentuais")
df_joga_dentro_proprio_pais = dados_final.loc[dados_final['joga_proprio_pais'] == 'Yes']
st.dataframe(df_joga_dentro_proprio_pais)

fig = px.choropleth(df_joga_dentro_proprio_pais.sort_values('Year'), locations="Country_Code",
                    color="percent",
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Plasma[::-1],
                    animation_frame='Year')

fig.show()

# fig.update_geos(fitbounds = "locations", visible = True)
st.plotly_chart(fig, use_container_width=True)

st.write("Onde estão concentrados os grandes jogadores do planeta? Esta concentração mudou ao longo das Copas?")
data_complete2 = df_csv.copy()
st.dataframe(data_complete2)

data_complete2 = ajustaCodeCountryISO(data_complete2, 'ClubCountry', countries)

st.write("Dados agrupados por clube, ano ")
dados_final2 = data_complete2.groupby(["ClubCountry", "ClubCountry_Code", "Year"])[
    "total_players"].count().reset_index()
st.dataframe(dados_final2)

st.write("Dados percentuais")
dados_final2['%'] = 100 * dados_final2.groupby(["ClubCountry", "Year"])['total_players'].transform('sum') / \
                    dados_final2.groupby(["Year"])["total_players"].transform('sum')
st.dataframe(dados_final2)

st.write("O gráfico da evolução ao longo das copas")
fig = px.choropleth(dados_final2.sort_values('Year'), locations="ClubCountry_Code",
                    color="%",
                    hover_name="ClubCountry",
                    color_continuous_scale=px.colors.sequential.Plasma[::-1],
                    animation_frame='Year')

fig.show()
st.plotly_chart(fig, use_container_width=True)

if st.checkbox(
        'Marque para exibir uma nuvem de palavra dos clubes que tiveram mais jogadores convocados ao longo das Copas'):
    # categorical = [var for var in df_csv.columns if df_csv[var].dtype=='O']
    # column = st.selectbox('Selecione a coluna', categorical)
    # imagem = st.selectbox('Selecione a imagem', ['bola.png'])

    column = 'Club'
    imagem = 'bola.png'

    x = column.strip()

    try:
        text = " ".join(var for var in df_csv[x])
        st.write("Existem {} palavras na combinação de todos os valores do atributo {}.".format(len(text), x))
        show_wordcloud2(text, imagem)
    except:
        st.write("Não foi possível gerar o gráfico. Tente outro atributo ou outra imagem.")
        pass
