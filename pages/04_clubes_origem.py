data_complete = pd.read_csv('../squads.csv', sep=';', engine='python', encoding='utf8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
import pycountry

countries = {}
for Country in pycountry.countries:
  countries[Country.name] = Country.alpha_3

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

data_complete = ajustaCodeCountryISO(data_complete, 'Country')

data_complete['joga_proprio_pais'] = data_complete.apply(lambda x : 'Yes' if x['Country'] == x['ClubCountry'] else 'No', axis=1)
data_complete['total_players'] = 0
dados_final = data_complete.groupby(["Country", "Country_Code", "Year", "joga_proprio_pais"])["total_players"].count().reset_index()
dados_final['percent']= dados_final['total_players'] / dados_final.groupby(["Country", "Country_Code", "Year"])['total_players'].transform('sum')
df_joga_dentro_proprio_pais = dados_final.loc[dados_final['joga_proprio_pais'] == 'Yes']

fig = px.choropleth(df_joga_dentro_proprio_pais.sort_values('Year'), locations="Country_Code",
                    color="percent",
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Plasma[::-1],
                    animation_frame='Year')

fig.update_geos(fitbounds = "locations", visible = True)
fig.show()

data_complete2 = data_complete.copy()
data_complete2 = ajustaCodeCountryISO(data_complete2, 'ClubCountry')
dados_final2 = data_complete2.groupby(["ClubCountry", "ClubCountry_Code", "Year"])["total_players"].count().reset_index()
dados_final2['%'] = 100 * dados_final2.groupby(["ClubCountry", "Year"])['total_players'].transform('sum')/dados_final2.groupby(["Year"])["total_players"].transform('sum') 

fig = px.choropleth(dados_final2.sort_values('Year'), locations="ClubCountry_Code",
                    color="%",
                    hover_name="ClubCountry",
                    color_continuous_scale=px.colors.sequential.Plasma[::-1],
                    animation_frame='Year')

fig.show()
