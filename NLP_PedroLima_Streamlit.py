import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
import gensim
from gensim.models import word2vec
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
import spacy
from collections import Counter
import pickle


st.title('Projeto NLP Pedro Lima - Resposta após clusterização do feedback')

user_input = st.selectbox(
    'Por favor, conte-nos mais sobre o motivo da sua insatisfação:',
     pd.Series(['Não estou conseguindo cadastrar meus pontos de fidelidade',
'Não me ajudou em nada!',
'Péssimo o atendimento pelo robô',
'Não tem a informação que eu preciso',
'Quero conversar com um atendente humano',
'Fiz um pedido mas ainda não foi entregue',
'Não solucionou o meu problema',
'Não tem atendente disponível',
'Não deu solução para meu problema,'
'Não consegui sanar minha dúvida',
'Não consegui finalizar minha compra',
'Meu problema não foi resolvido',
'Não fui atendida',
'Quero consultar meus pontos no site mas não consigo']))



path_arquivo_modelo_kmeans = 'pickle_kmeans_model.plk'
with open(path_arquivo_modelo_kmeans, 'rb') as file:  
    modelo_kmeans_carregado = pickle.load(file)

path_dicionario_embeddings = 'pickle_dicionario_embeddings.plk'
with open(path_dicionario_embeddings, 'rb') as file:  
    dicionario_embeddings = pickle.load(file)

embedding_frase_selecionada = dicionario_embeddings[user_input]

embedding_frase_selecionada = pd.DataFrame(embedding_frase_selecionada.reshape(1,100))

cluster_identificado = modelo_kmeans_carregado.predict(embedding_frase_selecionada)[0]

respostas_dict = {0:"Pedimos desculpas pelos seus problemas no site. Salvamos as informações referentes aos seu pedido e em breve retornaremos com a solução",
    1:"Pedimos desculpas por não ter resolvido o seu problema. Um atendente tem os detalhes do seu problema e entrará em contato.",
    2:"Pedimos desculpas pelo atendimento. Um humano entrará em contato com você em breve.",
    3:"Pedimos desculpas por não termos sanado as suas dúvidas. Entraremos em contato para esclarecê-las para você em breve."}
    
print(respostas_dict[cluster_identificado])
st.write(respostas_dict[cluster_identificado])