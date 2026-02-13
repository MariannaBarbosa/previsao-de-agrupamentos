
import streamlit as st
import pandas as pd
import joblib


scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
kmeans = joblib.load('kmeans.pkl')


st.title('Grupos de Interesse para Marketing')
st.write("""
         Neste projeto, aplicamos o algoritmo de clusterização K-means para identificar e prever agrupamentos de interesses de usuários, com o objetivo de direcionar campanhas de marketing de forma mais eficaz.
         Através dessa análise, conseguimos segmentar o público em bolhas de interesse, permitindo a criação de campanhas personalizadas e mais assertivas, com base nos padrões de comportamento e preferências de cada grupo.
         """)


up_file = st.file_uploader("Escola um arquivo csv para realizar a previsão.", type='csv')



def processar_prever (df):

    encoded_sexo = encoder.transform(df[['sexo']])
    encoded_df = pd.DataFrame(encoded_sexo, columns=encoder.get_feature_names_out(['sexo']))
    dados = pd.concat([df, encoded_df], axis=1).drop(columns='sexo')

    dados_escalados = scaler.transform(dados)
    cluster = kmeans.predict(dados_escalados)

    return cluster


# st.download_button(label='Baixe aqui:', data='oie rapariguinha', file_name='dados.csv', mime='text/csv')


if up_file is not None:
    st.write("""
                ### Descrição dos Grupos:
                - **Grupo 0** é focado em um público jovem com forte interesse em moda, música e aparência.
                - **Grupo 1** está muito associado a esportes, especialmente futebol americano, basquete e atividades culturais como banda e rock.
                - **Grupo 2** é mais equilibrado, com interesses em música, dança, e moda.
             """)
    
    df = pd.read_csv(up_file)
    
    cluster = processar_prever(df)

    df.insert(0, column='Grupo', value=cluster) 
    st.write('Visualização dos resultados (10 primeiros registros):')
    st.write(df.head(10))

    csv = df.to_csv(index=False)
    st.download_button(label='Resultados completos', data=csv, file_name='resultado_agrupamento.txt', mime='text/csv')