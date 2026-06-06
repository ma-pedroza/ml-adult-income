import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Predição de Renda",
    page_icon="💹",
    layout="centered"
)

modelo = joblib.load("modelo_final.joblib")
scaler = joblib.load("scaler.joblib")
colunas = joblib.load("colunas_modelo.joblib")

num_cols = ["age", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]

educacao_num_map = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}

escolaridades = sorted([
    coluna.replace("education_", "")
    for coluna in colunas
    if coluna.startswith("education_")
])

paises = sorted([
    coluna.replace("native-country_", "")
    for coluna in colunas
    if coluna.startswith("native-country_")
])

ocupacoes = sorted([
    coluna.replace("occupation_", "")
    for coluna in colunas
    if coluna.startswith("occupation_")
])

estados_civis = sorted([
    coluna.replace("marital-status_", "")
    for coluna in colunas
    if coluna.startswith("marital-status_")
])

relacionamentos = sorted([
    coluna.replace("relationship_", "")
    for coluna in colunas
    if coluna.startswith("relationship_")
])

classes_trabalho = sorted([
    coluna.replace("workclass_", "")
    for coluna in colunas
    if coluna.startswith("workclass_")
])

racas = sorted([
    coluna.replace("race_", "")
    for coluna in colunas
    if coluna.startswith("race_")
])

st.title("💹 Predição de Renda")
st.markdown(
    """
    Este aplicativo utiliza um modelo de **Machine Learning** treinado com o dataset
    **Adult Income** para prever se uma pessoa possui renda anual **maior ou menor que 50K**.
    """
)

st.info(
    """
    ℹ️ Algumas categorias aparecem em inglês porque foram mantidas no formato original do dataset Adult Income.
    Essa decisão foi tomada para preservar a consistência entre os dados de treinamento e as entradas utilizadas pelo modelo.
    """
)

st.divider()

st.subheader("📋 Dados da pessoa")

col1, col2 = st.columns(2)

with col1:
    idade = st.number_input(
        "Idade",
        min_value=18,
        max_value=100,
        value=30,
    )

    horas = st.number_input(
        "Horas trabalhadas por semana",
        min_value=1,
        max_value=100,
        value=40,
    )

    capital_gain = st.number_input(
        "Ganho de capital",
        min_value=0,
        value=0,
        help="Ganhos financeiros provenientes de investimentos, venda de ações, imóveis ou outros ativos."
    )

    genero = st.selectbox(
        "Gênero",
        ["Masculino", "Feminino"],
    )

with col2:
    educacao_label = st.selectbox(
        "Escolaridade",
        escolaridades,
    )

    estado_civil = st.selectbox(
        "Estado civil",
        estados_civis,
    )

    trabalho = st.selectbox(
        "Classe de trabalho",
        classes_trabalho,
        help="Categoria do vínculo profissional da pessoa."
    )

    ocupacao = st.selectbox(
        "Ocupação",
        ocupacoes,
        help="Função ou área profissional exercida pela pessoa."
    )

capital_loss = st.number_input(
    "Perda de capital",
    min_value=0,
    value=0,
    help="Perdas financeiras provenientes de investimentos ou venda de ativos."
)

relacionamento = st.selectbox(
    "Relacionamento",
    relacionamentos,
)

raca = st.selectbox(
    "Raça",
    racas,
)

pais = st.selectbox(
    "País de origem",
    paises,
)

st.divider()

if st.button("🔎 Prever renda", use_container_width=True):
    educacao = educacao_label
    educacao_num = educacao_num_map.get(educacao_label, 10)

    entrada = pd.DataFrame(0, index=[0], columns=colunas)

    entrada["age"] = idade
    entrada["educational-num"] = educacao_num
    entrada["capital-gain"] = capital_gain
    entrada["capital-loss"] = capital_loss
    entrada["hours-per-week"] = horas
    entrada["gender"] = 1 if genero == "Masculino" else 0

    entrada["has_capital_gain"] = 1 if capital_gain > 0 else 0
    entrada["has_capital_loss"] = 1 if capital_loss > 0 else 0

    campos_categoricos = {
        f"workclass_{trabalho}": 1,
        f"education_{educacao}": 1,
        f"marital-status_{estado_civil}": 1,
        f"occupation_{ocupacao}": 1,
        f"relationship_{relacionamento}": 1,
        f"race_{raca}": 1,
        f"native-country_{pais}": 1,
    }

    for coluna, valor in campos_categoricos.items():
        if coluna in entrada.columns:
            entrada[coluna] = valor

    entrada[num_cols] = scaler.transform(entrada[num_cols])

    predicao = modelo.predict(entrada)[0]
    probabilidades = modelo.predict_proba(entrada)[0]

    st.subheader("📊 Resultado da predição")

    if predicao == 1:
        st.success("Renda prevista: maior que 50K")
    else:
        st.info("Renda prevista: menor ou igual a 50K")

    st.metric("Probabilidade de renda >50K", f"{probabilidades[1] * 100:.2f}%")
    st.metric("Probabilidade de renda <=50K", f"{probabilidades[0] * 100:.2f}%")

    st.caption(
        "Esta predição é uma estimativa baseada em padrões históricos do dataset Adult Income "
        "e não deve ser interpretada como decisão absoluta."
    )