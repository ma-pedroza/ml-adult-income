import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Predição de Renda",
    page_icon="💰",
    layout="centered"
)

modelo = joblib.load("modelo_final.joblib")
scaler = joblib.load("scaler.joblib")
colunas = joblib.load("colunas_modelo.joblib")

num_cols = ["age", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]

educacao_map = {
    "Ensino médio": ("HS-grad", 9),
    "Alguma faculdade": ("Some-college", 10),
    "Técnico/Associado": ("Assoc-voc", 11),
    "Bacharelado": ("Bachelors", 13),
    "Mestrado": ("Masters", 14),
    "Doutorado": ("Doctorate", 16),
}

st.title("💰 Predição de Renda")
st.markdown(
    """
    Este aplicativo utiliza um modelo de **Machine Learning** treinado com o dataset
    **Adult Income** para prever se uma pessoa possui renda anual **maior ou menor que 50K**.
    """
)

st.divider()

st.subheader("📋 Dados da pessoa")

col1, col2 = st.columns(2)

with col1:
    idade = st.number_input("Idade", min_value=18, max_value=100, value=30)
    horas = st.number_input("Horas trabalhadas por semana", min_value=1, max_value=100, value=40)
    capital_gain = st.number_input("Ganho de capital", min_value=0, value=0)
    genero = st.selectbox("Gênero", ["Masculino", "Feminino"])

with col2:
    educacao_label = st.selectbox("Escolaridade", list(educacao_map.keys()))
    estado_civil = st.selectbox(
        "Estado civil",
        ["Never-married", "Married-civ-spouse", "Separated", "Widowed"]
    )
    trabalho = st.selectbox(
        "Classe de trabalho",
        ["Private", "Local-gov", "State-gov", "Self-emp-inc", "Self-emp-not-inc"]
    )
    ocupacao = st.selectbox(
        "Ocupação",
        ["Prof-specialty", "Exec-managerial", "Craft-repair", "Sales", "Other-service", "Tech-support"]
    )

capital_loss = st.number_input("Perda de capital", min_value=0, value=0)

relacionamento = st.selectbox(
    "Relacionamento",
    ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried"]
)

raca = st.selectbox(
    "Raça",
    ["White", "Black", "Asian-Pac-Islander", "Other"]
)

pais = st.selectbox(
    "País de origem",
    ["United-States", "Mexico", "Canada", "Germany", "Philippines"]
)

st.divider()

if st.button("🔎 Prever renda", use_container_width=True):
    educacao, educacao_num = educacao_map[educacao_label]

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