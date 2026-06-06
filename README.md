# Predição de Renda com Machine Learning - Adult Income

## Integrantes

* Matheus Gomes Pedroza (RA: 1998912)
* Guilherme Dorce de Britto (RA: 1991866)
* Rodrigo Shinji Yamashita (RA: 2001443)

---

## Descrição do Problema

A previsão de renda é um problema amplamente estudado na área de Ciência de Dados e Machine Learning. Neste projeto, foi utilizado o dataset Adult Income para desenvolver um modelo capaz de prever se uma pessoa possui renda anual superior a US$ 50.000 com base em características demográficas, educacionais e profissionais.

---

## Objetivo do Projeto

Desenvolver, avaliar e disponibilizar um modelo de Machine Learning capaz de classificar indivíduos em duas categorias:

* Renda anual menor ou igual a US$ 50.000;
* Renda anual superior a US$ 50.000.

Além disso, disponibilizar o modelo por meio de uma aplicação web utilizando Streamlit.

---

## Dataset Utilizado

Dataset: Adult Income Dataset

O conjunto de dados contém informações de indivíduos, incluindo:

* Idade;
* Escolaridade;
* Estado civil;
* Ocupação;
* Horas trabalhadas por semana;
* Ganhos e perdas de capital;
* País de origem;
* Entre outras variáveis.

O dataset foi utilizado para prever a variável alvo:

* income (>50K ou <=50K)
---

## Tipo de Problema de Machine Learning

Classificação Supervisionada Binária.

O objetivo é classificar cada indivíduo em uma das duas classes possíveis:

* <=50K
* > 50K

---

## Metodologia

O desenvolvimento do projeto seguiu as seguintes etapas:

1. Análise exploratória dos dados (EDA);
2. Tratamento de dados ausentes;
3. Engenharia de atributos;
4. Codificação de variáveis categóricas;
5. Normalização das variáveis numéricas;
6. Divisão dos dados em treino, validação e teste;
7. Treinamento dos modelos;
8. Validação cruzada;
9. Avaliação dos resultados;
10. Deploy do modelo utilizando Streamlit.

---

## Modelos Treinados

Foram avaliados os seguintes algoritmos:

* Regressão Logística
* Random Forest
* Gradient Boosting

---

## Modelo Final Escolhido

Após a comparação dos resultados, o modelo selecionado foi:

**Gradient Boosting**

O modelo apresentou os melhores resultados gerais de classificação e capacidade de generalização.

---

## Métricas de Avaliação

As seguintes métricas foram utilizadas:

* Accuracy
* Precision
* Recall
* F1-Score
* AUC-ROC

Também foram analisadas:

* Curva ROC
* Matriz de Confusão
* Validação Cruzada

---

## Principais Resultados

Os resultados mostraram que:

* Todos os modelos apresentaram bom desempenho;
* Gradient Boosting apresentou o melhor desempenho geral;
* O modelo final demonstrou boa capacidade de generalização em dados não vistos.

As variáveis mais relevantes para a predição incluíram fatores relacionados à escolaridade, estado civil e ganhos de capital.

---

## Estrutura dos Arquivos

```text
.
├── app.py
├── requirements.txt
├── README.md
├── modelo_final.joblib
├── scaler.joblib
├── colunas_modelo.joblib
├── notebook.ipynb
└── relatorio.pdf
```

---

## Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit
* Matplotlib
* Jupyter Notebook

---

## Como Executar o Notebook

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Abra o notebook:

```bash
jupyter notebook
```

3. Execute as células na ordem apresentada.

---

## Como Executar o App Streamlit

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Execute:

```bash
streamlit run app.py
```

ou

```bash
py -3.11 -m streamlit run app.py
```

3. Acesse a URL exibida no terminal.

---

## Aplicação Publicada

* Streamlit: COLOCAR_LINK_STREAMLIT
* GitHub: COLOCAR_LINK_GITHUB

---

## Limitações

* O modelo foi treinado utilizando apenas os dados disponíveis no Adult Income Dataset;
* Os resultados representam estimativas estatísticas e não devem ser utilizados como critério absoluto para tomada de decisão.

---

## Conclusão

O projeto demonstrou a aplicação completa de um fluxo de Machine Learning, desde a preparação dos dados até o deploy de uma aplicação web.

Entre os modelos avaliados, o Gradient Boosting apresentou o melhor desempenho, sendo escolhido como modelo final. A aplicação desenvolvida permite realizar previsões de forma simples e intuitiva, demonstrando na prática a utilização de técnicas de Ciência de Dados e Machine Learning para problemas de classificação.
