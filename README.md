import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests

# --- Projeto 1: Previsão de Preços de Automóveis ---
# Carregar dados fictícios de automóveis
data = pd.DataFrame({
    'marca': ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes'],
    'ano': [2015, 2018, 2012, 2020, 2019],
    'km_rodados': [50000, 30000, 70000, 20000, 25000],
    'preco': [40000, 45000, 30000, 80000, 85000]
})

# Separar variáveis preditoras e alvo
X = data.drop(columns=['preco'])
y = data['preco']

# Definir colunas categóricas e numéricas
cat_features = ['marca']
num_features = ['ano', 'km_rodados']

# Criar transformações para os dados
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(), cat_features)
])

# Criar pipeline de modelagem
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Separar dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
pipeline.fit(X_train, y_train)

# Fazer previsões
predictions = pipeline.predict(X_test)

# Avaliar modelo
mse = mean_squared_error(y_test, predictions)
print(f'Erro Quadrático Médio (MSE): {mse:.2f}')

# --- Projeto 2: Previsão de Criptomoedas ---
# Obter dados de preços de criptomoedas
def fetch_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=10"
    response = requests.get(url)
    data = response.json()
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
    prices['price_change'] = prices['price'].diff().fillna(0)
    prices['target'] = (prices['price_change'] > 0).astype(int)
    return prices

crypto_data = fetch_crypto_data()
crypto_data.to_csv('crypto_prices.csv', index=False)

# Separar dados para treinamento
features = ['price']
X_crypto = crypto_data[features]
y_crypto = crypto_data['target']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_crypto, y_crypto, test_size=0.2, random_state=42)

# Criar modelo simples
crypto_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LinearRegression())
])

# Treinar modelo
crypto_model.fit(X_train_c, y_train_c)

# Avaliar modelo
accuracy = crypto_model.score(X_test_c, y_test_c)
print(f'Acurácia do modelo: {accuracy:.2f}')
