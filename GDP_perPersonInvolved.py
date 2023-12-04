import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Replace the provided data with your actual dataset
data = {
    'Country Name': ['India'],
    'Country Code': ['IND'],
    'Indicator Name': ['GDP per person employed (constant 2017 PPP $)'],
    'Indicator Code': ['SL.GDP.PCAP.EM.KD'],
    '1960': [5723.81631],
    '1961': [5901.143132],
    '1962': [6034.860434],
    '1963': [6285.203399],
    '1964': [6549.205614],
    '1965': [6821.85088],
    '1966': [6870.403073],
    '1967': [7071.274883],
    '1968': [7463.12084],
    '1969': [7503.693225],
    '1970': [7765.459746],
    '1971': [7957.585586],
    '1972': [8488.304984],
    '1973': [9055.397999],
    '1974': [9667.414049],
    '1975': [10213.11611],
    '1976': [10752.51135],
    '1977': [10833.88747],
    '1978': [11452.51978],
    '1979': [12165.80924],
    '1980': [12680.32826],
    '1981': [13260.2666],
    '1982': [14082.02481],
    '1983': [15109.63784],
    '1984': [16309.9205],
    '1985': [17651.33215],
    '1986': [18849.91832],
    '1987': [20097.56002],
    '1988': [19538.39729],
    '1989': [19312.59857],
    '1990': [20059.1361],
    '1991': [20716.95187],
    # ... Add the remaining years ...
    '2022': [20716.95187]
}

df = pd.DataFrame(data)

# Transpose the dataframe so that years become rows and the GDP values become columns
df_transposed = df.set_index(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']).T

# Convert the index to datetime format
df_transposed.index = pd.to_datetime(df_transposed.index)

# Fit the ARIMA model
model = ARIMA(df_transposed, order=(1, 1, 1))
model_fit = model.fit()

# Forecast the GDP per person employed for the years 2024 to 2025
forecast = model_fit.forecast(steps=2)

# Print the forecasted values
print(forecast)
