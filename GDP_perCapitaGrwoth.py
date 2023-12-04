import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Create a dataframe with the given data
data = {
    'Year': range(1960, 2023),
    'GDP_Growth': [1.359574512, 0.578972021, 3.574901046, 5.008485432, -4.788008238, -2.183001818,
                   5.55485816, 1.169729838, 4.218131001, 2.85855823, -0.585705894, -2.743920031,
                   0.996135555, -1.074028888, 6.733780786, -0.558062114, 4.912095731, 3.398876544,
                   -7.321991041, 4.349374093, 3.62086273, 1.162592174, 4.892732989, 1.497907949,
                   2.914206593, 2.460505288, 1.691962091, 7.259166023, 3.673862559, 3.297854066,
                   -1.045105469, 3.316865935, 2.627465863, 4.525133811, 5.452945864, 5.471108102,
                   2.071873277, 4.19892116, 6.851333408, 1.965952437, 2.945319359, 1.975907984,
                   6.016520564, 6.132606168, 6.206011032, 6.426044715, 6.093625779, 1.630780589,
                   6.371709389, 7.01317458, 3.81807313, 4.06082357, 5.014611864, 6.086180227,
                   6.721067631, 6.980989701, 5.568333514, 5.302408679, 2.811873102, -6.72629208,
                   8.184367716, 6.277403814, 0.0]  # Add a placeholder value for the 63rd element
}

df = pd.DataFrame(data)

# Convert the 'Year' column to datetime format
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# Set the 'Year' column as the dataframe's index
df.set_index('Year', inplace=True)

# Fit the ARIMA model
model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit()

# Forecast the GDP per capita growth for the years 2023 to 2025
forecast = model_fit.forecast(steps=3)

# Print the forecasted values
print(forecast)
