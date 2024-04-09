import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('datos_temperatura.csv')

X = data['Celsius'].values.reshape(-1, 1)
y = data['Fahrenheit'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print("Precisión del modelo:", accuracy)

def celsius_to_fahrenheit(celsius_temp):
    fahrenheit_temp = model.predict([[celsius_temp]])
    return fahrenheit_temp[0]

celsius_input = 30
fahrenheit_output = celsius_to_fahrenheit(celsius_input)
print(f"{celsius_input} grados Celsius son {fahrenheit_output} grados Fahrenheit.")
