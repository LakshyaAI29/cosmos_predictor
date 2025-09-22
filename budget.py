import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = LinearRegression()
budget_data = pd.read_csv(r"C:\Users\rajpu\Downloads\archive\Global_Space_Exploration_Dataset.csv")


x = budget_data[["Mission Type", "Satellite Type", "Technology Used", "Duration (in Days)"]]

y = budget_data[["Budget (in Billion $)", "Success Rate (%)"]]

x = pd.get_dummies(x, columns=["Mission Type", "Satellite Type", "Technology Used"], drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

mission_type = input("Enter the mission type: ")
satellite_type = input("Enter the satellite type: ")
technology_used = input("Enter the technology used: ")
duration = int(input("Enter the duration (in days): "))

# Create new input DataFrame
new_data = pd.DataFrame([[mission_type, satellite_type, technology_used, duration]],
                        columns=["Mission Type", "Satellite Type", "Technology Used", "Duration (in Days)"])

# One-hot encode the new data like training
new_data_encoded = pd.get_dummies(new_data, columns=["Mission Type", "Satellite Type", "Technology Used"], drop_first=True)

# Align columns with training data
new_data_encoded = new_data_encoded.reindex(columns=x_train.columns, fill_value=0)

# Predict using the model
prediction = model.predict(new_data_encoded)[0]
predicted_budget = prediction[0]
predicted_success_rate = prediction[1]
print(f"Predicted Budget (in Billion $): {predicted_budget}")
print(f"Predicted Success Rate (%): {predicted_success_rate}")