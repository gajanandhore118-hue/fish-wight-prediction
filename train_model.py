import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Fish[1].csv")

# Convert categorical → numerical
df = pd.get_dummies(df, columns=["Species"])

# Features & target
X = df.drop("Weight", axis=1)
y = df["Weight"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save columns
with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns, f)

print("Model trained and saved ✅")