import pandas as pd

# Load the credit scoring dataset
data = pd.read_csv("credit_data.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())
# Separate features and target
X = data.drop("Creditworthy", axis=1)
y = data["Creditworthy"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")
# Make predictions on the test data
y_pred = model.predict(X_test)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)

# Calculate model performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nModel Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Creditworthy", "Creditworthy"]
)

disp.plot()
plt.title("Credit Scoring - Confusion Matrix")
plt.show()
# Calculate ROC-AUC
from sklearn.metrics import roc_auc_score, roc_curve

# Get probability predictions
y_probability = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_probability)

print("\nROC-AUC Score:", roc_auc)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probability)

plt.figure()
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Credit Scoring Model")
plt.legend()
plt.show()


# ----------------------------------------
# Predict Creditworthiness for a New Customer
# ----------------------------------------

print("\n--- New Customer Credit Prediction ---")

age = float(input("Enter Age: "))
income = float(input("Enter Income: "))
loan_amount = float(input("Enter Loan Amount: "))
credit_score = float(input("Enter Credit Score: "))
debt = float(input("Enter Debt: "))
payment_history = float(input("Enter Payment History (0-10): "))
employment_years = float(input("Enter Employment Years: "))

# Create input data
new_customer = pd.DataFrame([[
    age,
    income,
    loan_amount,
    credit_score,
    debt,
    payment_history,
    employment_years
]], columns=X.columns)

# Make prediction
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)[0][1]

# Display result
if prediction[0] == 1:
    print("\nResult: CREDITWORTHY")
    print(f"Creditworthy Probability: {probability * 100:.2f}%")
    print("The customer is likely to be creditworthy.")
else:
    print("\nResult: NOT CREDITWORTHY")
    print(f"Creditworthy Probability: {probability * 100:.2f}%")
    print("The customer is likely not to be creditworthy.")
    print("\n===================================")
print("       CREDIT SCORING REPORT")
print("===================================")

print("Age:", age)
print("Income:", income)
print("Loan Amount:", loan_amount)
print("Credit Score:", credit_score)
print("Debt:", debt)
print("Payment History:", payment_history)
print("Employment Years:", employment_years)

print("-----------------------------------")

if prediction[0] == 1:
    print("Final Decision: CREDITWORTHY")
else:
    print("Final Decision: NOT CREDITWORTHY")

print("Creditworthiness Probability:",
      f"{probability * 100:.2f}%")

print("===================================")