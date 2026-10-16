from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load your credit scoring model
model = pickle.load(open("credit_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    income = float(request.form["income"])
    loan_amount = float(request.form["loan_amount"])
    credit_score = float(request.form["credit_score"])
    debt = float(request.form["debt"])
    payment_history = float(request.form["payment_history"])
    employment_years = float(request.form["employment_years"])

    new_customer = pd.DataFrame([[
        age,
        income,
        loan_amount,
        credit_score,
        debt,
        payment_history,
        employment_years
    ]], columns=[
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "Debt",
        "PaymentHistory",
        "EmploymentYears"
    ])

    prediction = model.predict(new_customer)
    probability = model.predict_proba(new_customer)[0][1]

    if prediction[0] == 1:
        result = "CREDITWORTHY"
    else:
        result = "NOT CREDITWORTHY"

    return render_template(
        "index.html",
        result=result,
        probability=f"{probability * 100:.2f}%"
    )


if __name__ == "__main__":
    app.run(debug=True)
