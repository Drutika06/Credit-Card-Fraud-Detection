from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Initialize LabelEncoder for categorical columns
label_encoder = LabelEncoder()

# Example fit for 'type' column used in training (This should match the training step)
label_encoder.fit(['PAYMENT', 'CASHOUT', 'TRANSFER', 'DEBIT', 'CREDIT'])

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None  # Initialize prediction result variable
    if request.method == 'POST':
        input_data = request.form.get('input_data')
        if input_data:
            try:
                # Convert the input string to a list of values
                input_values = input_data.split(',')

                # Ensure the correct number of features (11 columns for the input data)
                if len(input_values) == 11:
                    # Preprocess the 'type' column (categorical encoding)
                    input_values[1] = label_encoder.transform([input_values[1]])[0]  # Encode 'type' as numeric

                    # Convert the rest of the columns to float (except 'type' which is already numeric)
                    input_values[0] = float(input_values[0])  # step
                    input_values[2] = float(input_values[2])  # amount
                    input_values[4] = float(input_values[4])  # oldbalanceOrg
                    input_values[5] = float(input_values[5])  # newbalanceOrig
                    input_values[7] = float(input_values[7])  # oldbalanceDest
                    input_values[8] = float(input_values[8])  # newbalanceDest

                    # Create a DataFrame with the input values, ensuring correct column names and order
                    input_df = pd.DataFrame([input_values], columns=[
                        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
                        "newbalanceOrig", "nameDest", "oldbalanceDest",
                        "newbalanceDest", "isFraud", "isFlaggedFraud"
                    ])

                    # Drop 'nameOrig' and 'nameDest' as they are not used for prediction
                    input_df = input_df.drop(['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud'], axis=1)

                    # Make a prediction
                    prediction = model.predict(input_df)
                    if prediction[0] == 0:
                        prediction_result = "The transaction is Legitimate."
                    else:
                        prediction_result = "The transaction is Fraudulent."
                else:
                    prediction_result = "Invalid input: Please provide 11 values."

            except Exception as e:
                prediction_result = f"Error processing input: {str(e)}"

    return render_template('index.html', prediction_result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)





