import streamlit as st
import joblib
import os
import numpy as np
import torch
from model import load_model, ModelPrediction

model_path = os.path.join(".", "wights_raw.pickle")
scaler_path = os.path.join(".", "scaler.pkl")
print(f"Loading model from: {model_path}")
print(f"Loading scaler from: {scaler_path}")
model, meta_data = load_model(model_path)
scaler = joblib.load(scaler_path)
class_mapping = {0: "Normal", 1: "Failure"}


def make_prediction(data):
    try:
        data_t = np.array(data, dtype=np.float32)
        if data_t.shape[0] != 16:
            return "Please enter 16 comma-separated values."

        data_t = scaler.transform(data_t.reshape(-1, 16))
        data_t = torch.tensor(data_t, dtype=torch.float32)
        preds = model(data_t).detach().cpu().numpy()[0]
        epred = np.exp(preds)
        probs = epred / epred.sum()
        pclass = np.argmax(probs)
        conf = np.max(probs)
        lbl = class_mapping[int(pclass)]
        prediction = ModelPrediction(
            predicted_class=lbl,
            predicted_label=int(pclass),
            raw_out=preds.tolist(),
            probabilities=probs.tolist(),
            confidence=conf
        )
        return prediction
    except ValueError:
        return "Invalid input. Please enter 16 comma-separated values."


def main():
    st.sidebar.title("Contents")
    selected_option = st.sidebar.selectbox("Go to:", ["Prediction", "Data Visualization", "About", "Group"])
    if selected_option == "Prediction":
        display_main_page()
    elif selected_option == "Data Visualization":
        display_data_visualization()
    elif selected_option == "About":
        display_about_section()
    elif selected_option == "Group":
        display_team_section()


def display_main_page():
    st.empty()  # Clear the page content
    st.title("Machinery Anomaly Detection for Preventive Maintenance")
    input_data = st.text_input("Enter sensor data (comma-separated values):")
    if st.button("Predict"):
        if input_data:
            input_data = input_data.split(',')
            try:
                input_data = [float(x) for x in input_data]
                prediction = make_prediction(input_data)
                if isinstance(prediction, str):
                    st.error(prediction)
                else:
                    st.success(f"Predicted Class: {prediction.predicted_class}")
                    st.write(f"Confidence: {prediction.confidence}")
            except ValueError:
                st.error("Please enter 16 comma-separated numerical values.")


def display_data_visualization():
    st.empty()  # Clear the page content
    st.title("Data Visualization")
    st.subheader("This section contains various graphs and visualizations.")
    cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs',
            'Oil_temperature', 'Motor_current', 'COMP', 'DV_electric', 'Towers',
            'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses']
    selected_feature = st.selectbox("Select a feature:", cols)
    feature_images = {
        "Oil_level": "Picture 1.jpg",
        "Caudal_impulses": "Picture2.jpg",
        "LPS": "Picture3.jpg",
        "Oil_temperature": "Picture4.jpg",
        "Reservoirs": "1.png",
        "Motor_current": "2.png",
        "COMP": "3.png",
        "DV_electric": "4.png",
        "Towers": "5.png",
        "MPG": "6.png",
        "H1": "7.png",
        "DV_pressure": "8.png",
        "Pressure_switch": "9.png",
        "TP2": "10.png",
        "TP3": "11.png",
        # Add more features and image paths as needed
    }
    if selected_feature in feature_images:
        image_path = feature_images[selected_feature]
        st.image(image_path, caption=selected_feature, use_column_width=True)
    cols2 = ['Box Plot with Outliers', 'Box Plot without Outliers', 'Bar graph representing non- anomaly (0) and '
                                                                    'anomaly class (1)',
             'Accuracy score vs K-value', 'Error rate vs K-value']
    selected_feature2 = st.selectbox("Select a feature:", cols2)
    feature_images2 = {
        'Box Plot with Outliers': "Picture17.png",
        'Box Plot without Outliers': "Picture16.jpg",
        'Bar graph representing non- anomaly (0) and anomaly class (1)': "Picture18.jpg",
        'Accuracy score vs K-value': "Picture20.jpg",
        'Error rate vs K-value': "Picture19.jpg"
    }
    if selected_feature2 in feature_images2:
        image_path = feature_images2[selected_feature2]
        st.image(image_path, caption=selected_feature2, use_column_width=True)


def display_about_section():
    st.title("About Section")
    st.subheader("This section contains information about the project.")
    st.header("About the Project")
    st.write("The project aims to develop a machine learning model for machinery anomaly detection for preventive "
             "maintenance. The goal is to identify instances when the machinery is in a failure state, "
             "which would require maintenance or repair actions to prevent further issues or breakdowns.")
    st.write("The dataset is preprocessed by handling missing values, dropping unnecessary columns, and converting "
             "data types. A crucial step is creating a target variable called 'Status' that indicates whether the "
             "machinery is in a failure state (1) or not (0). This is done by analyzing predefined failure time "
             "intervals, where the data samples falling within those intervals are labeled as failures (Status = "
             "1), and the rest are labeled as normal operation (Status = 0)")
    st.write("The dataset is then split into "
             "positive (failure) and negative (non-failure) samples. To balance the classes, the negative samples "
             "are under sampled to match the number of positive samples.")
    st.write("The project explores various machine "
             "learning models, including Logistic Regression, K-Nearest Neighbors (KNN), Naive Bayes (Gaussian, "
             "Multinomial, and Bernoulli), Random Forest, Support Vector Machines (SVM), and Long Short-Term "
             "Memory (LSTM) neural networks. These models are trained and evaluated using metrics like accuracy, "
             "precision, recall, F1-score, and confusion matrices.")
    st.write("The code also performs hyperparameter tuning "
             "for some models, such as Random Forest and SVM, using techniques like RandomizedSearchCV and grid "
             "search to optimize their performance.")
    st.write("Overall, the project aims to develop an effective machine "
             "learning model for detecting machinery anomalies or failure states based on sensor data, "
             "which can be used for preventive maintenance purposes to avoid unexpected breakdowns and minimize "
             "downtime.")
    st.header("About the Dataset")
    st.write("This project uses the MetroPT dataset, which contains sensor data collected from metro trains in "
             "Porto, Portugal. The dataset includes various measurements related to the trains' operations and "
             "maintenance. The dataset consists of 15169480 data points collected at 1Hz from February to "
             "August 2020 and is described by 15 features from 7 analogue (1-7) and 8 digital (8-15) sensors: ")
    st.write("1. TP2 "
             "(bar): the measure of the pressure on the compressor.")
    st.write("2.TP3 (bar): the measure of the "
             "pressure generated at the pneumatic panel.")
    st.write("3. H1 (bar): the measure of the pressure "
             "generated due to pressure drop when the discharge of the cyclonic separator filter "
             "occurs.")
    st.write("4. DV pressure (bar): the measure of the pressure drop generated when the towers "
             "discharge air dryers; a zero reading indicates that the compressor is operating under "
             "load.")
    st.write("5. Reservoirs (bar): the measure of the downstream pressure of the reservoirs, "
             "which should be close to the pneumatic panel pressure (TP3).")
    st.write("6. Motor Current (A): the "
             "measure of the current of one phase of the three-phase motor; it presents values close to 0A - when "
             "it turns off, 4A - when working offloaded, 7A - when working under load, and 9A - when it starts "
             "working.")
    st.write("7. Oil Temperature (ºC): the measure of the oil temperature on the "
             "compressor.")
    st.write("8. COMP: the electrical signal of the air intake valve on the compressor; "
             "it is active when there is no air intake, indicating that the compressor is either turned off or "
             "operating in an offloaded state.")
    st.write("9. DV electric: the electrical signal that controls "
             "the compressor outlet valve; it is active when the compressor is functioning under load and "
             "inactive when the compressor is either off or operating in an offloaded "
             "state.")
    st.write("10. TOWERS: the electrical signal that defines the tower responsible for drying "
             "the air and the tower responsible for draining the humidity removed from the air; when not active, "
             "it indicates that tower one is functioning; when active, it indicates that tower two is in "
             "operation.")
    st.write("11. MPG: the electrical signal responsible for starting the compressor under "
             "load by activating the intake valve when the pressure in the air production unit (APU) falls below "
             "8.2 bar; it activates the COMP sensor, which assumes the same behaviour as the MPG "
             "sensor.")
    st.write("12. LPS: the electrical signal that detects and activates when the pressure "
             "drops below 7 bars.")
    st.write("13. Pressure Switch: the electrical signal that detects the "
             "discharge in the air-drying towers.")
    st.write("14. Oil Level: the electrical signal that detects "
             "the oil level on the compressor; it is active when the oil is below the expected "
             "values.")
    st.write("15. Caudal Impulse: the electrical signal that counts the pulse outputs "
             "generated by the absolute amount of air flowing from the APU to the reservoirs.")


def display_team_section():
    # Load images
    st.empty()
    image1 = "image1.jpg"
    image2 = "image2.jpg"
    image3 = "image3.jpeg"
    image4 = "image4.jpg"
    # Set width and height for passport size
    passport_width = 150  # Adjust as needed
    # Display images side by side with multi-line captions
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(image1, caption='Aranyak Karan - 12017544', width=passport_width)
    with col2:
        st.image(image2, caption='Sparsh Baliyan - 12013338', width=passport_width)
    with col3:
        st.image(image3, caption="Akash Alaria - 12019571", width=passport_width)
    with col4:
        st.image(image4, caption="Jaskirat Singh Bagga - 12012409", width=passport_width)


if __name__ == '__main__':
    main()
