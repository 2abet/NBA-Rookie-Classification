import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.base import TransformerMixin

# Define a transformer for converting sparse matrices to dense
class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        return X.toarray()

st.title('NBA Rookie Prediction App')

model_choice = st.sidebar.selectbox("Select the model for prediction:", ["Logistic Regression", "Gaussian Naive Bayes", "Neural Network (MLP)"])

data_file = st.file_uploader("Upload your NBA Rookie dataset here:", type=["csv"])

if data_file is not None:
    df = pd.read_csv(data_file)
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    if 'TARGET_5Yrs' not in df.columns:
        st.error("The uploaded dataset does not contain a 'TARGET_5Yrs' column.")
        st.stop()

    # Assuming 'Name' or a similar column for player identification; adjust as necessary
    if 'Name' not in df.columns:
        st.error("The uploaded dataset does not contain a 'Name' column for player identification.")
        st.stop()

    target_column = 'TARGET_5Yrs'
    features = df.drop(columns=[target_column, 'Name'])  
    feature_columns = st.multiselect("Select features to include for model training:", features.columns, default=features.columns.tolist())

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    if model_choice == "Gaussian Naive Bayes":
        classifier = Pipeline([('to_dense', DenseTransformer()), 
                               ('classifier', GaussianNB())])
    elif model_choice == "Logistic Regression":
        classifier = LogisticRegression()
    else:  # Neural Network (MLP)
        classifier = MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu', max_iter=1000, random_state=42)

    clf = Pipeline(steps=[('preprocessor', preprocessor), 
                          ('classifier', classifier)])

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    st.write(f"### {model_choice} Accuracy: ", accuracy)

    cm = confusion_matrix(y_test, predictions)
    st.write("### Confusion Matrix:")
    st.dataframe(pd.DataFrame(cm, index=['actual negative', 'actual positive'], columns=['predicted negative', 'predicted positive']))

    report = classification_report(y_test, predictions, output_dict=True)
    st.write("### Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    # Prediction by Player's Name
    st.write("## Prediction for a Specific Player")
    player_name_input = st.text_input("Enter the player's name:")
    
    if player_name_input:
        player_data = df[df['Name'].str.contains(player_name_input, case=False, na=False)]
        
        if not player_data.empty:
            player_features = player_data[feature_columns]
            player_prediction = clf.predict(player_features)
            
            st.write(f"### Players Found:")
            st.dataframe(player_data[['Name'] + feature_columns])
            
            st.write("### Prediction Results:")
            player_data['Prediction'] = player_prediction
            player_data['Prediction'] = player_data['Prediction'].apply(lambda x: 'Likely to last 5 years' if x == 1 else 'Unlikely to last 5 years')
            st.dataframe(player_data[['Name', 'Prediction']])
        else:
            st.write("No players found with that name.")

    # Find Players Based on Criteria
    st.write("## Find Players Based on Criteria")
    criteria_columns = st.multiselect("Select criteria columns:", feature_columns, default=None)
    
    if criteria_columns:
        criteria_values = {}
        for column in criteria_columns:
            min_val, max_val = float(df[column].min()), float(df[column].max())
            criteria_values[column] = st.slider(f"Minimum {column}:", min_val, max_val, min_val, step=(max_val - min_val) / 100)
        
        filtered_players = df.copy()
        for column, min_val in criteria_values.items():
            filtered_players = filtered_players[filtered_players[column] >= min_val]
        
        if not filtered_players.empty:
            st.write("### Players Matching Criteria:")
            st.dataframe(filtered_players[['Name'] + criteria_columns])
        else:
            st.write("No players found matching the criteria.")
