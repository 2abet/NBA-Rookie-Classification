
# NBA Rookie Success Predictor Streamlit App

This repository contains a Streamlit app designed to predict the long-term success of NBA rookies based on their first-year statistics. The app allows users to upload a dataset, select a machine learning model for prediction, and interactively explore player data based on specific criteria.

## Features

- **Predictive Modeling:** Choose between Logistic Regression, Gaussian Naive Bayes, and Neural Network (MLP) models to predict whether NBA rookies will last at least 5 years in the league based on their performance metrics.
- **Player Search:** Enter a player's name to get a prediction of their long-term success in the NBA.
- **Criteria-Based Filtering:** Find players based on custom criteria such as minimum games played, points per game, and more.

## Installation

To use this online: Visit nbarookieapp.streamlit.app

To run this app locally, you'll need Python 3.6 or later. Follow these steps to get started:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/2abet/nba-rookie-predictor.git
   cd nba-rookie-predictor
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

Replace `app.py` with the path to your Streamlit script if it's named differently.

## Usage

After launching the app, follow the on-screen instructions:

1. **Upload Dataset:** Upload your NBA rookie dataset in CSV format. The dataset should include player statistics and a target variable indicating success.
2. **Model Selection:** Use the sidebar to select the prediction model you want to use.
3. **Explore:** Use the app's features to make predictions for specific players or filter the player dataset based on criteria you set.

## Contributing

Contributions to this project are welcome! Here are a few ways you can help:

- Report bugs and request features by opening issues.
- Contribute code: Fork the repository, make your changes, and submit a pull request.

Please refer to CONTRIBUTING.md for more details on making contributions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data sourced from Kaggle.

## Contact

For any queries, please open an issue.
