
# NBA Rookie Prediction Models

This project focuses on predicting the success of NBA rookies based on their first-year performance metrics. Using a dataset containing various statistics for NBA rookies, we build and evaluate several machine learning models to predict which players are likely to have a successful career over five years.

## Dataset Overview

The dataset includes various performance metrics for NBA rookies, such as points per game, assists, rebounds, and more. The target variable is `TARGET_5Yrs`, indicating whether the player had a successful career five years after their rookie season.

## Models Implemented

- **Logistic Regression:** A basic classification algorithm used for binary outcomes.
- **Gaussian Naive Bayes:** A simple probabilistic classifier based on applying Bayes' theorem.
- **Neural Network (Multi-layer Perceptron):** A more complex model capable of capturing nonlinear relationships.

## Model Evaluation

We evaluate the models using accuracy, confusion matrices, and classification reports to understand their performance comprehensively.

## Feature Selection and Scaling

Feature selection was performed to identify the most relevant features, and feature scaling was applied to normalize the data before feeding it into the models.

## Principal Component Analysis (PCA)**

PCA was applied to reduce the dimensionality of the data, improving the computational efficiency and potentially enhancing the model's performance by focusing on the most significant features.

## Getting Started

To run the models and evaluate their performance, ensure you have the necessary libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Then, you can clone this repository and run the Jupyter notebook to see the models in action.

## Conclusion

The models provide insights into the factors that contribute to an NBA rookie's long-term success and offer a framework for further exploration and analysis.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
