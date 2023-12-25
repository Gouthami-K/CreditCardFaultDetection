# Credit Card Fault Detection

## Overview

This project focuses on building a credit card fault detection model using machine learning techniques. The model aims to predict default payments based on various demographic and credit-related features.

## Dataset

The dataset used in this project contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan. It includes 25 variables such as ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0 to PAY_6, BILL_AMT1 to BILL_AMT6, PAY_AMT1 to PAY_AMT6, and the target variable default.payment.next.month.

## Setup

### Dependencies

- Python 3.8
- Dependencies listed in requirements.txt

### Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Exploration:** Explore the dataset to understand its structure and characteristics.
2. **Data Preprocessing:** Balancing imbalance data, and scale numerical features.
3. **Model Selection:** Train and evaluate various machine learning models (SVM, KNN, Decision Tree, Gradient Boosting, Logistic Regression, AdaBoosting, Naive Bayes).
4. **Model Training:** Choose the best-performing model and train it on the dataset.
5. **Results:** Evaluate and analyze the model's performance, strengths, weaknesses, and any challenges encountered.
6. **Deployment on AWS:** Deploy the model on AWS for real-world applications.
7. **Video Demonstration:** Check out this [link](https://drive.google.com/file/d/15V08jQHR1d2vFW48bOMHUbh1fH86wWQf/view?usp=drive_link) for a demonstration on AWS.
   
## Results

The Gradient Boosting model emerged as the most effective, achieving an good accuracy on the test set.

## Future Work

Explore additional features, fine-tune hyperparameters, and consider more advanced techniques for handling imbalanced data.

## Contributors

- Gouthami K
- Abhijith Paul

## License

This project is licensed under the MIT License.
