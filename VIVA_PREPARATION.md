# Viva Preparation: AI-Based Student Performance Prediction System

## 1. What is your project?
It is a Python console application that manages student records in SQLite and uses machine learning to predict a performance category.

## 2. Why is it an AI project?
It trains and uses a Decision Tree Classifier from scikit-learn. The model learns patterns from labeled examples and predicts a category for new inputs.

## 3. Which algorithm did you use?
Decision Tree Classification. It learns a sequence of feature-based split rules to assign a class.

## 4. What are the input features?
Previous marks, attendance percentage, and average daily study hours.

## 5. What is the target variable?
Performance category: Needs Support, Average, or Strong.

## 6. What is the dataset?
A clearly labeled synthetic dataset created for demonstration. It is not real student data.

## 7. Why split data into training and testing sets?
Training data is used to fit the model; held-out test data estimates performance on examples not used for training.

## 8. What is accuracy?
The fraction of test predictions that match the test labels. Accuracy alone can hide class-specific errors, so the project also shows precision, recall, F1-score, and a confusion matrix.

## 9. What is overfitting?
When a model learns the training data too closely and performs poorly on new data. Limiting tree depth and minimum leaf size helps reduce this risk.

## 10. Why use SQLite?
It is a lightweight local database that does not require a separate database server.

## 11. What is CRUD?
Create, Read, Update, and Delete. These operations manage student records.

## 12. Is the prediction always correct?
No. Predictions are uncertain, and the synthetic test score does not prove real-world performance.

## 13. What is scikit-learn?
A Python library that provides machine-learning algorithms, model-selection tools, and evaluation metrics.

## 14. Why is the dataset synthetic?
It allows a reproducible demonstration without exposing real student data. Its limitations must be explained honestly.

## 15. What can be improved in the future?
Use an authorized real dataset, test on representative unseen data, add a graphical interface, and collect feedback while protecting student privacy.
