import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load CSV files
df_train = pd.read_csv("/content/Training.csv")
df_test = pd.read_csv("/content/Test.csv")
df_validation = pd.read_csv("/content/Validation.csv")

# Prepare training, validation, and test data
x_train = df_train['Text']
y_train = df_train['Emotion']
x_validation = df_validation['Text']
y_validation = df_validation['Emotion']
x_test = df_test['Text']
y_test = df_test['Emotion']

# Train the model
print('SVM starts:')
pipe_svm = Pipeline(steps=[
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(kernel='linear', probability=True))
])

print('Looking for the best C parameter:')
# Hyperparameter tuning using GridSearchCV
param_grid = {'svm__C': [1, 10, 100]}  # Try different values for 'C'
grid_search = GridSearchCV(pipe_svm, param_grid, cv=5, scoring='accuracy')

# Fit model with training data
grid_search.fit(x_train, y_train)

# Best parameters and score from GridSearchCV
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation score: {grid_search.best_score_}')

# Evaluate on training, validation, and test sets
print(f'Training score: {grid_search.score(x_train, y_train)}')
print(f'Validation score: {grid_search.score(x_validation, y_validation)}')
print(f'Test score: {grid_search.score(x_test, y_test)}')

# Save the best model after GridSearchCV
with open('Cortext.pkl', 'wb') as pipeline_file:
    pickle.dump(grid_search.best_estimator_, pipeline_file)

print("Model saved as 'Cortext.pkl'.")