import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
import os

df = pd.read_csv('StudentPerformance1.csv')

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Parental_Involvement'] = df['Parental_Involvement'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'Yes': 1, 'No': 0})
df['Motivation_Level'] = df['Motivation_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['Internet_Access'] = df['Internet_Access'].map({'Yes': 1, 'No': 0})
df['Family_Income'] = df['Family_Income'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['Teacher_Quality'] = df['Teacher_Quality'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['School_Type'] = df['School_Type'].map({'Public': 0, 'Private': 1})

X = df[['Hours_Studied', 'Attendance', 'Parental_Involvement', 'Extracurricular_Activities', 
        'Sleep_Hours', 'Previous_Scores', 'Motivation_Level', 'Internet_Access', 
        'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality', 'School_Type', 
        'Physical_Activity', 'Gender']]
y = df['Exam_Score']

imputer = SimpleImputer(strategy='mean') 
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if os.path.exists('models/lasso_model.pkl'):
    lasso = joblib.load('models/lasso_model.pkl')
    linear = joblib.load('models/linear_model.pkl')
    mlp = joblib.load('models/mlp_model.pkl')
    stacking_regressor = joblib.load('models/stacking_model.pkl')
    decision_tree = joblib.load('models/decision_tree_model.pkl')
else:
    linear = LinearRegression()
    linear.fit(X_train, y_train)

    param_grid_lasso = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    }

    lasso = GridSearchCV(Lasso(), param_grid=param_grid_lasso, cv=3, n_jobs=-1)
    lasso.fit(X_train, y_train)

    param_grid_mlp = {
        'hidden_layer_sizes': [(100,), (150,), (100, 100), (150, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [1000],
        'early_stopping': [True],
        'validation_fraction': [0.1],
        'n_iter_no_change': [10]
    }

    mlp = GridSearchCV(MLPRegressor(random_state=42), param_grid=param_grid_mlp, cv=3, n_jobs=-1)
    mlp.fit(X_train, y_train)

    base_models = [
        ('linear', linear),
        ('lasso', lasso.best_estimator_),
        ('mlp', mlp.best_estimator_),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ]

    stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), n_jobs=-1)
    stacking_regressor.fit(X_train, y_train)
    
    decision_tree = DecisionTreeRegressor(random_state=42)
    decision_tree.fit(X_train, y_train)

    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(linear, 'models/linear_model.pkl')
    joblib.dump(lasso, 'models/lasso_model.pkl')
    joblib.dump(mlp.best_estimator_, 'models/mlp_model.pkl')
    joblib.dump(stacking_regressor, 'models/stacking_model.pkl')
    joblib.dump(decision_tree, 'models/decision_tree_model.pkl')

y_pred_lasso = lasso.predict(X_test)
y_pred_linear = linear.predict(X_test)
y_pred_mlp = mlp.predict(X_test)
y_pred_stacking = stacking_regressor.predict(X_test)
y_pred_decision_tree = decision_tree.predict(X_test)

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

lassoMAE, lassoMSE, lassoR2 = evaluate_model(y_test, y_pred_lasso)
linearMAE, linearMSE, linearR2 = evaluate_model(y_test, y_pred_linear)
mlpMAE, mlpMSE, mlpR2 = evaluate_model(y_test, y_pred_mlp)
stackingMAE, stackingMSE, stackingR2 = evaluate_model(y_test, y_pred_stacking)
decision_treeMAE, decision_treeMSE, decision_treeR2 = evaluate_model(y_test, y_pred_decision_tree)

st.title("Dự đoán điểm thi với nhiều mô hình")

st.write("Nhập dữ liệu để dự đoán:")

hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, step=1)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, step=1)
parental_involvement = st.selectbox("Parental Involvement", ['Low', 'Medium', 'High'])
extracurricular_activities = st.selectbox("Extracurricular Activities", ['Yes', 'No'])
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=12, step=1)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, step=1)
motivation_level = st.selectbox("Motivation Level", ['Low', 'Medium', 'High'])
internet_access = st.selectbox("Internet Access", ['Yes', 'No'])
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, step=1)
family_income = st.selectbox("Family Income", ['Low', 'Medium', 'High'])
teacher_quality = st.selectbox("Teacher Quality", ['Low', 'Medium', 'High'])
school_type = st.selectbox("School Type", ['Public', 'Private'])
physical_activity = st.number_input("Physical Activity (hours per week)", min_value=0, max_value=10, step=1)
gender = st.selectbox("Gender", ['Male', 'Female'])

if st.button("Dự đoán"):
    new_data = [[
        hours_studied, 
        attendance, 
        0 if parental_involvement == 'Low' else 1 if parental_involvement == 'Medium' else 2, 
        1 if extracurricular_activities == 'Yes' else 0, 
        sleep_hours, 
        previous_scores, 
        0 if motivation_level == 'Low' else 1 if motivation_level == 'Medium' else 2,
        1 if internet_access == 'Yes' else 0,
        tutoring_sessions,
        0 if family_income == 'Low' else 1 if family_income == 'Medium' else 2,
        0 if teacher_quality == 'Low' else 1 if teacher_quality == 'Medium' else 2,
        0 if school_type == 'Public' else 1,
        physical_activity,
        1 if gender == 'Male' else 0
    ]]
    
    prediction_linear = linear.predict(new_data)[0]
    prediction_lasso = lasso.predict(new_data)[0]
    prediction_mlp = mlp.predict(new_data)[0]
    prediction_stacking = stacking_regressor.predict(new_data)[0]
    prediction_decision_tree = decision_tree.predict(new_data)[0]
    st.write("### Kết quả dự đoán")
    st.table(pd.DataFrame({
    'Model': ['Linear Regression', 'Lasso', 'MLP', 'Stacking', 'Decision Tree'], 
    'Dự đoán': [prediction_linear, prediction_lasso, prediction_mlp, prediction_stacking, prediction_decision_tree]
    }))

    st.write("### Đánh giá mô hình")
    st.table(pd.DataFrame({
    'Model': ['Linear Regression', 'Lasso', 'MLP', 'Stacking', 'Decision Tree'], 
    'MAE': [linearMAE, lassoMAE, mlpMAE, stackingMAE, decision_treeMAE],
    'MSE': [linearMSE, lassoMSE, mlpMSE, stackingMSE, decision_treeMSE],
    'R²': [linearR2, lassoR2, mlpR2, stackingR2, decision_treeR2],
    }))



    st.write("### Đồ thị sai số")
    errors_lasso = y_test - y_pred_lasso
    errors_linear = y_test - y_pred_linear
    errors_mlp = y_test - y_pred_mlp
    errors_stacking = y_test - y_pred_stacking
    errors_decision_tree = y_test - y_pred_decision_tree

    fig, axs = plt.subplots(1, 5, figsize=(30, 8))

    axs[0].hist(errors_lasso, bins=50, edgecolor='k')
    axs[0].set_title('Lasso - Phân phối sai số')
    axs[0].set_xlabel('Error')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(errors_linear, bins=50, edgecolor='k')
    axs[1].set_title('Linear Regression - Phân phối sai số')
    axs[1].set_xlabel('Error')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(errors_mlp, bins=50, edgecolor='k')
    axs[2].set_title('MLP - Phân phối sai số')
    axs[2].set_xlabel('Error')
    axs[2].set_ylabel('Frequency')

    axs[3].hist(errors_stacking, bins=50, edgecolor='k')
    axs[3].set_title('Stacking - Phân phối sai số')
    axs[3].set_xlabel('Error')
    axs[3].set_ylabel('Frequency')

    axs[4].hist(errors_decision_tree, bins=50, edgecolor='k')
    axs[4].set_title('Decision Tree - Phân phối sai số')
    axs[4].set_xlabel('Error')
    axs[4].set_ylabel('Frequency')

    st.pyplot(fig)
    st.write("### So sánh giá trị thực và dự đoán")

    fig, axs = plt.subplots(3, 2, figsize=(16, 18))

    axs[0, 0].scatter(y_test, y_pred_lasso, color='blue', alpha=0.5)
    axs[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    axs[0, 0].set_xlabel('Giá trị thực')
    axs[0, 0].set_ylabel('Giá trị dự đoán')
    axs[0, 0].set_title('Lasso')

    axs[0, 1].scatter(y_test, y_pred_linear, color='green', alpha=0.5)
    axs[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    axs[0, 1].set_xlabel('Giá trị thực')
    axs[0, 1].set_ylabel('Giá trị dự đoán')
    axs[0, 1].set_title('Linear Regression')

    axs[1, 0].scatter(y_test, y_pred_mlp, color='red', alpha=0.5)
    axs[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    axs[1, 0].set_xlabel('Giá trị thực')
    axs[1, 0].set_ylabel('Giá trị dự đoán')
    axs[1, 0].set_title('MLP')

    axs[1, 1].scatter(y_test, y_pred_stacking, color='orange', alpha=0.5)
    axs[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    axs[1, 1].set_xlabel('Giá trị thực')
    axs[1, 1].set_ylabel('Giá trị dự đoán')
    axs[1, 1].set_title('Stacking')

    axs[2, 0].scatter(y_test, y_pred_decision_tree, color='purple', alpha=0.5)
    axs[2, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    axs[2, 0].set_xlabel('Giá trị thực')
    axs[2, 0].set_ylabel('Giá trị dự đoán')
    axs[2, 0].set_title('Decision Tree')

    axs[2, 1].axis('off')

    st.pyplot(fig)


