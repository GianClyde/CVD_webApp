from django.shortcuts import render, redirect, HttpResponse
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from django.shortcuts import render
import io
import matplotlib.pyplot as plt
import urllib, base64
from django.template.loader import get_template
from xhtml2pdf import pisa
from django.http import HttpResponse
import datetime
from django.contrib import messages
# Create your views here.

def computation(request):
    context ={}
    return render(request,'cvd/test.html',context)


    


def full_results(request):
    
    additional_prob_linearReg = request.session['additional_prob_linearReg']
    y_prob_additional_KNN = request.session['y_prob_additional_KNN']
    additional_prob_SVM = request.session['additional_prob_SVM']
    
    linearReg = f"{additional_prob_linearReg*100:.4f}"
    KNN_prob = f"{y_prob_additional_KNN*100:.4f}"
    SVM_prob = f"{additional_prob_SVM*100:.4f}"
    
    image_linearReg = request.session['image_linearReg'] 
    image_KNN =  request.session['image_KNN'] 
    image_SVM = request.session['image_SVM'] 
    
    accuracy_KNN = (request.session['accuracy_KNN'])*100
    accuracy_linearReg = (request.session['accuracy_linearReg'])*100
    accuracy_SVM = (request.session['accuracy_SVM'])*100
    
    gender_disp = ""
    history_disp = ""
 
    accuracy_KNN_disp = f"{accuracy_KNN:.2f}"
    accuracy_linearReg_disp = f"{accuracy_linearReg:.2f}" 
    accuracy_SVM_disp = f"{accuracy_SVM:.2f}" 
    age = request.session['age']
    gender = request.session['gender']
    trestbps = request.session['trestbps']
    heart_disease = request.session['heart_disease']
    cp = request.session['cp']
    chol = request.session['chol']
    fbs = request.session['fbs']
    restecg = request.session['restecg']
    thalac = request.session['thalac']
    thal = request.session['thal']
    
    if gender == 0:
        gender_disp = "Male"
    else:
        gender_disp = "Female"   
    
    if heart_disease == 0:
        history_disp = "No"
    else:
        history_disp = "Yes"   
    url = f"{age}-{gender}-{trestbps}-{heart_disease}-{cp}-{chol}-{fbs}-{restecg}-{thalac}-{thal}"
    
    context ={'image_SVM':image_SVM, 'image_linearReg':image_linearReg, 'image_KNN':image_KNN, 'accuracy_KNN':accuracy_KNN_disp,
              'accuracy_linearReg':accuracy_linearReg_disp, 'accuracy_SVM':accuracy_SVM_disp,
              'age':age, 'gender':gender_disp, 'trestbps':trestbps,'heart_disease':history_disp,'cp':cp,
              'chol':chol,'fbs':fbs,'restecg': restecg, 'thal':thal, 'pk':url,
              'linearReg':linearReg , 'KNN_prob':KNN_prob, 'SVM_prob':SVM_prob}
    return render(request,'cvd/full_result.html', context)


def result(request):
    #s1 = request.session['s1']
    #s2 = request.session['s2']
    #s3 = request.session['s3']
    
    additional_prob_linearReg = request.session['additional_prob_linearReg']
    y_prob_additional_KNN = request.session['y_prob_additional_KNN']
    additional_prob_SVM = request.session['additional_prob_SVM']
    
    accuracy_KNN = request.session['accuracy_KNN']
    accuracy_linearReg = request.session['accuracy_linearReg']
    accuracy_SVM = request.session['accuracy_SVM']
    
    user_prob_KNN_val = request.session['user_prob_KNN_val']
    user_prob_LinearReg_val = request.session['user_prob_LinearReg_val']
    user_prob_SVM = request.session['user_prob_SVM']
    
    
    high_risk = request.session['high_risk']
    
    user_final_prob = 0
    risk = " "
    variables = {
        'KNN': accuracy_KNN,
        'Linear Regression': accuracy_linearReg,
        'SVM': accuracy_SVM
    }

    # Find the variable with the highest value
    highest_variable = max(variables, key=variables.get)
    highest_value = variables[highest_variable]
    
    if highest_variable == "KNN":
        user_final_prob =f"{y_prob_additional_KNN*100:.2f}%"
    elif highest_variable == "SVM":
        user_final_prob =  f"{additional_prob_SVM*100:.2f}%"
    else:
        user_final_prob = f"{additional_prob_linearReg*100:.2f}%"
    
    if high_risk:
        risk = " High Chance of CVD "
    else:
        risk = "Chance of CVD"
    context={'risk':risk, 'user_prob': user_final_prob}
   
    return render(request,'cvd/results.html', context)

def greater_than_35(request):
    high_risk = False

    additional_prob_linearReg= 0
    y_prob_additional_KNN=0
    additional_prob_SVM =0
    age = 0
    if request.method == "POST":

    
        # Get set of additional variables from the user
        chol = float(request.POST.get('chol'))
        fbs = float(request.POST.get('fbs'))
        restecg = float(request.POST.get('restecg'))
        thalac = float(request.POST.get('thalac'))
        thal = float(request.POST.get('thal'))
        
        request.session['chol']  = chol 
        request.session['fbs']  = fbs
        request.session['restecg']  = restecg 
        request.session['thalac']  = thalac 
        request.session['thal']  = thal  
        
        #prob = request.session['prob']
        age = request.session['age']
        gender = request.session['gender']
        trestbps = request.session['trestbps']
        heart_disease = request.session['heart_disease']
        cp = request.session['cp']
        
        
        user_prob_KNN_val = request.session['user_prob_KNN_val']
        user_prob_LinearReg_val = request.session['user_prob_LinearReg_val']
        user_prob_SVM= request.session['user_prob_SVM']
        
        
        # Load the dataset
        file_path = "heart_attack.csv"
        data = pd.read_csv(file_path)

        #-------Linear Regression-----------
        # Select features (X) and target variable (y)
        features = ['age', 'gender', 'trestbps', 'heart_disease', 'cp']
        X = data[features]
        y = data['heart_disease']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features (optional but often recommended)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create a logistic regression model
        model = LogisticRegression()

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    

        
        
     

        # Create a DataFrame with the additional variables
        additional_data = pd.DataFrame({
            'age': [age],  # Include 'age' for consistency with the original features
            'gender': [gender],  # Include 'gender' for consistency with the original features
            'trestbps': [trestbps],  # Include 'trestbps' for consistency with the original features
            'heart_disease': [heart_disease],  # Include 'heart_disease' for consistency with the original features
            'cp': [cp],  # Include 'cp' for consistency with the original features
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalac': [thalac],
            'thal': [thal],
        })

        # Reorder columns to match the order during model training
        additional_data = additional_data[X.columns]

        # Standardize the additional variables
        additional_data_scaled = scaler.transform(additional_data)

        # Make a prediction with the additional variables
        additional_prob_linearReg = model.predict_proba(additional_data_scaled)[:, 1]

        request.session['additional_prob_linearReg'] = additional_prob_linearReg[0]
        
        # Display the result of the additional prediction
        print(f"Additional Predicted Probability: {additional_prob_linearReg[0]:.4f}")

        # Additional User Evaluation
        y_additional = model.predict(additional_data_scaled)
        mse_additional = mean_squared_error(y_additional, [1])  # Assuming the user has heart disease
        rmse_additional = np.sqrt(mse_additional)

        # Display Additional User Evaluation
        print("\nAdditional User Evaluation:")
        print(f"Mean Squared Error: {mse_additional:.4f}")
        print(f"Root Mean Squared Error: {rmse_additional:.4f}")

        # Make predictions on the test set
        y_pred_test = model.predict(X_test_scaled)
        y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate the model on the test set
        accuracy_linearReg = accuracy_score(y_test, y_pred_test)
        conf_matrix_test_linearReg = confusion_matrix(y_test, y_pred_test)
        precision_recall_f1_test = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
     
        request.session['accuracy_linearReg'] = accuracy_linearReg
        
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix_test_linearReg, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Add labels to the plot
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])  # Adjust labels based on your classes
        plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])  # Adjust labels based on your classes

        # Convert the plot to a PNG image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png_linearReg = buffer.getvalue()
        buffer.close()

        # Encode the PNG image to base64 to embed in HTML
        graphic = base64.b64encode(image_png_linearReg).decode('utf-8')

        # Embed the base64 encoded image in an HTML img tag
        image_linearReg = f'data:image/png;base64,{graphic}'
        
        request.session['image_linearReg'] = image_linearReg
        #if additional_prob - prob >= 10:
        #---------end linear regression------------------
        
        #--------- KNN-----------------------------------
        # Select features (X) and target variable (y)
        features = ['age', 'gender', 'trestbps', 'heart_disease', 'cp']
        X = data[features]
        y = data['heart_disease']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features (important for KNN)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create a KNN classifier
        knn_model = KNeighborsClassifier(n_neighbors=200)  # You can adjust the number of neighbors as needed

        # Train the KNN model
        knn_model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = knn_model.predict(X_test_scaled)
        y_prob = knn_model.predict_proba(X_test_scaled)[:, 1]
        # Create a DataFrame with the additional variables
        additional_data = pd.DataFrame({
            'age': [age],  # Include 'age' for consistency with the original features
            'gender': [gender],  # Include 'gender' for consistency with the original features
            'trestbps': [trestbps],  # Include 'trestbps' for consistency with the original features
            'heart_disease': [heart_disease],  # Include 'heart_disease' for consistency with the original features
            'cp': [cp],  # Include 'cp' for consistency with the original features
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalac': [thalac],
            'thal': [thal],
        })

        # Standardize the additional variables
        additional_data_scaled = scaler.transform(additional_data.drop(columns=['chol', 'fbs', 'restecg', 'thalac', 'thal']))

        # Make predictions for the additional user data
        y_additional = knn_model.predict(additional_data_scaled)
        y_prob_additional_KNN = knn_model.predict_proba(additional_data_scaled)[:, 1]

        request.session['y_prob_additional_KNN'] = y_prob_additional_KNN[0]
        
        # Display the result of the additional prediction
        print(f"Additional Predicted Probability: {y_prob_additional_KNN[0]:.4f}")
        
        # Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for additional user evaluation
        mse_additional = ((y_additional - [0]) ** 2).mean()
        rmse_additional = np.sqrt(mse_additional)

        # Display Additional User Evaluation
        print("\nAdditional User Evaluation:")
        print(f"User Predicted Class: {y_additional[0]}")
        print(f"User Predicted Probability: {y_prob_additional_KNN[0]:.4f}")
        print(f"Mean Squared Error: {mse_additional:.4f}")
        print(f"Root Mean Squared Error: {rmse_additional:.4f}")

        # Test Set Evaluation
        y_pred_test = knn_model.predict(X_test_scaled)

        # Calculate Accuracy, Confusion Matrix, Precision, Recall, and F1 Score for the test set
        accuracy_test_KNN = accuracy_score(y_test, y_pred_test)
        conf_matrix_test_KNN = confusion_matrix(y_test, y_pred_test)
        precision_recall_f1_test = precision_recall_fscore_support(y_test, y_pred_test, average='binary')

        request.session['accuracy_KNN'] = accuracy_test_KNN
        
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix_test_KNN, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Add labels to the plot
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])  # Adjust labels based on your classes
        plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])  # Adjust labels based on your classes

        # Convert the plot to a PNG image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png_KNN = buffer.getvalue()
        buffer.close()

        # Encode the PNG image to base64 to embed in HTML
        graphic = base64.b64encode(image_png_KNN).decode('utf-8')

        # Embed the base64 encoded image in an HTML img tag
        image_KNN = f'data:image/png;base64,{graphic}'
        
        request.session['image_KNN'] = image_KNN
        #------------END KNN ----------------------------

         #------------SVM -------------------------------
         
         # Create a DataFrame with the additional variables
         # Load the dataset
        file_path = "heart_attack.csv"
        data = pd.read_csv(file_path)

        # Select features (X) and target variable (y)
        features = ['age', 'gender', 'trestbps', 'heart_disease', 'cp']
        X = data[features]
        y = data['heart_disease']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features (important for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create an SVM classifier
        svm_model = SVC(probability=True)  # Note: probability=True enables probability estimates

        # Train the SVM model
        svm_model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = svm_model.predict(X_test_scaled)
        y_prob = svm_model.decision_function(X_test_scaled)  # Use decision function for confidence measure
        
        # Create a DataFrame with the additional variables
        additional_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'trestbps': [trestbps],
            'heart_disease': [heart_disease],
            'cp': [cp],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
        })

        # Reorder columns to match the order during model training
        additional_data = additional_data[X.columns]

        # Standardize the additional variables
        additional_data_scaled = scaler.transform(additional_data)

        # Make a prediction with the additional variables
        additional_prob_SVM = svm_model.predict_proba(additional_data_scaled)[:, 1]

        # Display the result of the additional prediction
        print(f"Additional Predicted Probability: {additional_prob_SVM[0]:.4f}")

        # Additional User Evaluation
        y_additional = svm_model.predict(additional_data_scaled)
        mse_additional = mean_squared_error(y_additional, [0])
        rmse_additional = np.sqrt(mse_additional)

        # Display Additional User Evaluation
        print("\nAdditional User Evaluation:")
        print(f"User Predicted Class: {y_additional[0]}")
        print(f"User Predicted Probability: {additional_prob_SVM[0]:.4f}")
        print(f"Mean Squared Error: {mse_additional:.4f}")
        print(f"Root Mean Squared Error: {rmse_additional:.4f}")
        
        # Test Set Evaluation
        y_pred_test = svm_model.predict(X_test_scaled)

        # Calculate Accuracy, Confusion Matrix, Precision, Recall, and F1 Score for the test set
        accuracy_test_SVM = accuracy_score(y_test, y_pred_test)
        conf_matrix_test_SVM = confusion_matrix(y_test, y_pred_test)
        precision_recall_f1_test = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
        
        
        request.session['additional_prob_SVM'] = additional_prob_SVM[0]
        request.session['accuracy_SVM'] = accuracy_test_SVM
        
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix_test_SVM, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Add labels to the plot
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])  # Adjust labels based on your classes
        plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])  # Adjust labels based on your classes

        # Convert the plot to a PNG image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png_SVM = buffer.getvalue()
        buffer.close()

        # Encode the PNG image to base64 to embed in HTML
        graphic = base64.b64encode(image_png_SVM).decode('utf-8')

        # Embed the base64 encoded image in an HTML img tag
        image_SVM = f'data:image/png;base64,{graphic}'
        
        request.session['image_SVM'] = image_SVM
         #---------------END SVM-------------------------
         
        s1 = float(additional_prob_linearReg[0]) - float(user_prob_LinearReg_val) 
        request.session['s1'] = s1
        s2 = float(y_prob_additional_KNN[0]) - float(user_prob_KNN_val) 
        request.session['s2'] = s1
        s3 = float(additional_prob_SVM) - float(user_prob_SVM) 
        request.session['s3'] = s3
         #check for 10% increase
        if s1 >= 10 or s2 >=10 or s3 >=10:
            high_risk = True
        
      
        
        request.session['high_risk'] = high_risk
        
        #return HttpResponse(f"{additional_prob_linearReg[0]} - {user_prob_LinearReg_val} = {s1} \n {y_prob_additional_KNN[0]} - {user_prob_KNN_val} = {s2}\n{additional_prob_SVM} - {user_prob_SVM} = {s3}")
        return redirect('result')
         
    context ={}
    
    return render(request,"cvd/greater_than_35.html",context)
 
def less_than_35(request):
    
    if request.method == "POST":
        # Get set of additional variables from the user
        chol = float(request.POST.get('chol'))
        fbs = float(request.POST.get('fbs'))
        restecg = float(request.POST.get('restecg'))
        
        #prob = request.session['prob']
        age = request.session['age']
        gender = request.session['gender']
        trestbps = request.session['trestbps']
        heart_disease = request.session['heart_disease']
        cp = request.session['cp']
    # Load the dataset
        file_path = "heart_attack.csv"
        data = pd.read_csv(file_path)

        #------------------LINEAR REGRESSION -----------------------------
        
        # Select features (X) and target variable (y)
        features = ['age', 'gender', 'trestbps', 'heart_disease', 'cp']
        X = data[features]
        y = data['heart_disease']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features (optional but often recommended)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create a logistic regression model
        model = LogisticRegression()

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        

        # Create a DataFrame with the additional variables
        additional_data = pd.DataFrame({
            'age': [age],  # Include 'age' for consistency with the original features
            'gender': [gender],  # Include 'gender' for consistency with the original features
            'trestbps': [trestbps],  # Include 'trestbps' for consistency with the original features
            'heart_disease': [heart_disease],  # Include 'heart_disease' for consistency with the original features
            'cp': [cp],  # Include 'cp' for consistency with the original features
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
        })

        # Reorder columns to match the order during model training
        additional_data = additional_data[X.columns]

        # Standardize the additional variables
        additional_data_scaled = scaler.transform(additional_data)

        # Make a prediction with the additional variables
        additional_prob_linearReg = model.predict_proba(additional_data_scaled)[:, 1]

        request.session['additional_prob_linearReg'] = additional_prob_linearReg[0]
        
        # Display the result of the additional prediction
        print(f"Additional Predicted Probability: {additional_prob_linearReg[0]:.4f}")

        # Additional User Evaluation
        y_additional = model.predict(additional_data_scaled)
        mse_additional = mean_squared_error(y_additional, [1])  # Assuming the user has heart disease
        rmse_additional = np.sqrt(mse_additional)

        # Display Additional User Evaluation
        print("\nAdditional User Evaluation:")
        print(f"Mean Squared Error: {mse_additional:.4f}")
        print(f"Root Mean Squared Error: {rmse_additional:.4f}")

        # Make predictions on the test set
        y_pred_test = model.predict(X_test_scaled)
        y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate the model on the test set
        accuracy_linearReg = accuracy_score(y_test, y_pred_test)
        conf_matrix_test_linearReg = confusion_matrix(y_test, y_pred_test)
        precision_recall_f1_test = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
     
        request.session['accuracy_linearReg'] = accuracy_linearReg
        
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix_test_linearReg, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Add labels to the plot
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])  # Adjust labels based on your classes
        plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])  # Adjust labels based on your classes

        # Convert the plot to a PNG image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png_linearReg = buffer.getvalue()
        buffer.close()

        # Encode the PNG image to base64 to embed in HTML
        graphic = base64.b64encode(image_png_linearReg).decode('utf-8')

        # Embed the base64 encoded image in an HTML img tag
        image_linearReg = f'data:image/png;base64,{graphic}'
        
        request.session['image_linearReg'] = image_linearReg
        #----------------END LINEAR REGRESSION --------------------------

        #---------------- KNN ------------------------------------------
        # Select features (X) and target variable (y)
        features = ['age', 'gender', 'trestbps', 'heart_disease', 'cp']
        X = data[features]
        y = data['heart_disease']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features (important for KNN)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create a KNN classifier
        knn_model = KNeighborsClassifier(n_neighbors=200)  # You can adjust the number of neighbors as needed

        # Train the KNN model
        knn_model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = knn_model.predict(X_test_scaled)
        y_prob = knn_model.predict_proba(X_test_scaled)[:, 1]
        
 

        # Create a DataFrame with the additional variables
        additional_data = pd.DataFrame({
            'age': [age],  # Include 'age' for consistency with the original features
            'gender': [gender],  # Include 'gender' for consistency with the original features
            'trestbps': [trestbps],  # Include 'trestbps' for consistency with the original features
            'heart_disease': [heart_disease],  # Include 'heart_disease' for consistency with the original features
            'cp': [cp],  # Include 'cp' for consistency with the original features
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
        })

        # Standardize the additional variables
        additional_data_scaled = scaler.transform(additional_data.drop(columns=['chol', 'fbs', 'restecg', 'thalac', 'thal']))

        # Make predictions for the additional user data
        y_additional = knn_model.predict(additional_data_scaled)
        y_prob_additional_KNN = knn_model.predict_proba(additional_data_scaled)[:, 1]

        request.session['y_prob_additional_KNN'] = y_prob_additional_KNN[0]
        
        # Display the result of the additional prediction
        print(f"Additional Predicted Probability: {y_prob_additional_KNN[0]:.4f}")
        
        # Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for additional user evaluation
        mse_additional = ((y_additional - [0]) ** 2).mean()
        rmse_additional = np.sqrt(mse_additional)

        # Display Additional User Evaluation
        print("\nAdditional User Evaluation:")
        print(f"User Predicted Class: {y_additional[0]}")
        print(f"User Predicted Probability: {y_prob_additional_KNN[0]:.4f}")
        print(f"Mean Squared Error: {mse_additional:.4f}")
        print(f"Root Mean Squared Error: {rmse_additional:.4f}")

        # Test Set Evaluation
        y_pred_test = knn_model.predict(X_test_scaled)

        # Calculate Accuracy, Confusion Matrix, Precision, Recall, and F1 Score for the test set
        accuracy_test_KNN = accuracy_score(y_test, y_pred_test)
        conf_matrix_test_KNN = confusion_matrix(y_test, y_pred_test)
        precision_recall_f1_test = precision_recall_fscore_support(y_test, y_pred_test, average='binary')

        request.session['accuracy_KNN'] = accuracy_test_KNN
        
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix_test_KNN, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Add labels to the plot
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])  # Adjust labels based on your classes
        plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])  # Adjust labels based on your classes

        # Convert the plot to a PNG image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png_KNN = buffer.getvalue()
        buffer.close()

        # Encode the PNG image to base64 to embed in HTML
        graphic = base64.b64encode(image_png_KNN).decode('utf-8')

        # Embed the base64 encoded image in an HTML img tag
        image_KNN = f'data:image/png;base64,{graphic}'
        
        request.session['image_KNN'] = image_KNN
        # ------------------ END KNN -----------------------------------
        
        #---------------------------SVM -------------------------------
        #Select features (X) and target variable (y)
        features = ['age', 'gender', 'trestbps', 'heart_disease', 'cp']
        X = data[features]
        y = data['heart_disease']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features (important for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create an SVM classifier
        svm_model = SVC(probability=True)  # Note: probability=True enables probability estimates

        # Train the SVM model
        svm_model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = svm_model.predict(X_test_scaled)
        y_prob = svm_model.decision_function(X_test_scaled)  # Use decision function for confidence measure
        
        # Get another set of additional variables from the user

        # Create a DataFrame with the additional variables
        additional_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'trestbps': [trestbps],
            'heart_disease': [heart_disease],
            'cp': [cp],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
        })

         # Reorder columns to match the order during model training
        additional_data = additional_data[X.columns]

        # Standardize the additional variables
        additional_data_scaled = scaler.transform(additional_data)

        # Make a prediction with the additional variables
        additional_prob_SVM = svm_model.predict_proba(additional_data_scaled)[:, 1]

        # Display the result of the additional prediction
        print(f"Additional Predicted Probability: {additional_prob_SVM[0]:.4f}")

        # Additional User Evaluation
        y_additional = svm_model.predict(additional_data_scaled)
        mse_additional = mean_squared_error(y_additional, [0])
        rmse_additional = np.sqrt(mse_additional)

        # Display Additional User Evaluation
        print("\nAdditional User Evaluation:")
        print(f"User Predicted Class: {y_additional[0]}")
        print(f"User Predicted Probability: {additional_prob_SVM[0]:.4f}")
        print(f"Mean Squared Error: {mse_additional:.4f}")
        print(f"Root Mean Squared Error: {rmse_additional:.4f}")
        
        # Test Set Evaluation
        y_pred_test = svm_model.predict(X_test_scaled)

        # Calculate Accuracy, Confusion Matrix, Precision, Recall, and F1 Score for the test set
        accuracy_test_SVM = accuracy_score(y_test, y_pred_test)
        conf_matrix_test_SVM = confusion_matrix(y_test, y_pred_test)
        precision_recall_f1_test = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
        
        
        request.session['additional_prob_SVM'] = additional_prob_SVM[0]
        request.session['accuracy_SVM'] = accuracy_test_SVM
        
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix_test_SVM, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Add labels to the plot
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])  # Adjust labels based on your classes
        plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])  # Adjust labels based on your classes

        # Convert the plot to a PNG image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png_SVM = buffer.getvalue()
        buffer.close()

        # Encode the PNG image to base64 to embed in HTML
        graphic = base64.b64encode(image_png_SVM).decode('utf-8')

        # Embed the base64 encoded image in an HTML img tag
        image_SVM = f'data:image/png;base64,{graphic}'
        
        request.session['image_SVM'] = image_SVM
        #---------------------------END SVM ---------------------------
        return redirect('result')
        #ask if need pa nung 10% shit
    context = {}
    return render(request, 'cvd/less_than_35.html',context)

def home(request):
    context = {}
    return render(request,'cvd/home.html',context)

def preliminary_entry(request):
    user_prob_KNN_val = 0
    user_prob_LinearReg_val = 0
    user_prob_SVM = 0
    if request.method == "POST":
        
        
   
        #--------------------------Logistic Regression--------------------------
        # Load the dataset
        file_path = "heart_attack.csv"
        data = pd.read_csv(file_path)

        # Select features (X) and target variable (y)
        features = ['age', 'gender', 'trestbps', 'heart_disease', 'cp']
        X = data[features]
        y = data['heart_disease']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features (optional but often recommended)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create a logistic regression model
        model = LogisticRegression()

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Prompt the user for input
        age = request.POST.get('age')
        trestbps = request.POST.get('restbps')
        pre_heart_disease = request.POST.get('history')
        pre_gender = request.POST.get('sex')
        cp = request.POST.get('cp')
        if pre_gender == "male":   
            gender = 0 
            
        else:
            gender = 1
          
        
        
       
        
        pre_heart_disease = request.POST.get('history')
        
        if pre_heart_disease == "yes":
            
            heart_disease =1
        else:
            heart_disease = 0
       
        



        
    
        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'trestbps': [trestbps],
            'heart_disease': [heart_disease],
            'cp': [cp]
        })

        # Standardize the user input
        user_data_scaled = scaler.transform(user_data)

        # Make a prediction with the user input
        user_prob_LinearReg = model.predict_proba(user_data_scaled)[:, 1]
        user_prob_LinearReg_val = user_prob_LinearReg[0]
        request.session['user_prob_LinearReg_val'] = user_prob_LinearReg_val

        
        #--------------------------END Logistic Regression--------------------------
        request.session['age'] = age
        request.session['gender'] = gender 
        request.session['trestbps'] = trestbps
        request.session['heart_disease'] = heart_disease   
        request.session['cp'] = cp  
        #---------------------------KNN -------------------------------------------
        
        # Select features (X) and target variable (y)
        features = ['age', 'gender', 'trestbps', 'heart_disease', 'cp']
        X = data[features]
        y = data['heart_disease']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features (important for KNN)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create a KNN classifier
        knn_model = KNeighborsClassifier(n_neighbors=200)  # You can adjust the number of neighbors as needed

        # Train the KNN model
        knn_model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = knn_model.predict(X_test_scaled)
        y_prob = knn_model.predict_proba(X_test_scaled)[:, 1]
        
        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'trestbps': [trestbps],
            'heart_disease': [heart_disease],
            'cp': [cp]
        })

        # Standardize the user input
        user_data_scaled = scaler.transform(user_data)

        # Make predictions for the user
        user_pred = knn_model.predict(user_data_scaled)
        user_prob_KNN = knn_model.predict_proba(user_data_scaled)[:, 1]
        user_prob_KNN_val = user_prob_KNN[0]
        
        request.session['user_prob_KNN_val'] = user_prob_KNN_val
        

        
        #------------------------ END KNN-----------------------
       
        #------------------------------- SVM -------------------
        # Load the dataset
        file_path = "heart_attack.csv"
        data = pd.read_csv(file_path)

        # Select features (X) and target variable (y)
        features = ['age', 'gender', 'trestbps', 'heart_disease', 'cp']
        X = data[features]
        y = data['heart_disease']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create an SVM classifier with probability estimates
        svm_model = SVC(kernel='linear', probability=True)

        # Train the SVM model
        svm_model.fit(X_train_scaled, y_train)


        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'trestbps': [trestbps],
            'heart_disease': [heart_disease],
            'cp': [cp]
        })

        # Standardize the user input
        user_data_scaled = scaler.transform(user_data)

        # Make a prediction with the user input
        user_prob_SVM = svm_model.predict_proba(user_data_scaled)[:, 1]

        # Display the user's predicted probability
        print(f"\nUser Predicted Probability: {user_prob_SVM[0]:.4f}")

        request.session['user_prob_SVM'] = user_prob_SVM[0]
        
            

        #--------------------END SVM ------------------------
        if float(user_prob_KNN) > 0.35 or float(user_prob_LinearReg) > 0.35 or float(user_prob_SVM) > 0.35:
    
            return redirect('greater-than')
          
        else:
            return redirect('less-than-35')
          
     
        #-----------------------------------------------------------------------
    
    
    context = {}
    return render(request, "cvd/prelim.html", context)

def sample(request):
    today = datetime.date.today()
    age = request.session['age']
    gender = request.session['gender']
    trestbps = request.session['trestbps']
    heart_disease = request.session['heart_disease']
    cp = request.session['cp']
    chol = request.session['chol']
    fbs = request.session['fbs']
    restecg = request.session['restecg']
    thalac = request.session['thalac']
    thal = request.session['thal']
    
    gender_disp = ""
    
    if gender == 0:
        gender_disp = "Male"
    else:
        gender_disp = "Female"    
    additional_prob_linearReg = request.session['additional_prob_linearReg']
    y_prob_additional_KNN = request.session['y_prob_additional_KNN']
    additional_prob_SVM = request.session['additional_prob_SVM']
    
    linearReg = f"{additional_prob_linearReg*100:.4f}"
    KNN_prob = f"{y_prob_additional_KNN*100:.4f}"
    SVM_prob = f"{additional_prob_SVM*100:.4f}"
    
 
    
    if float(additional_prob_linearReg) > 40:
        recommendation = " Immediate medical attention is need for further medical analysis of the patient"
    
    else:
        recommendation = " No immediate medical attention is need but seek medical advice for clarification"
    
    context = {'today': today,
            'age': age,
            'recommendation':recommendation,
            'gender': gender_disp,
            'trestbps': trestbps,
            'heart_disease': heart_disease,
            'cp': cp,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalac': thalac,
            'thal': thal, 'KNN_prob':KNN_prob, 'SVM_prob':SVM_prob,
              'linearReg':linearReg
                   }
    template_path = 'cvd/result_pdf.html'
    file_date = datetime.datetime.now()
    filename = f"CVD_Analysis:{file_date}.pdf"
       # Create a Django response object, and specify content_type as pdf
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename={filename}'
    #response['Content-Disposition'] = 'filename= f"payment_reports{today}.pdf"'
    # find the template and render it.
    template = get_template(template_path)
    html = template.render(context)

    # create a pdf
    pisa_status = pisa.CreatePDF(
       html, dest=response)
    # if error then show some funny view
    if pisa_status.err:
       return HttpResponse('We had some errors <pre>' + html + '</pre>')
    return response
    return render(request, "cvd/test.html",context)

"""def print(request):
    template_path = 'cvd/result_pdf.html'
    
    file_date = datetime.datetime.now()
    today = datetime.date.today()
    age = request.session['age']
    gender = request.session['gender']
    trestbps = request.session['trestbps']
    heart_disease = request.session['heart_disease']
    cp = request.session['cp']
    chol = request.session['chol']
    fbs = request.session['fbs']
    restecg = request.session['restecg']
    thalac = request.session['thalac']
    thal = request.session['thal']
    
    context = {'today': today,
            'age': age,
            'gender': gender,
            'trestbps': trestbps,
            'heart_disease': heart_disease,
            'cp': cp,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalac': thalac,
            'thal': thal
                   }
     
    filename = f"CVD_Analysis:{file_date}.pdf"
    context = {'today': today,
            'age': age,
            'gender': gender,
            'trestbps': trestbps,
            'heart_disease': heart_disease,
            'cp': cp,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalac': thalac,
            'thal': thal
                   }
    # Create a Django response object, and specify content_type as pdf
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename={filename}'
    #response['Content-Disposition'] = 'filename= f"payment_reports{today}.pdf"'
    # find the template and render it.
    template = get_template(template_path)
    html = template.render(context)

    # create a pdf
    pisa_status = pisa.CreatePDF(
       html, dest=response)
    # if error then show some funny view
    if pisa_status.err:
       return HttpResponse('We had some errors <pre>' + html + '</pre>')
    return response"""
    
  
    
