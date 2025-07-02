from django.shortcuts import render,redirect
from admins.views import viewuser
from django.contrib import messages
from .models import Reg_User

# Create your views here.
def index(request):
    return render(request, 'base.html')

def register(request):
    return render(request, 'user_register.html')

def ActivateUser(request, id):
    if request.method == 'GET':
        if id is not None:
            status = 'Activated'
            print("PID = ", id, status)
            Reg_User.objects.filter(id=id).update(status=status)
        return redirect(viewuser)

def BlockUser(request, id):
    if request.method == 'GET':
        if id is not None:
            status = 'Waiting'
            print("PID = ", id, status)
            Reg_User.objects.filter(id=id).update(status=status)
        return redirect(viewuser)
    
def userbase(request):
    return render(request,'users/UserHomePage.html')

def userlogin(request):
    if request.method == 'POST':
        username = request.POST.get('uname')
        password = request.POST.get('psw')
        print(f"Username: {username}, Password: {password}")
        try:
            data=Reg_User.objects.get(username=username, password=password)
            if data.status == 'Activated':
                return redirect('userbase')
            else:
                messages.error(request, 'Invalid Credentials')
        except Exception as e:
            print(f'the exception is {e}')
            messages.error(request, f'Invalid Credentials {str(e)}')

        
    return render(request, 'userlogin.html')

#==========================================================================================

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def training(request):
    # Load Data
    data = pd.read_csv(r'media/FlightDelay.csv')
    data.fillna('1', inplace=True)  # Fill NaN values with '1'

    # Create directories
    encoder_dir = "media/encoders"
    model_dir = "media/models"
    cm_dir = "media/confusion_matrices"
    vis_dir = "media/visualizations"  # New directory for visualizations

    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Encode Categorical Data
    categorical_cols = ['FL_DATE', 'OP_UNIQUE_CARRIER', 'OP_CARRIER', 'TAIL_NUM',
                        'ORIGIN', 'DEST', 'DEP_TIME', 'TAXI_OUT', 'WHEELS_OFF',
                        'WHEELS_ON', 'TAXI_IN', 'ARR_TIME', 'CANCELLED', 'DISTANCE', 'ARR_DELAY_GROUP']

    for col in categorical_cols:
        encoder_path = os.path.join(encoder_dir, f"{col}_encoder.pkl")
        le = LabelEncoder()
        data[col] = data[col].astype(str)

        if os.path.exists(encoder_path):
            with open(encoder_path, "rb") as f:
                le = pickle.load(f)
        else:
            le.fit(data[col])
            with open(encoder_path, "wb") as f:
                pickle.dump(le, f)

        data[col] = le.transform(data[col])

    # Split Features and Target
    X = data.drop(['ARR_DELAY_GROUP'], axis=1)
    y = data['ARR_DELAY_GROUP']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define Classification Models
    models = {
        "Logistic Regression": LogisticRegression(solver='saga', max_iter=500),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    accuracies = {}
    confusion_matrices = {}

    # Train, Evaluate, and Save Models
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        accuracies[model_name] = round(accuracy * 100, 2)

        # Save trained model
        model_path = os.path.join(model_dir, f"{model_name.replace(' ', '_')}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Compute Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        confusion_matrices[model_name] = cm

        # Plot Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix ({model_name})")
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        cm_filename = f"{cm_dir}/{model_name.replace(' ', '_')}_cm.png"
        plt.savefig(cm_filename)
        plt.show()
        plt.close()

    # --- Generate Bar Graph for ARR_DELAY_GROUP ---
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title("Distribution of ARR_DELAY_GROUP")
    plt.xlabel("Delay Group")
    plt.ylabel("Count")
    bar_chart_path = os.path.join(vis_dir, "bar_chart.png")
    plt.savefig(bar_chart_path)
    plt.show()
    plt.close()

    # --- Generate Pie Chart for ARR_DELAY_GROUP ---
    plt.figure(figsize=(8, 6))
    y.value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap="Set2")
    plt.title("Proportion of ARR_DELAY_GROUP")
    plt.ylabel("")
    pie_chart_path = os.path.join(vis_dir, "pie_chart.png")
    plt.savefig(pie_chart_path)
    plt.show()
    plt.close()

    return render(request, "users/training.html", {
        "accuracies": accuracies,
        "confusion_matrices": confusion_matrices,
        "bar_chart": bar_chart_path,
        "pie_chart": pie_chart_path
    })


#==================================================================
import os
import pickle
import pandas as pd
from django.shortcuts import render

def prediction(request):
    if request.method == 'POST':
        input_features = [
            'FL_DATE', 'OP_UNIQUE_CARRIER', 'OP_CARRIER', 'TAIL_NUM',
            'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN', 'DEST_AIRPORT_ID',
            'DEST', 'CRS_DEP_TIME', 'DEP_TIME', 'TAXI_OUT', 'WHEELS_OFF',
            'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'CANCELLED', 'DISTANCE'
        ]

        input_data = {feature: request.POST.get(feature, "") for feature in input_features}
        input_df = pd.DataFrame([input_data])

        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        model_path = "media/models/Random_Forest.pkl"
        if not os.path.exists(model_path):
            return render(request, 'users/predictForm.html', {'output': "Model not found!"})

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        encoder_dir = "media/encoders"
        for feature in input_features:
            encoder_path = os.path.join(encoder_dir, f"{feature}_encoder.pkl")
            if os.path.exists(encoder_path):
                with open(encoder_path, "rb") as f:
                    encoder = pickle.load(f)

                input_df[feature] = input_df[feature].astype(str)

                input_df[feature] = input_df[feature].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )

        try:
            prediction = model.predict(input_df)
        except Exception as e:
            return render(request, 'users/predictForm.html', {'output': f"Error in prediction: {str(e)}"})

        if prediction[0] == 0:
            result = "early_arrival"
        elif prediction[0] == 1:
            result = "ontime"
        else:
            result = "delayed"

        return render(request, 'users/predictForm.html', {'output': result})

    return render(request, 'users/predictForm.html')

