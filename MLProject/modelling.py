import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report ,f1_score
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from joblib import dump

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
 
 
# Create a new MLflow Experiment
mlflow.set_experiment("Online Training House price prediction")

dataset = sys.argv[1] if len(sys.argv) == 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean dataset.csv")
df = pd.read_csv(dataset)

LRModel = LinearRegression()

with mlflow.start_run():
    mlflow.autolog()

    X = df.drop(['price'],axis=1)
    y = df['price']

    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=42)

    LRModel.fit(X_train , y_train)

    result_3 = LRModel.predict(X_test)

    mlflow.log_metric("Mean Squared Error", mean_squared_error(y_test, result_3))
    mlflow.log_metric("R-squared", r2_score(y_test, result_3))

    print("Linear Regression Model Evaluation:")
    print("Mean Squared Error:", mean_squared_error(y_test, result_3))
    print("R-squared:", r2_score(y_test, result_3))

    # Simpan model ke file lokal
    dump(LRModel, "online_model_LinearRegression.joblib")


    mlflow.log_artifact("online_model_LinearRegression.joblib" , artifact_path="model_artifacts")


    mlflow.sklearn.log_model(
        sk_model = LRModel,
        artifact_path = "online_model_LinearRegression",
        input_example=X.iloc[:5]

    )
