import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import sys

dataset = sys.argv[1] if len(sys.argv) == 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean dataset.csv")
df = pd.read_csv(dataset)

LRModel = LinearRegression()

with mlflow.start_run():

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
 
    mlflow.sklearn.log_model(
        sk_model = LRModel,
        artifact_path = "online_model_LinearRegression",
        input_example=X.iloc[:5]

    )
