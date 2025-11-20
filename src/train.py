
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import mlflow

def main():
    # Загрузка обработанных данных
    train_data = pd.read_csv('data/processed/train.csv')  # Убедитесь, что файл существует
    X = train_data.drop('target', axis=1)
    y = train_data['target']

    # Инициализация MLflow
    mlflow.start_run()

    # Обучение модели
    model = LogisticRegression()
    model.fit(X, y)

    # Логирование параметров и метрик
    mlflow.log_param("model", "LogisticRegression")

    # Предсказание и расчет точности (accuracy)
    acc = model.score(X, y)
    mlflow.log_metric("accuracy", acc)

    # Сохранение модели и логирование артефакта
    joblib.dump(model, 'model.pkl')
    mlflow.log_artifact('model.pkl')

    # Завершение запуска MLflow
    mlflow.end_run()

if __name__ == '__main__':
    main()
