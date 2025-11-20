
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    # Загрузка обработанных данных
    train_data = pd.read_csv('data/processed/train.csv')
    X = train_data.drop('target', axis=1)  # Предполагаем, что есть целевая переменная 'target'
    y = train_data['target']

    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Сохранение модели
    joblib.dump(model, 'model.pkl')

if __name__ == '__main__':
    main()
