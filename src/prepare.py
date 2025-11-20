
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Загрузка данных
    data = pd.read_csv('data/raw/data.csv')  # Предполагаем, что есть raw данные
    # Сплит на train/test
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    # Сохранение обработанных данных
    train.to_csv('data/processed/train.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)

if __name__ == '__main__':
    main()
