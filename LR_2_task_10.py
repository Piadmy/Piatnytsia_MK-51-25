import json
import numpy as np
import yfinance as yf
from sklearn import covariance, cluster

# Вхідний файл із символічними позначеннями компаній
input_file = 'company_symbol_mapping.json'

# Завантаження прив'язок символів компаній до їх повних назв
with open(input_file, 'r', encoding='utf-8') as f:
    company_symbols_map = json.load(f)

# Формування масивів символів і назв компаній
symbols, names = np.array(list(company_symbols_map.items())).T

# Завантаження архівних даних котирувань
start_date = '2003-07-03'
end_date = '2007-05-04'

quotes = []
valid_symbols = []
valid_names = []

for symbol, name in zip(symbols, names):
    try:
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        # Пропускаємо недоступні або порожні дані
        if data.empty or 'Open' not in data.columns or 'Close' not in data.columns:
            continue

        quotes.append(data)
        valid_symbols.append(symbol)
        valid_names.append(name)

    except Exception:
        continue

# Перетворення списків у масиви
valid_symbols = np.array(valid_symbols)
valid_names = np.array(valid_names)

# Перевірка наявності завантажених даних
if len(quotes) == 0:
    raise ValueError("Не вдалося завантажити жодних біржових даних.")

# Вилучення котирувань, що відповідають відкриттю та закриттю біржі
opening_quotes = []
closing_quotes = []

for quote in quotes:
    open_vals = np.ravel(quote['Open'].to_numpy())
    close_vals = np.ravel(quote['Close'].to_numpy())

    opening_quotes.append(open_vals)
    closing_quotes.append(close_vals)

# Вирівнювання довжин часових рядів
min_len = min(len(x) for x in opening_quotes)

opening_quotes = np.array([x[:min_len] for x in opening_quotes], dtype=float)
closing_quotes = np.array([x[:min_len] for x in closing_quotes], dtype=float)

# Обчислення різниці між двома видами котирувань
quotes_diff = closing_quotes - opening_quotes

# Нормалізація даних
X = quotes_diff.T   # форма: (кількість днів, кількість компаній)

# Обчислення стандартного відхилення для кожної компанії
stds = X.std(axis=0)

# Залишаємо лише ті ознаки, де стандартне відхилення коректне
valid_mask = np.isfinite(stds) & (stds > 0)

X = X[:, valid_mask]
valid_names = valid_names[valid_mask]
valid_symbols = valid_symbols[valid_mask]

# Повторна перевірка після фільтрації
if X.shape[1] == 0:
    raise ValueError("Після фільтрації не залишилося жодної компанії з коректними даними.")

# Нормалізація
X = X / X.std(axis=0)

# Створення графової моделі
edge_model = covariance.GraphicalLassoCV()

# Навчання моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Створення моделі кластеризації на основі поширення подібності
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
num_labels = labels.max()

# Виведення результатів кластеризації
print('\nКластеризація акцій на основі різниці між котируваннями відкриття та закриття:\n')
for i in range(num_labels + 1):
    print("Кластер", i + 1, "==>", ', '.join(valid_names[labels == i]))
