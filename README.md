# Kółko i Krzyżyk AI - Instrukcja Użytkownika

## Opis Projektu

Zaawansowana implementacja gry Kółko i Krzyżyk z czterema algorytmami AI:
- **Minimax** z Alpha-Beta Pruning (niepokonalny)
- **Reguły** - system ekspertowy
- **Q-learning** - niepokonalny agent uczenia się
- **MCTS** - Monte Carlo Tree Search

## Struktura Folderów

```
kolkokrzyzyk/
├── ai/                      # Wszystkie algorytmy AI
│   ├── gotowe_tabele/       # Wytrenowane modele do użycia
│   ├── q_tables/           # Foldery treningowe Q-learning
│   ├── agent_q_learning.py # Implementacja Q-learning
│   ├── minimax.py          # Algorytm Minimax
│   ├── reguly.py           # System reguł
│   ├── mcts.py             # Monte Carlo Tree Search
│   └── losowy_gracz.py     # Losowy gracz (do treningu)
├── gra/                    # Logika gry
│   └── logika.py           # Podstawowa mechanika gry
├── gui/                    # Interfejs użytkownika
│   ├── główne_okno.py      # Główne okno aplikacji
│   └── okno_wizualizacji.py # Wizualizacja drzewa Minimax
├── Narzędzia/              # Narzędzia do treningu i oceny
│   ├── evaluate_agent.py   # Ocena wydajności agentów
│   └── logs/               # Logi z oceny wydajności
├── main_final.py           # Główny plik do uruchomienia
└── requirements.txt        # Wymagane biblioteki
```

## Modele AI

### Gdzie Znajdują Się Modele

- **Gotowe modele**: `ai/gotowe_tabele/model.pkl`
- **Sesje treningowe**: `ai/q_tables/q_learning_table_X/model.pkl`
- **Każdy folder treningowy zawiera**:
  - `model.pkl` - wytrenowany model
  - `training.log` - szczegółowe logi treningu

### Jak Używać Modeli

**Główny program automatycznie ładuje model z**: `ai/gotowe_tabele/model.pkl`

Jeśli chcesz użyć innego modelu:
1. Skopiuj wybrany `model.pkl` do folderu `ai/gotowe_tabele/`
2. Uruchom program ponownie

## Trenowanie Modeli

### Proces Treningu Q-learning

Trening odbywa się w 4 fazach:

1. **Faza 1**: Uczenie od eksperta (Minimax) - 500 gier
2. **Faza 2**: Eksploracja (vs Random) - 20 x 5000 gier
3. **Faza 3**: Trening strategiczny (mieszani przeciwnicy) - 40 iteracji
4. **Faza 4**: Finalne dostrojenie (vs Minimax) - 10 iteracji

### Gdzie Trafiają Wytrenowane Modele

- **Nowe modele**: `ai/q_tables/q_learning_table_X/`
- **Najlepsze modele**: ręcznie kopiowane do `ai/gotowe_tabele/`

### Trenowanie Nowego Modelu

```bash
cd ai
python agent_q_learning.py
```

Model zostanie zapisany w nowym folderze `q_learning_table_X/`

## Logi i Monitoring

### Rodzaje Logów

1. **Logi treningu**: `ai/q_tables/q_learning_table_X/training.log`
   - Postęp treningu z czasem
   - Statystyki wygranych/przegranych
   - Rozmiar tabeli Q
   - Parametry uczenia

2. **Logi oceny**: `Narzędzia/logs/YYYY-MM-DD_HH-MM-SS.txt`
   - Wyniki testów przeciwko różnym przeciwnikom
   - Czas odpowiedzi
   - Analiza wydajności

### Gdzie Trafiają Logi

- **Logi treningu**: w folderze każdego modelu
- **Logi oceny**: w folderze `Narzędzia/logs/`

## Użycie Głównego Programu

### Uruchomienie

```bash
python main_final.py
```

### Wymagania

```bash
pip install -r requirements.txt
```

**Wymagane biblioteki**:
- PySide6 (interfejs graficzny)
- numpy (obliczenia)

### Funkcje Programu

1. **Wybór trybu gry**: Menu rozwijane z opcjami
   - Gracz vs Gracz
   - Gracz vs Minimax
   - Gracz vs Reguły
   - Gracz vs Q-learning
   - Gracz vs MCTS

2. **Losowy start**: AI lub gracz rozpoczyna losowo

3. **Wizualizacja**: Dla Minimax - zobacz drzewo decyzyjne

4. **Statystyki**: Wyświetlane w konsoli po każdej grze

### Umieszczanie Pliku .pkl

**Gdzie umieścić plik modelu**:
```
ai/gotowe_tabele/model.pkl
```

**Jeśli main nie znajduje modelu**:
1. Sprawdź czy plik nazywa się dokładnie `model.pkl`
2. Sprawdź czy jest w folderze `ai/gotowe_tabele/`
3. Jeśli nie ma folderu, utwórz go
4. Program wyświetli ostrzeżenie, ale będzie działał bez Q-learning

## Ocena Wydajności Modeli

### Narzędzie Oceny

```bash
cd Narzędzia
python evaluate_agent.py
```

### Co Testuje

- **Random**: 5000 gier (test bazowy)
- **Smart Random**: 5000 gier (preferuje środek/narożniki)
- **Reguły**: 3000 gier (system ekspertowy)
- **MCTS**: 2000 gier (Monte Carlo)
- **Minimax**: 2000 gier (test doskonałości)

### Wyniki

- **DOSKONAŁA**: 0% przegranych z Minimax
- **ZNAKOMITA**: >90% wygranych z Random
- **DOBRA**: >70% wygranych z Random
- **SŁABA**: <70% wygranych z Random

### Logi Oceny

Szczegółowe wyniki w: `Narzędzia/logs/YYYY-MM-DD_HH-MM-SS.txt`

## Rozwiązywanie Problemów

### Model Się Nie Ładuje

1. Sprawdź ścieżkę: `ai/gotowe_tabele/model.pkl`
2. Sprawdź czy plik nie jest uszkodzony
3. Spróbuj skopiować model z `ai/q_tables/q_learning_table_X/`

### Program Się Nie Uruchamia

1. Zainstaluj wymagania: `pip install -r requirements.txt`
2. Sprawdź wersję Python (>=3.8)
3. Upewnij się, że masz PySide6

### AI Nie Odpowiada

1. Sprawdź konsolę - mogą być błędy
2. Zrestartuj program
3. Sprawdź czy model jest poprawny

## Wskazówki

- **Najlepszy model**: Użyj modelu z najwyższym numerem sesji
- **Testowanie**: Regularne uruchamianie `evaluate_agent.py`
- **Wizualizacja**: Graj z Minimax aby zobaczyć drzewo decyzyjne
- **Statystyki**: Obserwuj wyniki w konsoli
- **Trening**: Dla nowych modeli trenuj minimum 2-3 godziny

## Podsumowanie

Program oferuje kompletną implementację AI do gry Kółko i Krzyżyk z zaawansowanymi algorytmami, systemem treningu i oceną wydajności. Użyj `main_final.py` do gry, a `evaluate_agent.py` do testowania modeli.