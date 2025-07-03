import os
import sys
import time
import pickle
import random
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from gra.logika import StanGry
from ai.minimax import znajdz_najlepszy_ruch
from ai.reguly import znajdz_najlepszy_ruch as reguly_najlepszy_ruch
from ai.mcts import znajdz_najlepszy_ruch as mcts_najlepszy_ruch


def konfiguruj_logowanie(folder_logow: str):
    plik_logow = os.path.join(folder_logow, 'training.log')
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatowanie_pliku = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    formatowanie_konsoli = logging.Formatter('%(message)s')
    obsluga_pliku = logging.FileHandler(plik_logow, encoding='utf-8', mode='w')
    obsluga_pliku.setLevel(logging.INFO)
    obsluga_pliku.setFormatter(formatowanie_pliku)
    obsluga_konsoli = logging.StreamHandler(sys.stdout)
    obsluga_konsoli.setLevel(logging.INFO)
    obsluga_konsoli.setFormatter(formatowanie_konsoli)
    logger.addHandler(obsluga_pliku)
    logger.addHandler(obsluga_konsoli)
    return logger


def loguj_i_drukuj(logger, wiadomosc):
    logger.info(wiadomosc)


def smart_random_ruch(stan_gry: StanGry) -> Optional[Tuple[int, int]]:
    mozliwe_ruchy = stan_gry.otrzymaj_mozliwe_ruchy()
    if not mozliwe_ruchy:
        return None
    srodek = (1, 1)
    rogi = [(0, 0), (0, 2), (2, 0), (2, 2)]
    if srodek in mozliwe_ruchy:
        return srodek
    dostepne_rogi = [r for r in rogi if r in mozliwe_ruchy]
    if dostepne_rogi:
        return random.choice(dostepne_rogi)
    return random.choice(mozliwe_ruchy)


class AgentQLearning:
    def __init__(self, wspolczynnik_uczenia: float = 0.3, wspolczynnik_dyskontujacy: float = 0.95,
                 wspolczynnik_eksploracji: float = 0.1):
        self.wspolczynnik_uczenia = wspolczynnik_uczenia
        self.wspolczynnik_dyskontujacy = wspolczynnik_dyskontujacy
        self.epsilon = wspolczynnik_eksploracji
        self.tabela_q = defaultdict(float)

    def pobierz_klucz_stanu(self, plansza: np.ndarray) -> tuple:
        symetrie = []
        tymczasowa = plansza
        for _ in range(4):
            symetrie.append(tuple(tymczasowa.flatten()))
            tymczasowa = np.rot90(tymczasowa)
        tymczasowa = np.fliplr(plansza)
        for _ in range(4):
            symetrie.append(tuple(tymczasowa.flatten()))
            tymczasowa = np.rot90(tymczasowa)
        return min(symetrie)

    def wygrana_lub_blok(self, stan_gry: StanGry) -> Optional[Tuple[int, int]]:
        prawidlowe_ruchy = stan_gry.otrzymaj_mozliwe_ruchy()
        obecny_gracz = stan_gry.obecny_gracz
        for ruch in prawidlowe_ruchy:
            testowa_plansza = stan_gry.plansza.copy()
            testowa_plansza[ruch[0], ruch[1]] = obecny_gracz
            rzad, kolumna = ruch
            if (np.all(testowa_plansza[rzad, :] == obecny_gracz) or
                    np.all(testowa_plansza[:, kolumna] == obecny_gracz) or
                    (rzad == kolumna and np.all(np.diag(testowa_plansza) == obecny_gracz)) or
                    (rzad + kolumna == 2 and np.all(np.diag(np.fliplr(testowa_plansza)) == obecny_gracz))):
                return ruch
        przeciwnik = -obecny_gracz
        for ruch in prawidlowe_ruchy:
            testowa_plansza = stan_gry.plansza.copy()
            testowa_plansza[ruch[0], ruch[1]] = przeciwnik
            rzad, kolumna = ruch
            if (np.all(testowa_plansza[rzad, :] == przeciwnik) or
                    np.all(testowa_plansza[:, kolumna] == przeciwnik) or
                    (rzad == kolumna and np.all(np.diag(testowa_plansza) == przeciwnik)) or
                    (rzad + kolumna == 2 and np.all(np.diag(np.fliplr(testowa_plansza)) == przeciwnik))):
                return ruch
        return None

    def wybierz_akcje(self, stan_gry: StanGry, epsilon: float = 0.0, uzyj_heurystyk: bool = True,
                      uzyj_reguly: bool = False) -> Optional[
        Tuple[int, int]]:
        if uzyj_reguly:
            ruch_z_regul = reguly_najlepszy_ruch(stan_gry)
            if ruch_z_regul:
                return ruch_z_regul
        if uzyj_heurystyk:
            wymuszony_ruch = self.wygrana_lub_blok(stan_gry)
            if wymuszony_ruch:
                return wymuszony_ruch
        mozliwe_ruchy = stan_gry.otrzymaj_mozliwe_ruchy()
        if not mozliwe_ruchy:
            return None
        if random.random() < epsilon:
            return random.choice(mozliwe_ruchy)
        klucz_stanu = self.pobierz_klucz_stanu(stan_gry.plansza)
        wartosci_q = [(ruch, self.tabela_q.get((klucz_stanu, ruch), 0.0)) for ruch in mozliwe_ruchy]
        najlepsza_wartosc = max(wartosci_q, key=lambda x: x[1])[1]
        najlepsze_ruchy = [ruch for ruch, q in wartosci_q if q >= najlepsza_wartosc - 0.0001]
        return random.choice(najlepsze_ruchy)

    def aktualizuj(self, klucz_stanu: tuple, akcja: Tuple[int, int],
                   nagroda: float, nastepny_klucz_stanu: tuple, zakonczone: bool):
        obecne_q = self.tabela_q.get((klucz_stanu, akcja), 0.0)
        if zakonczone:
            cel = nagroda
        else:
            tymczasowa_plansza = np.array(nastepny_klucz_stanu).reshape(3, 3)
            mozliwe_nastepne_akcje = [
                (r, k) for r in range(3) for k in range(3) if tymczasowa_plansza[r, k] == 0
            ]
            nastepne_max = max([self.tabela_q.get((nastepny_klucz_stanu, a), 0.0) for a in
                                mozliwe_nastepne_akcje]) if mozliwe_nastepne_akcje else 0.0
            cel = nagroda + self.wspolczynnik_dyskontujacy * nastepne_max
        self.tabela_q[(klucz_stanu, akcja)] = obecne_q + self.wspolczynnik_uczenia * (cel - obecne_q)

    def zaladuj_tabele_q(self, nazwa_pliku: str):
        try:
            with open(nazwa_pliku, 'rb') as f:
                dane = pickle.load(f)
            if 'q_table' in dane:
                self.tabela_q = defaultdict(float, dane['q_table'])
                print(f"✅ Załadowano Q-table z {len(self.tabela_q)} wpisami")
            if 'version' in dane:
                print(f"📦 Wersja modelu: {dane['version']}")
            if 'is_perfect' in dane:
                print(f"🏆 Status: {'DOSKONAŁY' if dane['is_perfect'] else 'PRAWIE DOSKONAŁY'}")
        except FileNotFoundError:
            print(f"❌ Nie znaleziono pliku: {nazwa_pliku}")
        except Exception as e:
            print(f"❌ Błąd ładowania modelu: {e}")

    def znajdz_najlepszy_ruch(self, stan_gry: StanGry, uzyj_reguly: bool = False):
        return self.wybierz_akcje(stan_gry, epsilon=0.0, uzyj_heurystyk=True, uzyj_reguly=uzyj_reguly)

    @property
    def q_table(self):
        return self.tabela_q

    @q_table.setter
    def q_table(self, value):
        self.tabela_q = value

    @property
    def learning_rate(self):
        return self.wspolczynnik_uczenia

    @learning_rate.setter
    def learning_rate(self, value):
        self.wspolczynnik_uczenia = value

    @property
    def discount_factor(self):
        return self.wspolczynnik_dyskontujacy

    @discount_factor.setter
    def discount_factor(self, value):
        self.wspolczynnik_dyskontujacy = value


def graj_partie_batch(argumenty):
    tabela_q_agenta, typ_przeciwnika, epsilon, liczba_gier, id_workera, uzyj_heurystyk, uzyj_reguly = argumenty
    agent1 = AgentQLearning()
    agent1.tabela_q = defaultdict(float, tabela_q_agenta)
    agent2 = None
    if typ_przeciwnika == "self":
        agent2 = AgentQLearning()
        agent2.tabela_q = defaultdict(float, tabela_q_agenta)
    elif typ_przeciwnika == "smart_random":
        agent2 = None
    statystyki = {'wins': 0, 'losses': 0, 'draws': 0}
    aktualizacje_q = []
    for indeks_gry in range(liczba_gier):
        stan_gry = StanGry()
        id_gracza_agent1 = 1 if (typ_przeciwnika != "self" or indeks_gry % 2 == 0) else -1
        agenci = {1: agent1, -1: agent2}
        if typ_przeciwnika == "self" and id_gracza_agent1 == -1: agenci = {1: agent2, -1: agent1}
        historia = []
        while not stan_gry.czy_koniec_gry():
            klucz_stanu = agent1.pobierz_klucz_stanu(stan_gry.plansza)
            id_obecnego_gracza = stan_gry.obecny_gracz
            akcja = None
            if id_obecnego_gracza == id_gracza_agent1 or typ_przeciwnika == "self":
                obecny_agent = agenci[id_obecnego_gracza]
                akcja = obecny_agent.wybierz_akcje(stan_gry, epsilon, uzyj_heurystyk=uzyj_heurystyk,
                                                   uzyj_reguly=uzyj_reguly)
            else:
                if typ_przeciwnika == "minimax":
                    akcja = znajdz_najlepszy_ruch(stan_gry)
                elif typ_przeciwnika == "reguly" or typ_przeciwnika == "rules":
                    akcja = reguly_najlepszy_ruch(stan_gry)
                elif typ_przeciwnika == "mcts":
                    akcja = mcts_najlepszy_ruch(stan_gry, 2000)
                elif typ_przeciwnika == "smart_random":
                    akcja = smart_random_ruch(stan_gry)
                elif typ_przeciwnika == "random":
                    prawidlowe_ruchy = stan_gry.otrzymaj_mozliwe_ruchy()
                    akcja = random.choice(prawidlowe_ruchy) if prawidlowe_ruchy else None
            if akcja:
                historia.append({'state': klucz_stanu, 'action': akcja, 'player': id_obecnego_gracza})
                stan_gry.wykonaj_ruch(akcja[0], akcja[1])
            else:
                break
        zwyciezca = stan_gry.sprawdz_zwyciezce()
        if zwyciezca == id_gracza_agent1:
            statystyki['wins'] += 1
        elif zwyciezca == 0:
            statystyki['draws'] += 1
        else:
            statystyki['losses'] += 1
        for i, dane_ruchu in enumerate(historia):
            gracz, czy_koniec = dane_ruchu['player'], (i == len(historia) - 1)
            nagroda = 0.0 if zwyciezca == 0 else 1.0 if zwyciezca == gracz else -1.0
            nastepny_klucz_stanu = historia[i + 1]['state'] if not czy_koniec else dane_ruchu['state']
            if typ_przeciwnika == "self" or (typ_przeciwnika != "self" and gracz == id_gracza_agent1):
                aktualizacje_q.append(
                    (dane_ruchu['state'], dane_ruchu['action'], nagroda if czy_koniec else 0, nastepny_klucz_stanu,
                     czy_koniec))
    return statystyki, aktualizacje_q, id_workera


def ucz_sie_od_minimax(logger, agent: AgentQLearning, liczba_gier: int = 500, uzyj_reguly: bool = False):
    loguj_i_drukuj(logger, "\nUczenie się od eksperta minimax przez grę...")
    loguj_i_drukuj(logger, f"  Gra {liczba_gier} gier przeciwko minimax")
    wygrane = przegrane = remisy = 0
    for numer_gry in range(liczba_gier):
        stan_gry, gracz_agenta = StanGry(), random.choice([1, -1])
        if stan_gry.obecny_gracz != gracz_agenta:
            akcja = znajdz_najlepszy_ruch(stan_gry)
            if akcja: stan_gry.wykonaj_ruch(akcja[0], akcja[1])
        ruchy_agenta = []
        while not stan_gry.czy_koniec_gry():
            if stan_gry.obecny_gracz == gracz_agenta:
                akcja = agent.wybierz_akcje(stan_gry, epsilon=0.8, uzyj_heurystyk=True, uzyj_reguly=uzyj_reguly)
                if akcja:
                    ruchy_agenta.append((agent.pobierz_klucz_stanu(stan_gry.plansza), akcja))
                    stan_gry.wykonaj_ruch(akcja[0], akcja[1])
            else:
                akcja = znajdz_najlepszy_ruch(stan_gry)
                if akcja: stan_gry.wykonaj_ruch(akcja[0], akcja[1])
        zwyciezca = stan_gry.sprawdz_zwyciezce()
        nagroda = 0.0 if zwyciezca == 0 else 1.0 if zwyciezca == gracz_agenta else -1.0
        if zwyciezca == gracz_agenta:
            wygrane += 1
        elif zwyciezca == 0:
            remisy += 1
        else:
            przegrane += 1
        for i, (stan, akcja) in enumerate(ruchy_agenta):
            czy_koniec = (i == len(ruchy_agenta) - 1)
            nastepny_stan = ruchy_agenta[i + 1][0] if not czy_koniec else stan
            agent.aktualizuj(stan, akcja, nagroda if czy_koniec else 0, nastepny_stan, czy_koniec)
        if (numer_gry + 1) % 100 == 0:
            loguj_i_drukuj(logger, f"  Postęp {numer_gry + 1}/{liczba_gier}: W:{wygrane} D:{remisy} L:{przegrane}")
    loguj_i_drukuj(logger, f"  Uczenie nadzorowane zakończone. Rozmiar Q-table: {len(agent.tabela_q)} wpisów")


def trenuj_iteracje_rownolegle(logger, agent: AgentQLearning, iteracja: int, gier_na_iteracje: int,
                               uzyj_reguly: bool = False, przeciwnik_override: str = None,
                               epsilon_override: float = None) -> Dict:
    if przeciwnik_override:
        przeciwnik = przeciwnik_override
        epsilon = epsilon_override
    else:
        if iteracja <= 10:
            epsilon, przeciwnik = 0.3, "smart_random"
        elif iteracja <= 25:
            epsilon, przeciwnik = 0.25, "self"
        else:
            if iteracja % 3 == 0:
                epsilon, przeciwnik = 0.15, "reguly"
            elif iteracja % 3 == 1:
                epsilon, przeciwnik = 0.1, "minimax"
            else:
                epsilon, przeciwnik = 0.2, "self"

    liczba_workerow = cpu_count()
    gier_na_workera = gier_na_iteracje // liczba_workerow
    argumenty_workerow = []
    for i in range(liczba_workerow):
        dodatkowe_gry = 1 if i < gier_na_iteracje % liczba_workerow else 0
        argumenty_workerow.append((dict(agent.tabela_q), przeciwnik, epsilon,
                                   gier_na_workera + dodatkowe_gry, i, True, uzyj_reguly))
    loguj_i_drukuj(logger,
                   f"\nIteracja {iteracja}: Trenowanie vs {przeciwnik} (epsilon={epsilon:.3f}, Gry={gier_na_iteracje})")
    czas_startu = time.time()
    with Pool(processes=liczba_workerow) as pool:
        wyniki = pool.map(graj_partie_batch, argumenty_workerow)
    loguj_i_drukuj(logger, f"  Trenowanie zakończone w {time.time() - czas_startu:.1f}s")

    wszystkie_aktualizacje, calkowite_statystyki = [], {'wins': 0, 'losses': 0, 'draws': 0}
    for statystyki, aktualizacje_q, _ in wyniki:
        for klucz in calkowite_statystyki: calkowite_statystyki[klucz] += statystyki[klucz]
        wszystkie_aktualizacje.extend(aktualizacje_q)

    loguj_i_drukuj(logger, f"  Stosowanie {len(wszystkie_aktualizacje)} aktualizacji Q...")
    for argumenty_aktualizacji in wszystkie_aktualizacje: agent.aktualizuj(*argumenty_aktualizacji)

    calkowita_liczba_gier = sum(calkowite_statystyki.values())
    wspolczynnik_wygranych = calkowite_statystyki['wins'] / calkowita_liczba_gier * 100
    wspolczynnik_przegranych = calkowite_statystyki['losses'] / calkowita_liczba_gier * 100
    wspolczynnik_remisow = calkowite_statystyki['draws'] / calkowita_liczba_gier * 100

    loguj_i_drukuj(logger,
                   f"  Wyniki: W:{calkowite_statystyki['wins']} L:{calkowite_statystyki['losses']} D:{calkowite_statystyki['draws']}")
    loguj_i_drukuj(logger,
                   f"  Współczynniki: Wygrane {wspolczynnik_wygranych:.1f}% | Przegrane {wspolczynnik_przegranych:.1f}% | Remisy {wspolczynnik_remisow:.1f}%")
    loguj_i_drukuj(logger,
                   f"  Rozmiar Q-table: {len(agent.tabela_q)} | Współczynnik uczenia: {agent.wspolczynnik_uczenia:.3f}")

    return calkowite_statystyki


def weryfikacja_miedzyetapowa(logger, agent: AgentQLearning, liczba_gier_na_poziom: int,
                              uzyj_reguly: bool = False) -> Dict:
    loguj_i_drukuj(logger, "\nWeryfikacja miedzyetapowa...")
    poziomy = {
        "random": "Losowy",
        "smart_random": "Mądry Losowy",
        "reguly": "Strategiczny (Reguły)",
        "mcts": "MCTS (2000 iter)",
        "minimax": "Perfekcyjny Minimax"
    }
    wszystkie_wyniki = {}
    for poziom, opis in poziomy.items():
        loguj_i_drukuj(logger, f"  Testowanie vs {opis} ({liczba_gier_na_poziom} gier)...")
        czas_startu_testu = time.time()
        liczba_workerow = min(4, cpu_count()) if poziom not in ["mcts", "minimax"] else 1
        gier_na_workera = liczba_gier_na_poziom // liczba_workerow
        argumenty_workerow = []
        for i in range(liczba_workerow):
            dodatkowe_gry = 1 if i < liczba_gier_na_poziom % liczba_workerow else 0
            argumenty_workerow.append((dict(agent.tabela_q), poziom, 0.0,
                                       gier_na_workera + dodatkowe_gry, i, True, uzyj_reguly))
        with Pool(processes=liczba_workerow) as pool:
            wyniki = pool.map(graj_partie_batch, argumenty_workerow)
        calkowite_statystyki = {'wins': 0, 'losses': 0, 'draws': 0}
        for statystyki, _, _ in wyniki:
            for klucz in calkowite_statystyki: calkowite_statystyki[klucz] += statystyki[klucz]

        wspolczynnik_przegranych = calkowite_statystyki['losses'] / liczba_gier_na_poziom * 100
        status = "ZALICZONO" if wspolczynnik_przegranych == 0 else "NIE ZALICZONO"
        loguj_i_drukuj(logger,
                       f"  {status} vs {opis}: W:{calkowite_statystyki['wins']} L:{calkowite_statystyki['losses']} D:{calkowite_statystyki['draws']} (Przegrane: {wspolczynnik_przegranych:.1f}%)")
        wszystkie_wyniki[poziom] = calkowite_statystyki
    perfekcyjne_vs_latwe = (
                wszystkie_wyniki['random']['losses'] == 0 and wszystkie_wyniki['smart_random']['losses'] == 0)
    perfekcyjne_vs_minimax = wszystkie_wyniki['minimax']['losses'] == 0
    return {'perfect_vs_easy': perfekcyjne_vs_latwe, 'perfect_vs_minimax': perfekcyjne_vs_minimax}


def trenuj_doskonaly_agent(logger, agent: AgentQLearning, uzyj_reguly: bool = False):
    loguj_i_drukuj(logger, "\n--- FAZA 1: SZEROKA EKSPLORACJA (vs Random) ---")
    liczba_iteracji_fazy1 = 20
    for i in range(1, liczba_iteracji_fazy1 + 1):
        trenuj_iteracje_rownolegle(logger, agent, i, 5_000, uzyj_reguly, "random", 0.9)
        loguj_i_drukuj(logger, "─" * 70)

    loguj_i_drukuj(logger, "\n--- FAZA 2: UCZENIE STRATEGII (vs Smart Random, Reguły, Self-Play) ---")
    agent_osiagnal_doskonalosc = False
    liczba_iteracji_fazy2 = 40
    for i in range(1, liczba_iteracji_fazy2 + 1):
        trenuj_iteracje_rownolegle(logger, agent, i, 5_000, uzyj_reguly)
        if i % 10 == 0:
            wyniki_weryfikacji = weryfikacja_miedzyetapowa(logger, agent, 500, uzyj_reguly)
            if wyniki_weryfikacji['perfect_vs_easy'] and wyniki_weryfikacji['perfect_vs_minimax']:
                loguj_i_drukuj(logger, "\n✅ Agent osiągnął doskonałość w trakcie Fazy 2. Przerywanie treningu...")
                agent_osiagnal_doskonalosc = True
                break
        loguj_i_drukuj(logger, "─" * 70)

    if agent_osiagnal_doskonalosc:
        return agent

    loguj_i_drukuj(logger, "\n--- FAZA 3: KOREKTA Z EKSPERTEM (vs Minimax) ---")
    agent.wspolczynnik_uczenia = 0.1
    liczba_iteracji_fazy3 = 10
    for i in range(1, liczba_iteracji_fazy3 + 1):
        trenuj_iteracje_rownolegle(logger, agent, i, 5_000, uzyj_reguly, "minimax", 0.05)
        loguj_i_drukuj(logger, "─" * 70)

    return agent


if __name__ == "__main__":
    czas_startu = time.time()

    folder_bazowy = os.path.join(current_dir, 'q_tables', 'q_learning_table')
    nazwa_folderu, licznik = folder_bazowy, 1
    while os.path.exists(nazwa_folderu):
        nazwa_folderu = f"{folder_bazowy}_{licznik}"
        licznik += 1
    os.makedirs(nazwa_folderu, exist_ok=True)

    logger = konfiguruj_logowanie(nazwa_folderu)
    loguj_i_drukuj(logger, "Szybkie Wielordzeniowe Trenowanie Q-Learning dla Doskonałego Kółko i Krzyżyk")
    loguj_i_drukuj(logger, "=" * 70)
    loguj_i_drukuj(logger, f"Folder trenowania: {nazwa_folderu}")
    loguj_i_drukuj(logger, f"Rozpoczęto: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    uzyj_reguly = False
    agent = AgentQLearning(wspolczynnik_uczenia=0.3, wspolczynnik_dyskontujacy=0.95)

    ucz_sie_od_minimax(logger, agent, liczba_gier=500, uzyj_reguly=uzyj_reguly)

    agent = trenuj_doskonaly_agent(logger, agent, uzyj_reguly=uzyj_reguly)

    loguj_i_drukuj(logger, "\nKońcowa faza weryfikacji (rozszerzona)...")
    loguj_i_drukuj(logger, "🎯 Testowanie przeciwko WSZYSTKIM dostępnym AI z większą liczbą gier")

    wyniki_weryfikacji = weryfikacja_miedzyetapowa(logger, agent, liczba_gier_na_poziom=2000, uzyj_reguly=uzyj_reguly)

    czy_doskonaly = wyniki_weryfikacji['perfect_vs_easy'] and wyniki_weryfikacji['perfect_vs_minimax']

    plik_modelu = os.path.join(nazwa_folderu, 'model.pkl')
    dane = {
        'q_table': dict(agent.tabela_q), 'version': '12.0-systematic-fixed',
        'training_method': '3-phase-q-learning-curriculum', 'is_perfect': czy_doskonaly,
        'q_table_size': len(agent.tabela_q), 'timestamp': datetime.now().isoformat(),
        'rules_available': True,
        'trained_with_rules': uzyj_reguly,
        'trained_against': ['random', 'self', 'reguly', 'minimax', 'smart_random'],
        'final_tests_vs_all_ai': True
    }
    with open(plik_modelu, 'wb') as f:
        pickle.dump(dane, f)
    rozmiar_pliku = os.path.getsize(plik_modelu) / 1024
    loguj_i_drukuj(logger, f"\nModel zapisany do {plik_modelu} ({rozmiar_pliku:.1f} KB)")
    loguj_i_drukuj(logger, "\n" + "=" * 70)

    if czy_doskonaly:
        loguj_i_drukuj(logger, "DOSKONAŁY AGENT POMYŚLNIE WYTRENOWANY! Agent NIGDY nie przegra.")
        loguj_i_drukuj(logger, "🎯 Model używa tylko: prostą heurystykę (win/block) + Q-table")
    else:
        loguj_i_drukuj(logger,
                       "SILNY AGENT WYTRENOWANY, ALE NIE W PEŁNI DOSKONAŁY. Rozważ więcej trenowania lub modyfikację harmonogramu.")

    loguj_i_drukuj(logger, f"\nŁączny czas: {(time.time() - czas_startu) / 60:.1f} minut.")
    loguj_i_drukuj(logger, f"Zakończono: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")