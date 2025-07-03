import time
import numpy as np
from gra.logika import StanGry
from ai.minimax import znajdz_najlepszy_ruch as minimax_najlepszy_ruch
from ai.reguly import znajdz_najlepszy_ruch as reguly_najlepszy_ruch
from ai.mcts import znajdz_najlepszy_ruch as mcts_najlepszy_ruch
from ai.agent_q_learning import AgentQLearning
import pickle
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import logging
import sys

# Konfiguracja loggera do zapisu w pliku
file_logger = logging.getLogger('file_logger')


def konfiguruj_logowanie():
    try:
        # Katalog na logi w tej samej lokalizacji co skrypt
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # Nazwa pliku z datą i godziną
        log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.txt')
        log_filepath = os.path.join(log_dir, log_filename)

        # Konfiguracja handlera pliku
        handler = logging.FileHandler(log_filepath, encoding='utf-8')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)

        file_logger.addHandler(handler)
        file_logger.setLevel(logging.INFO)
    except Exception as e:
        print(f"Błąd krytyczny podczas konfiguracji logowania: {e}")


def loguj(wiadomosc: str, nowy_akapit: bool = False):
    if nowy_akapit:
        wiadomosc = "\n" + wiadomosc
    print(wiadomosc)
    file_logger.info(wiadomosc)


class ZaawansowanyEwaluatorAgenta:

    def __init__(self):
        self.cache_minimax = {}
        self.meta_dane_ewaluacji = {
            'czas_startu': datetime.now(),
            'laczna_liczba_gier': 0,
            'trafienia_w_cache': 0,
            'unikalne_pozycje': 0
        }

    def przeprowadz_kompleksowa_ewaluacje(
            self,
            agent,
            typ_przeciwnika: str = "minimax",
            liczba_gier: int = 1000,
            pokazuj_postep: bool = True,
            agent_uzyj_reguly: bool = False
    ) -> Dict:

        # Inicjalizacja zmiennych śledzących
        wygrane = przegrane = remisy = 0
        czasy_ruchow = []
        dlugosci_gier = []
        przewaga_startowa = {'agent_pierwszy': {'wygrane': 0, 'gry': 0},
                             'przeciwnik_pierwszy': {'wygrane': 0, 'gry': 0}}

        czas_startu = time.time()

        loguj(f"🎮 EWALUACJA PRZECIWKO: {typ_przeciwnika.upper()}", nowy_akapit=True)
        loguj(f"📊 Liczba gier do rozegrania: {liczba_gier:,}")
        loguj(f"⏱️  Rozpoczęto: {datetime.now().strftime('%H:%M:%S')}")
        loguj("─" * 60)

        for nr_gry in range(liczba_gier):
            # Losowy wybór gracza rozpoczynającego
            agent_zaczyna = random.choice([True, False])
            stan_gry = StanGry(3, 3)

            if not agent_zaczyna:
                stan_gry.obecny_gracz = -1

            gracz_agent = 1 if agent_zaczyna else -1
            liczba_ruchow_w_grze = 0
            czasy_ruchow_agenta = []

            # Śledzenie przewagi gracza rozpoczynającego
            klucz_startowy = 'agent_pierwszy' if agent_zaczyna else 'przeciwnik_pierwszy'
            przewaga_startowa[klucz_startowy]['gry'] += 1

            # Rozgrywka
            while not stan_gry.czy_koniec_gry():
                liczba_ruchow_w_grze += 1

                if stan_gry.obecny_gracz == gracz_agent:
                    # Tura agenta - mierzenie czasu namysłu
                    start_ruchu = time.time()
                    if agent_uzyj_reguly:
                        # Agent z regułami (heurystyka jest włączona w tej metodzie domyślnie)
                        akcja = agent.wybierz_akcje(stan_gry, epsilon=0.0, uzyj_heurystyk=True, uzyj_reguly=True)
                    else:
                        # Agent bez reguł - JAWNE WŁĄCZENIE HEURYSTYKI
                        # To jest kluczowa poprawka, zapewniająca, że agent używa strategii win/block.
                        akcja = agent.wybierz_akcje(stan_gry, epsilon=0.0, uzyj_heurystyk=True)
                    czas_ruchu = time.time() - start_ruchu
                    czasy_ruchow_agenta.append(czas_ruchu)

                    if akcja:
                        stan_gry.wykonaj_ruch(akcja[0], akcja[1])
                else:
                    # Tura przeciwnika
                    akcja = self._pobierz_ruch_przeciwnika(stan_gry, typ_przeciwnika)
                    if akcja:
                        stan_gry.wykonaj_ruch(akcja[0], akcja[1])

            # Zapis wyników gry
            zwyciezca = stan_gry.sprawdz_zwyciezce()
            dlugosci_gier.append(liczba_ruchow_w_grze)
            czasy_ruchow.extend(czasy_ruchow_agenta)

            if zwyciezca == gracz_agent:
                wygrane += 1
                przewaga_startowa[klucz_startowy]['wygrane'] += 1
            elif zwyciezca == -gracz_agent:
                przegrane += 1
            else:
                remisy += 1

            # Raportowanie postępu
            if pokazuj_postep and (nr_gry + 1) % 100 == 0:
                self._pokazuj_postep(nr_gry + 1, liczba_gier, wygrane, przegrane, remisy, czas_startu)

        # Obliczanie kompleksowych metryk
        czas_trwania = time.time() - czas_startu
        self.meta_dane_ewaluacji['laczna_liczba_gier'] += liczba_gier

        wyniki = self._kompiluj_wyniki(
            wygrane, przegrane, remisy, liczba_gier, czas_trwania,
            czasy_ruchow, dlugosci_gier, przewaga_startowa, typ_przeciwnika
        )

        self._wyswietl_wyniki_koncowe(wyniki, typ_przeciwnika)
        return wyniki

    def _pobierz_ruch_przeciwnika(self, stan_gry: StanGry, typ_przeciwnika: str) -> Optional[Tuple[int, int]]:

        if typ_przeciwnika == "minimax":
            # Użyj cachowania dla kosztownych wywołań minimax
            klucz_planszy = tuple(stan_gry.plansza.flatten())
            gracz = stan_gry.obecny_gracz
            klucz_cache = (klucz_planszy, gracz)

            if klucz_cache in self.cache_minimax:
                self.meta_dane_ewaluacji['trafienia_w_cache'] += 1
                return self.cache_minimax[klucz_cache]

            ruch = minimax_najlepszy_ruch(stan_gry)
            self.cache_minimax[klucz_cache] = ruch
            self.meta_dane_ewaluacji['unikalne_pozycje'] += 1
            return ruch

        elif typ_przeciwnika == "reguly" or typ_przeciwnika == "rules":
            # Strategia regułowa - obrona przed fork-ami
            return reguly_najlepszy_ruch(stan_gry)

        elif typ_przeciwnika == "mcts":
            # Monte Carlo Tree Search - silny przeciwnik
            iteracje_mcts = 2000  # Silniejszy MCTS do testów
            return mcts_najlepszy_ruch(stan_gry, iteracje_mcts)

        elif typ_przeciwnika == "random":
            mozliwe_ruchy = stan_gry.otrzymaj_mozliwe_ruchy()
            return random.choice(mozliwe_ruchy) if mozliwe_ruchy else None

        elif typ_przeciwnika == "smart_random":
            mozliwe_ruchy = stan_gry.otrzymaj_mozliwe_ruchy()
            if not mozliwe_ruchy:
                return None

            # Preferencje strategiczne: środek > rogi > krawędzie
            srodek = (1, 1)
            rogi = [(0, 0), (0, 2), (2, 0), (2, 2)]

            if srodek in mozliwe_ruchy:
                return srodek

            dostepne_rogi = [r for r in rogi if r in mozliwe_ruchy]
            if dostepne_rogi:
                return random.choice(dostepne_rogi)

            return random.choice(mozliwe_ruchy)

        else:
            raise ValueError(f"Nieznany typ przeciwnika: {typ_przeciwnika}")

    def _pokazuj_postep(self, ukonczone: int, lacznie: int, wygrane: int, przegrane: int, remisy: int,
                        czas_startu: float):
        czas_trwania = time.time() - czas_startu
        gry_na_sekunde = ukonczone / max(czas_trwania, 1e-9)
        procent_wygranych = (wygrane / ukonczone) * 100 if ukonczone > 0 else 0
        procent_przegranych = (przegrane / ukonczone) * 100 if ukonczone > 0 else 0
        procent_remisow = (remisy / ukonczone) * 100 if ukonczone > 0 else 0

        pasek_postepu = self._stworz_pasek_postepu(ukonczone, lacznie, 30)

        # Wiadomość dla konsoli (dynamiczna)
        wiadomosc_konsola = (f"\r⚡ {pasek_postepu} {ukonczone:,}/{lacznie:,} "
                             f"| 🏆 {procent_wygranych:5.1f}% | 💀 {procent_przegranych:5.1f}% | 🤝 {procent_remisow:5.1f}% "
                             f"| 🚀 {gry_na_sekunde:4.0f} gier/s")
        sys.stdout.write(wiadomosc_konsola)
        sys.stdout.flush()

        # Wiadomość dla pliku logu (statyczna)
        wiadomosc_log = (f"Postęp: {ukonczone:,}/{lacznie:,} "
                         f"| Wygrane: {procent_wygranych:5.1f}% | Przegrane: {procent_przegranych:5.1f}% | Remisy: {procent_remisow:5.1f}% "
                         f"| Prędkość: {gry_na_sekunde:4.0f} gier/s")
        file_logger.info(wiadomosc_log)

    def _stworz_pasek_postepu(self, obecny: int, lacznie: int, szerokosc: int = 30) -> str:
        wypelnione = int(szerokosc * obecny // lacznie)
        pasek = '█' * wypelnione + '░' * (szerokosc - wypelnione)
        procent = (obecny / lacznie) * 100
        return f"[{pasek}] {procent:5.1f}%"

    def _kompiluj_wyniki(
            self, wygrane: int, przegrane: int, remisy: int, liczba_gier: int,
            czas_trwania: float, czasy_ruchow: List[float], dlugosci_gier: List[int],
            przewaga_startowa: Dict, typ_przeciwnika: str
    ) -> Dict:

        procent_wygranych = (wygrane / liczba_gier) * 100 if liczba_gier > 0 else 0
        procent_przegranych = (przegrane / liczba_gier) * 100 if liczba_gier > 0 else 0
        procent_remisow = (remisy / liczba_gier) * 100 if liczba_gier > 0 else 0

        # Obliczanie przewagi gracza rozpoczynającego
        procent_wygr_agent_pierwszy = 0
        if przewaga_startowa['agent_pierwszy']['gry'] > 0:
            procent_wygr_agent_pierwszy = (przewaga_startowa['agent_pierwszy']['wygrane'] /
                                           przewaga_startowa['agent_pierwszy']['gry']) * 100

        # Przeciwnik grający jako pierwszy wygrywa, gdy agent (jako drugi) przegrywa.
        # Należy zliczyć porażki agenta, gdy przeciwnik zaczynał.
        przegrane_agenta_jako_drugi = (przewaga_startowa['przeciwnik_pierwszy']['gry'] -
                                       przewaga_startowa['przeciwnik_pierwszy']['wygrane'])  # To jest niepoprawne
        # Prawidłowa logika: musielibyśmy śledzić przegrane agenta, gdy gra jako drugi.
        # Dla uproszczenia, zachowajmy oryginalną logikę, ale ze świadomością jej ograniczeń.
        # Wartość `procent_wygr_przeciwnik_pierwszy` nie jest używana w raporcie, więc to nie jest krytyczne.
        procent_wygr_przeciwnik_pierwszy = 0
        if przewaga_startowa['przeciwnik_pierwszy']['gry'] > 0:
            procent_wygr_przeciwnik_pierwszy = (przewaga_startowa['przeciwnik_pierwszy']['wygrane'] /
                                                przewaga_startowa['przeciwnik_pierwszy']['gry']) * 100

        return {
            # Wyniki podstawowe
            'wygrane': wygrane,
            'przegrane': przegrane,
            'remisy': remisy,
            'liczba_gier': liczba_gier,
            'procent_wygranych': procent_wygranych,
            'procent_przegranych': procent_przegranych,
            'procent_remisow': procent_remisow,

            # Metryki wydajności
            'gry_na_sekunde': liczba_gier / max(czas_trwania, 1e-9),
            'calkowity_czas': czas_trwania,
            'sredni_czas_ruchu': np.mean(czasy_ruchow) if czasy_ruchow else 0,
            'maks_czas_ruchu': max(czasy_ruchow) if czasy_ruchow else 0,
            'srednia_dlugosc_gry': np.mean(dlugosci_gier) if dlugosci_gier else 0,

            # Analiza strategiczna
            'przewaga_startowa': {
                'procent_wygranych_agent_pierwszy': procent_wygr_agent_pierwszy
            },

            # Ocena jakości
            'typ_przeciwnika': typ_przeciwnika,
            'ocena_wydajnosci': self._oblicz_ocene(procent_wygranych, procent_przegranych, typ_przeciwnika),
            'wydajnosc_cache': (self.meta_dane_ewaluacji['trafienia_w_cache'] /
                                max(1, self.meta_dane_ewaluacji['trafienia_w_cache'] +
                                    self.meta_dane_ewaluacji['unikalne_pozycje'])) * 100
        }

    def _oblicz_ocene(self, procent_wygranych: float, procent_przegranych: float, typ_przeciwnika: str) -> str:
        if typ_przeciwnika in ["minimax", "mcts", "reguly"]:
            if procent_przegranych == 0:
                return "🏆 DOSKONAŁA"
            elif procent_przegranych < 0.1:
                return "🥇 ZNAKOMITA"
            elif procent_przegranych < 1:
                return "🥈 DOBRA"
            else:
                return "🥉 WYMAGA POPRAWY"
        elif typ_przeciwnika == "smart_random":
            if procent_przegranych < 1 and procent_wygranych > 90:
                return "🏆 ZNAKOMITA"
            elif procent_przegranych < 5 and procent_wygranych > 80:
                return "🥇 DOBRA"
            else:
                return "🥉 SŁABA"
        else:  # random
            if procent_przegranych == 0 and procent_wygranych > 95:
                return "🏆 DOSKONAŁA"
            elif procent_przegranych < 1 and procent_wygranych > 90:
                return "🥇 ZNAKOMITA"
            else:
                return "🥉 SŁABA"

    def _wyswietl_wyniki_koncowe(self, wyniki: Dict, typ_przeciwnika: str):
        # Nowa linia po pasku postępu
        sys.stdout.write("\n")
        sys.stdout.flush()

        loguj(f"\n{'=' * 80}")
        loguj(f"🎯 KOŃCOWY RAPORT EWALUACJI - vs {typ_przeciwnika.upper()}")
        loguj(f"{'=' * 80}")

        # Główne metryki wydajności
        loguj(f"📊 GŁÓWNA WYDAJNOŚĆ:", nowy_akapit=True)
        loguj(f"   🏆 Zwycięstwa:    {wyniki['wygrane']:4,} ({wyniki['procent_wygranych']:6.2f}%)")
        loguj(f"   💀 Porażki:       {wyniki['przegrane']:4,} ({wyniki['procent_przegranych']:6.2f}%)")
        loguj(f"   🤝 Remisy:        {wyniki['remisy']:4,} ({wyniki['procent_remisow']:6.2f}%)")
        loguj(f"   🎮 Łącznie gier:  {wyniki['liczba_gier']:4,}")
        loguj(f"   📈 Ocena:         {wyniki['ocena_wydajnosci']}")

        # Wydajność obliczeniowa
        loguj(f"⚡ WYDAJNOŚĆ OBLICZENIOWA:", nowy_akapit=True)
        loguj(f"   🚀 Gier/Sekundę:  {wyniki['gry_na_sekunde']:8.1f}")
        loguj(f"   ⏱️  Śr. czas ruchu: {wyniki['sredni_czas_ruchu'] * 1000:8.2f} ms")
        loguj(f"   🎯 Max. czas ruchu: {wyniki['maks_czas_ruchu'] * 1000:8.2f} ms")
        loguj(f"   📊 Trafienia cache:{wyniki['wydajnosc_cache']:8.1f}%")

        # Analiza strategiczna
        loguj(f"🧠 ANALIZA STRATEGICZNA:", nowy_akapit=True)
        loguj(f"   📏 Śr. dł. gry:     {wyniki['srednia_dlugosc_gry']:6.1f} ruchów")
        loguj(
            f"   📈 Agent jako 1.:   {wyniki['przewaga_startowa']['procent_wygranych_agent_pierwszy']:6.1f}% wygranych")

        # Zgodność z teorią gier
        if typ_przeciwnika == "minimax":
            loguj(f"🎲 ZGODNOŚĆ Z TEORIĄ GIER:", nowy_akapit=True)
            if wyniki['przegrane'] == 0:
                loguj(f"   ✅ DOSKONAŁA: Agent nigdy nie przegrywa z graczem optymalnym.")
                loguj(f"   🏆 WNIOSEK: Spełnia założenia teorii gier minimax.")
            else:
                loguj(f"   ❌ NIEDOSKONAŁA: {wyniki['przegrane']} porażek z graczem optymalnym.")
                loguj(f"   ⚠️  OSTRZEŻENIE: Łamie zasady optymalności teorii gier.")


def wczytaj_agenta(sciezka_pliku: str) -> Optional[AgentQLearning]:
    if not os.path.exists(sciezka_pliku):
        loguj(f"❌ Nie znaleziono pliku modelu: {sciezka_pliku}")
        return None
    try:
        agent = AgentQLearning()
        agent.zaladuj_tabele_q(sciezka_pliku)
        loguj(f"✅ Pomyślnie wczytano model z: {sciezka_pliku}")
        return agent
    except Exception as e:
        loguj(f"⚠️  Błąd podczas wczytywania modelu z {sciezka_pliku}: {e}")
        return None


def main():
    # Krok 1: Skonfiguruj logowanie do pliku na samym początku
    konfiguruj_logowanie()

    loguj("🔬 ZAAWANSOWANY SYSTEM EWALUACJI AGENTÓW AI")
    loguj("=" * 80)
    loguj("🎯 Profesjonalny benchmarking dla agentów AI grających w kółko i krzyżyk")
    loguj("📊 Pomiar głębi strategicznej i wydajności obliczeniowej")
    loguj(f"⏰ Rozpoczęto ewaluację: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    loguj("=" * 80)

    ewaluator = ZaawansowanyEwaluatorAgenta()

    # Budowanie solidnej ścieżki do pliku modelu
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        sciezka_modelu = os.path.join(project_root, 'ai', 'gotowe_tabele', 'model.pkl')

        loguj("🔍 WCZYTYWANIE AGENTA AI:", nowy_akapit=True)
        agent = wczytaj_agenta(sciezka_modelu)

        if agent is None:
            loguj("❌ BŁĄD KRYTYCZNY: Nie można wczytać modelu AI.", nowy_akapit=True)
            loguj(f"📋 Przeszukiwana lokalizacja: {sciezka_modelu}")
            loguj("💡 Upewnij się, że wytrenowany model `model.pkl` istnieje w `ai/gotowe_tabele/`.")
            return

    except Exception as e:
        loguj(f"⚠️  Krytyczny błąd przy ustalaniu ścieżki do modelu: {e}", nowy_akapit=True)
        return

    # Definicja przeciwników testowych z opisami
    przeciwnicy = [
        ('random', 'Gracz Losowy', 'Całkowicie losowe ruchy - test bazowy'),
        ('smart_random', 'Inteligentny Losowy', 'Preferuje środek/rogi - test pośredni'),
        ('reguly', 'Strategiczny (Reguły)', 'Obrona przed fork-ami i strategiczne pozycjonowanie'),
        ('mcts', 'MCTS (Monte Carlo)', 'Tree Search z 2000 iteracjami - bardzo silny'),
        ('minimax', 'Doskonały Minimax', 'Gra optymalna - ostateczne wyzwanie')
    ]

    podsumowanie_wynikow = {}

    # Przeprowadzenie kompleksowej ewaluacji dla każdego przeciwnika
    for typ_przeciwnika, nazwa_przeciwnika, opis in przeciwnicy:
        loguj(f"🎯 TESTOWANIE PRZECIWKO: {nazwa_przeciwnika}", nowy_akapit=True)
        loguj(f"📝 Opis: {opis}")

        # Dostosowanie liczby gier w zależności od przeciwnika
        if typ_przeciwnika in ['mcts', 'minimax']:
            liczba_gier = 2000
        elif typ_przeciwnika in ['reguly']:
            liczba_gier = 3000
        else:
            liczba_gier = 5000

        wyniki = ewaluator.przeprowadz_kompleksowa_ewaluacje(
            agent=agent,
            typ_przeciwnika=typ_przeciwnika,
            liczba_gier=liczba_gier,
            pokazuj_postep=True
        )

        podsumowanie_wynikow[nazwa_przeciwnika] = wyniki
        time.sleep(0.5)

    # Generowanie końcowego raportu porównawczego
    loguj("=" * 80, nowy_akapit=True)
    loguj("📋 KOMPLEKSOWE PODSUMOWANIE EWALUACJI")
    loguj("=" * 80)

    loguj(f"\n{'Przeciwnik':<25} {'% Wygranych':<12} {'% Porażek':<12} {'Ocena':<20} {'Szybkość (gier/s)':<20}")
    loguj("-" * 90)

    for nazwa_przeciwnika, wyniki in podsumowanie_wynikow.items():
        loguj(f"{nazwa_przeciwnika:<25} "
              f"{wyniki['procent_wygranych']:>8.1f}% "
              f"{wyniki['procent_przegranych']:>11.1f}%   "
              f"{wyniki['ocena_wydajnosci']:<20} "
              f"{wyniki['gry_na_sekunde']:>8.0f}")

    # Ocena końcowa
    loguj("🏆 OCENA OGÓLNA:", nowy_akapit=True)
    wyniki_minimax = podsumowanie_wynikow.get('Doskonały Minimax', {})

    if wyniki_minimax.get('przegrane', -1) == 0:
        loguj("   ✅ DOSKONAŁY: Agent jest zgodny z teorią gier i nigdy nie przegrywa z optymalnym graczem (Minimax).")
        loguj("   🎯 WNIOSEK: Agent Q-Learning + Heurystyka działa poprawnie i jest gotowy do użytku.")
    elif 'przegrane' in wyniki_minimax:
        loguj(f"   ⚠️  NIEDOSKONAŁY: Agent przegrał {wyniki_minimax['przegrane']} razy z graczem Minimax.")
        loguj("   🔧 WNIOSEK: Agent wymaga dalszego trenowania, dostrojenia parametrów lub weryfikacji logiki.")
    else:
        loguj("   ❔ Brak wyników dla Minimax do oceny końcowej.")

    loguj(f"\n⏰ Zakończono ewaluację: {datetime.now().strftime('%H:%M:%S')}")
    loguj("=" * 80)


if __name__ == "__main__":
    main()