import math
import random
from typing import Tuple, Optional, List
import numpy as np
from gra.logika import StanGry


class wezelMCTS:
    def __init__(self, stan_gry: StanGry, rodzic: Optional['wezelMCTS'] = None,
                 ruch_do_wezla: Optional[Tuple[int, int]] = None):
        self.stan_gry = stan_gry
        self.rodzic = rodzic
        self.ruch_do_wezla = ruch_do_wezla
        self.dzieci: List['wezelMCTS'] = []
        self.odwiedziny = 0
        self.punkty = 0.0
        self.dostepne_ruchy = stan_gry.otrzymaj_mozliwe_ruchy().copy()
        if rodzic is None:
            self.gracz = None
        else:
            self.gracz = rodzic.stan_gry.obecny_gracz
        
    def czy_rozwiniety(self) -> bool:
        return len(self.dostepne_ruchy) == 0
    
    def czy_koncowy(self) -> bool:
        return self.stan_gry.czy_koniec_gry()
    
    def dodaj_dziecko(self, ruch: Tuple[int, int], stan_gry: StanGry) -> 'wezelMCTS':
        dziecko = wezelMCTS(stan_gry, rodzic=self, ruch_do_wezla=ruch)
        self.dzieci.append(dziecko)
        self.dostepne_ruchy.remove(ruch)
        return dziecko
    
    def wartosc_ucb(self, stala_eksploracji: float = math.sqrt(2)) -> float:
        if self.odwiedziny == 0:
            return float('inf')
        
        exploitation = self.punkty / self.odwiedziny
        exploration = stala_eksploracji * math.sqrt(math.log(self.rodzic.odwiedziny) / self.odwiedziny)
        return exploitation + exploration
    
    def wybierz_najlepsze_dziecko(self, stala_eksploracji: float = math.sqrt(2)) -> 'wezelMCTS':
        return max(self.dzieci, key=lambda c: c.wartosc_ucb(stala_eksploracji))
    
    def aktualizuj(self, wynik: float) -> None:
        self.odwiedziny += 1
        self.punkty += wynik


class AgentMCTS:
    def __init__(self, iteracje: int = 1000, stala_eksploracji: float = math.sqrt(2)):
        self.iteracje = iteracje
        self.stala_eksploracji = stala_eksploracji

    def znajdz_ruch(self, stan_gry: StanGry) -> Optional[Tuple[int, int]]:
        if stan_gry.czy_koniec_gry():
            return None

        mozliwe_ruchy = stan_gry.otrzymaj_mozliwe_ruchy()
        if not mozliwe_ruchy:
            return None

        if len(mozliwe_ruchy) == 1:
            return mozliwe_ruchy[0]

        korzen = wezelMCTS(stan_gry)
        gracz_poczatkowy = stan_gry.obecny_gracz

        for _ in range(self.iteracje):
            wezel = self._wybierz_i_rozwijaj(korzen)
            wynik = self._symuluj(wezel.stan_gry, gracz_poczatkowy)
            self._proguj_wstecz(wezel, wynik, gracz_poczatkowy)

        if not korzen.dzieci:
            return random.choice(mozliwe_ruchy)

        najlepsze_dziecko = max(korzen.dzieci, key=lambda c: c.odwiedziny)
        return najlepsze_dziecko.ruch_do_wezla

    def _wybierz_i_rozwijaj(self, korzen: wezelMCTS) -> wezelMCTS:
        wezel = korzen
        while not wezel.czy_koncowy() and wezel.czy_rozwiniety():
            wezel = wezel.wybierz_najlepsze_dziecko(self.stala_eksploracji)

        if not wezel.czy_koncowy() and not wezel.czy_rozwiniety():
            ruch = random.choice(wezel.dostepne_ruchy)
            nowy_stan = wezel.stan_gry.sklonuj()
            nowy_stan.wykonaj_ruch(ruch[0], ruch[1])
            wezel = wezel.dodaj_dziecko(ruch, nowy_stan)

        return wezel

    def _symuluj(self, stan_gry: StanGry, oryginalny_gracz: int) -> float:
        symulacja = stan_gry.sklonuj()

        while not symulacja.czy_koniec_gry():
            ruchy = symulacja.otrzymaj_mozliwe_ruchy()
            if not ruchy:
                break
            ruch = self._wybierz_ruch_symulacji(symulacja, ruchy)
            symulacja.wykonaj_ruch(ruch[0], ruch[1])

        zwyciezca = symulacja.sprawdz_zwyciezce()
        if zwyciezca == oryginalny_gracz:
            return 1.0
        elif zwyciezca == -oryginalny_gracz:
            return 0.0
        else:
            return 0.5

    def _wybierz_ruch_symulacji(self, stan_gry: StanGry, mozliwe_ruchy: List[Tuple[int, int]]) -> Tuple[int, int]:
        for ruch in mozliwe_ruchy:
            test = StanGry(stan_gry.rozmiar_planszy, stan_gry.warunek_wygranej)
            test.plansza = stan_gry.plansza.copy()
            test.obecny_gracz = stan_gry.obecny_gracz
            test.wykonaj_ruch(ruch[0], ruch[1])
            if test.sprawdz_zwyciezce() == stan_gry.obecny_gracz:
                return ruch
        
        przeciwnik = -stan_gry.obecny_gracz
        for ruch in mozliwe_ruchy:
            test = StanGry(stan_gry.rozmiar_planszy, stan_gry.warunek_wygranej)
            test.plansza = stan_gry.plansza.copy()
            test.obecny_gracz = przeciwnik
            test.wykonaj_ruch(ruch[0], ruch[1])
            if test.sprawdz_zwyciezce() == przeciwnik:
                return ruch
        
        ruchy_wazone = []
        for ruch in mozliwe_ruchy:
            if ruch == (1, 1):
                ruchy_wazone += [ruch] * 3
            elif ruch in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                ruchy_wazone += [ruch] * 2
            else:
                ruchy_wazone.append(ruch)
        return random.choice(ruchy_wazone)

    def _proguj_wstecz(self, wezel: wezelMCTS, wynik: float, oryginalny_gracz: int) -> None:
        while wezel is not None:
            if wezel.gracz == oryginalny_gracz:
                wezel.aktualizuj(wynik)
            else:
                wezel.aktualizuj(1.0 - wynik)
            wezel = wezel.rodzic


def znajdz_najlepszy_ruch(stan_gry: StanGry, iteracje: int = 1000) -> Optional[Tuple[int, int]]:
    wolne_pola = len(stan_gry.otrzymaj_mozliwe_ruchy())
    
    if wolne_pola > 7:
        iteracje = min(iteracje, 500)
    elif wolne_pola > 4:
        iteracje = min(iteracje, 800)
    else:
        iteracje = min(iteracje, 1500)
    
    agent = AgentMCTS(iteracje=iteracje)
    return agent.znajdz_ruch(stan_gry)


if __name__ == "__main__":
    gra = StanGry(3, 3)
    agent = AgentMCTS(iteracje=1000)
    
    print("Testing MCTS agent on Tic-Tac-Toe...")
    print("Initial board:")
    print(gra)
    
    ruch = agent.znajdz_ruch(gra)
    if ruch:
        print(f"MCTS suggests move: {ruch}")
        gra.wykonaj_ruch(ruch[0], ruch[1])
        print("Board after move:")
        print(gra)