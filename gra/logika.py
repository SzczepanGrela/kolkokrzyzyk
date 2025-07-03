import numpy as np
import copy
from typing import List, Tuple, Optional, Union


class StanGry:
    def __init__(self, rozmiar_planszy: int = 3, warunek_wygranej: int = 3) -> None:
        self.rozmiar_planszy = rozmiar_planszy
        self.warunek_wygranej = warunek_wygranej
        self.plansza = np.zeros((rozmiar_planszy, rozmiar_planszy), dtype=int)
        self.obecny_gracz = 1
    
    def wykonaj_ruch(self, rzad: int, kolumna: int) -> bool:
        if not (0 <= rzad < self.rozmiar_planszy and 0 <= kolumna < self.rozmiar_planszy):
            return False
        
        if self.plansza[rzad, kolumna] != 0:
            return False
        
        self.plansza[rzad, kolumna] = self.obecny_gracz
        self.obecny_gracz *= -1
        
        return True
    
    def otrzymaj_mozliwe_ruchy(self) -> List[Tuple[int, int]]:
        mozliwe_ruchy = []
        for row in range(self.rozmiar_planszy):
            for col in range(self.rozmiar_planszy):
                if self.plansza[row, col] == 0:
                    mozliwe_ruchy.append((row, col))
        return mozliwe_ruchy
    
    def sprawdz_zwyciezce(self) -> Optional[int]:
        zwyciezca = self._sprawdz_warunek_wygranej()
        if zwyciezca is not None:
            return zwyciezca
        
        if len(self.otrzymaj_mozliwe_ruchy()) == 0:
            return 0
        
        return None
    
    def _sprawdz_warunek_wygranej(self) -> Optional[int]:
        kierunki = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for rzad in range(self.rozmiar_planszy):
            for kolumna in range(self.rozmiar_planszy):
                if self.plansza[rzad, kolumna] != 0:
                    gracz = self.plansza[rzad, kolumna]

                    for kierunek_poziomy, kierunek_pionowy in kierunki:
                        if self._sprawdz_linie(rzad, kolumna, kierunek_poziomy, kierunek_pionowy, gracz):
                            return gracz

        return None
    
    def _sprawdz_linie(self, poczatkowy_rzad: int, poczatkowa_kolumna: int,
                       delta_rzad: int, delta_kolumna: int, gracz: int) -> bool:
        licznik = 0
        rzad, kolumna = poczatkowy_rzad, poczatkowa_kolumna
        
        while (0 <= rzad < self.rozmiar_planszy and
               0 <= kolumna < self.rozmiar_planszy and
               self.plansza[rzad, kolumna] == gracz):
            licznik += 1
            if licznik >= self.warunek_wygranej:
                return True
            rzad += delta_rzad
            kolumna += delta_kolumna
        
        return False
    
    def czy_koniec_gry(self) -> bool:
        return self.sprawdz_zwyciezce() is not None
    
    def zresetuj_plansze(self) -> None:
        self.plansza = np.zeros((self.rozmiar_planszy, self.rozmiar_planszy), dtype=int)
        self.obecny_gracz = 1

    def sklonuj(self) -> 'StanGry':
        return copy.deepcopy(self)


    def otrzymaj_kopie_planszy(self) -> np.ndarray:
        return self.plansza.copy()
    
    def __str__(self) -> str:
        symbole = {0: '.', 1: 'X', -1: 'O'}
        linie = []
        for rzad in self.plansza:
            line = ' '.join(symbole[komorka] for komorka in rzad)
            linie.append(line)
        return '\n'.join(linie)