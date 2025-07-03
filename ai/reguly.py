from typing import Tuple, Optional, List
import numpy as np
from gra.logika import StanGry


def znajdz_najlepszy_ruch(stan_gry: StanGry) -> Optional[Tuple[int, int]]:
    if stan_gry.rozmiar_planszy != 3:
        return _prosta_strategia(stan_gry)
    
    graczSI = stan_gry.obecny_gracz
    ludzki_gracz = -graczSI
    
    wygrywajacy_ruch = _znajdz_wygrywajacy_ruch(stan_gry, graczSI)
    if wygrywajacy_ruch:
        return wygrywajacy_ruch
    
    blokujacy_ruch = _znajdz_wygrywajacy_ruch(stan_gry, ludzki_gracz)
    if blokujacy_ruch:
        return blokujacy_ruch
    
    podwojne_zagrozenie = _znajdz_podwojne_zagrozenie(stan_gry, graczSI)
    if podwojne_zagrozenie:
        return podwojne_zagrozenie
    
    blok_podwojnego_zagrozenia = _znajdz_blok_podwojnego_zagrozenia(stan_gry, graczSI, ludzki_gracz)
    if blok_podwojnego_zagrozenia:
        return blok_podwojnego_zagrozenia
    
    srodek = (1, 1)
    if stan_gry.plansza[srodek[0], srodek[1]] == 0:
        return srodek
    
    przeciwny_rog = _znajdz_przeciwny_rog(stan_gry, ludzki_gracz)
    if przeciwny_rog:
        return przeciwny_rog
    
    pusty_rog = _znajdz_pusty_rog(stan_gry)
    if pusty_rog:
        return pusty_rog
    
    pusta_krawedz = _znajdz_pusta_krawedz(stan_gry)
    if pusta_krawedz:
        return pusta_krawedz
    
    return None


def _znajdz_wygrywajacy_ruch(stan_gry: StanGry, gracz: int) -> Optional[Tuple[int, int]]:
    for rzad, kolumna in stan_gry.otrzymaj_mozliwe_ruchy():
        kopia_stanu = StanGry(stan_gry.rozmiar_planszy, stan_gry.warunek_wygranej)
        kopia_stanu.plansza = stan_gry.plansza.copy()
        kopia_stanu.obecny_gracz = gracz
        
        kopia_stanu.wykonaj_ruch(rzad, kolumna)
        
        if kopia_stanu.sprawdz_zwyciezce() == gracz:
            return (rzad, kolumna)
    
    return None


def _znajdz_podwojne_zagrozenie(stan_gry: StanGry, gracz: int) -> Optional[Tuple[int, int]]:
    for rzad, kolumna in stan_gry.otrzymaj_mozliwe_ruchy():
        kopia_stanu = StanGry(stan_gry.rozmiar_planszy, stan_gry.warunek_wygranej)
        kopia_stanu.plansza = stan_gry.plansza.copy()
        kopia_stanu.obecny_gracz = gracz
        
        kopia_stanu.wykonaj_ruch(rzad, kolumna)
        
        liczba_zagrozen = 0
        for nastepny_rzad, nastepna_kolumna in kopia_stanu.otrzymaj_mozliwe_ruchy():
            kopia_stanu2 = StanGry(kopia_stanu.rozmiar_planszy, kopia_stanu.warunek_wygranej)
            kopia_stanu2.plansza = kopia_stanu.plansza.copy()
            kopia_stanu2.obecny_gracz = gracz
            
            kopia_stanu2.wykonaj_ruch(nastepny_rzad, nastepna_kolumna)
            
            if kopia_stanu2.sprawdz_zwyciezce() == gracz:
                liczba_zagrozen += 1
                if liczba_zagrozen >= 2:
                    return (rzad, kolumna)
    
    return None


def _znajdz_blok_podwojnego_zagrozenia(stan_gry: StanGry, graczSI: int, ludzki_gracz: int) -> Optional[Tuple[int, int]]:
    potencjalny_fork_przeciwnika = _znajdz_podwojne_zagrozenie(stan_gry, ludzki_gracz)
    
    if potencjalny_fork_przeciwnika:
        for rzad, kolumna in stan_gry.otrzymaj_mozliwe_ruchy():
            if (rzad, kolumna) == potencjalny_fork_przeciwnika:
                continue
                
            kopia_stanu = StanGry(stan_gry.rozmiar_planszy, stan_gry.warunek_wygranej)
            kopia_stanu.plansza = stan_gry.plansza.copy()
            kopia_stanu.obecny_gracz = graczSI
            
            kopia_stanu.wykonaj_ruch(rzad, kolumna)
            kopia_stanu.obecny_gracz = graczSI
            ruch_wygrywajacy = _znajdz_wygrywajacy_ruch(kopia_stanu, graczSI)
            
            if ruch_wygrywajacy and ruch_wygrywajacy != potencjalny_fork_przeciwnika:
                return (rzad, kolumna)
        
        return potencjalny_fork_przeciwnika
    
    return None


def _znajdz_przeciwny_rog(stan_gry: StanGry, przeciwnik: int) -> Optional[Tuple[int, int]]:
    rogi = [(0, 0), (0, 2), (2, 0), (2, 2)]
    przeciwne_rogi = {
        (0, 0): (2, 2),
        (2, 2): (0, 0),
        (0, 2): (2, 0),
        (2, 0): (0, 2)
    }
    
    for rog in rogi:
        if stan_gry.plansza[rog[0], rog[1]] == przeciwnik:
            przeciwny = przeciwne_rogi[rog]
            if stan_gry.plansza[przeciwny[0], przeciwny[1]] == 0:
                return przeciwny
    
    return None


def _znajdz_pusty_rog(stan_gry: StanGry) -> Optional[Tuple[int, int]]:
    rogi = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
    for rog in rogi:
        if stan_gry.plansza[rog[0], rog[1]] == 0:
            return rog
    
    return None


def _znajdz_pusta_krawedz(stan_gry: StanGry) -> Optional[Tuple[int, int]]:
    krawedzie = [(0, 1), (1, 0), (1, 2), (2, 1)]
    
    for krawedz in krawedzie:
        if stan_gry.plansza[krawedz[0], krawedz[1]] == 0:
            return krawedz
    
    return None


def _prosta_strategia(stan_gry: StanGry) -> Optional[Tuple[int, int]]:
    graczSI = stan_gry.obecny_gracz
    ludzki_gracz = -graczSI
    
    wygrywajacy_ruch = _znajdz_wygrywajacy_ruch(stan_gry, graczSI)
    if wygrywajacy_ruch:
        return wygrywajacy_ruch
    
    blokujacy_ruch = _znajdz_wygrywajacy_ruch(stan_gry, ludzki_gracz)
    if blokujacy_ruch:
        return blokujacy_ruch

    srodek = (stan_gry.rozmiar_planszy // 2, stan_gry.rozmiar_planszy // 2)
    if stan_gry.plansza[srodek[0], srodek[1]] == 0:
        return srodek
    
    dostepne_ruchy = stan_gry.otrzymaj_mozliwe_ruchy()
    if dostepne_ruchy:
        return dostepne_ruchy[0]
    
    return None