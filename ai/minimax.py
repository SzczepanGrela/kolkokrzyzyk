from typing import Tuple, Optional, List
import copy
import random
from gra.logika import StanGry

class Wezel:
    def __init__(self,
                 stan_gry: StanGry,
                 ruch: Optional[Tuple[int, int]] = None,
                 wynik: Optional[float] = None,
                 glebokosc: int = 0):
        self.stan_gry = stan_gry
        self.ruch = ruch
        self.wynik = wynik
        self.glebokosc = glebokosc
        self.dzieci: List['Wezel'] = []
        self.czy_tura_max: Optional[bool] = None
        self.alfa: Optional[float] = None
        self.beta: Optional[float] = None

def ocen_stan_gry(stan_gry: StanGry, graczSI: int, glebokosc: int = 0) -> int:
    zwyciezca = stan_gry.sprawdz_zwyciezce()
    if zwyciezca == graczSI:
        return 10 + glebokosc
    elif zwyciezca == -graczSI:
        return -10 - glebokosc
    else:
        return 0

def minimax(stan_gry: StanGry,
            glebokosc: int,
            alfa: float,
            beta: float,
            czy_tura_max: bool,
            graczSI: int) -> int:
    if glebokosc == 0 or stan_gry.czy_koniec_gry():
        return ocen_stan_gry(stan_gry, graczSI, glebokosc)

    if czy_tura_max:
        max_ocena = float('-inf')
        for rzad, kolumna in stan_gry.otrzymaj_mozliwe_ruchy():
            kopia_stanu = copy.deepcopy(stan_gry)
            kopia_stanu.wykonaj_ruch(rzad, kolumna)
            ocena_ruchu = minimax(kopia_stanu, glebokosc - 1, alfa, beta, False, graczSI)
            max_ocena = max(max_ocena, ocena_ruchu)
            alfa = max(alfa, ocena_ruchu)
            if beta <= alfa:
                break
        return max_ocena
    else:
        min_ocena = float('inf')
        for rzad, kolumna in stan_gry.otrzymaj_mozliwe_ruchy():
            kopia_stanu = copy.deepcopy(stan_gry)
            kopia_stanu.wykonaj_ruch(rzad, kolumna)
            ocena_ruchu = minimax(kopia_stanu, glebokosc - 1, alfa, beta, True, graczSI)
            min_ocena = min(min_ocena, ocena_ruchu)
            beta = min(beta, ocena_ruchu)
            if beta <= alfa:
                break
        return min_ocena

def minimax_z_drzewem(stan_gry: StanGry,
                      glebokosc: int,
                      alfa: float,
                      beta: float,
                      czy_tura_max: bool,
                      max_glebokosc: int,
                      graczSI: int) -> Tuple[int, Wezel]:
    poziom = max_glebokosc - glebokosc
    wezel = Wezel(stan_gry, glebokosc=poziom)
    wezel.czy_tura_max = czy_tura_max
    wezel.alfa = alfa
    wezel.beta = beta

    if glebokosc == 0 or stan_gry.czy_koniec_gry():
        ocena = ocen_stan_gry(stan_gry, graczSI, glebokosc)
        wezel.wynik = ocena
        return ocena, wezel

    if czy_tura_max:
        najlepsza_ocena = float('-inf')
        for rzad, kolumna in stan_gry.otrzymaj_mozliwe_ruchy():
            kopia_stanu = copy.deepcopy(stan_gry)
            kopia_stanu.wykonaj_ruch(rzad, kolumna)
            ocena_dziecka, wezel_dziecka = minimax_z_drzewem(
                kopia_stanu, glebokosc - 1, alfa, beta, False, max_glebokosc, graczSI)
            wezel_dziecka.ruch = (rzad, kolumna)
            wezel.dzieci.append(wezel_dziecka)
            najlepsza_ocena = max(najlepsza_ocena, ocena_dziecka)
            alfa = max(alfa, ocena_dziecka)
            if beta <= alfa:
                break
        wezel.wynik = najlepsza_ocena
        return najlepsza_ocena, wezel
    else:
        najlepsza_ocena = float('inf')
        for rzad, kolumna in stan_gry.otrzymaj_mozliwe_ruchy():
            kopia_stanu = copy.deepcopy(stan_gry)
            kopia_stanu.wykonaj_ruch(rzad, kolumna)
            ocena_dziecka, wezel_dziecka = minimax_z_drzewem(
                kopia_stanu, glebokosc - 1, alfa, beta, True, max_glebokosc, graczSI)
            wezel_dziecka.ruch = (rzad, kolumna)
            wezel.dzieci.append(wezel_dziecka)
            najlepsza_ocena = min(najlepsza_ocena, ocena_dziecka)
            beta = min(beta, ocena_dziecka)
            if beta <= alfa:
                break
        wezel.wynik = najlepsza_ocena
        return najlepsza_ocena, wezel

def dobierz_glebokosc(rozmiar_planszy: int, liczba_pustych_pol: int) -> int:
    if rozmiar_planszy == 3:
        return min(liczba_pustych_pol, 9)


def znajdz_najlepszy_ruch(stan_gry: StanGry,
                          glebokosc: Optional[int] = None) -> Optional[Tuple[int, int]]:
    if stan_gry.czy_koniec_gry():
        return None

    dostepne_ruchy = stan_gry.otrzymaj_mozliwe_ruchy()
    if not dostepne_ruchy:
        return None

    if glebokosc is None:
        glebokosc = dobierz_glebokosc(stan_gry.rozmiar_planszy, len(dostepne_ruchy))

    graczSI = stan_gry.obecny_gracz
    czy_tura_max = (stan_gry.obecny_gracz == graczSI)
    najlepsza_ocena = float('-inf') if czy_tura_max else float('inf')
    najlepsze_ruchy: List[Tuple[int, int]] = []

    for rzad, kolumna in dostepne_ruchy:
        kopia_stanu = copy.deepcopy(stan_gry)
        kopia_stanu.wykonaj_ruch(rzad, kolumna)
        wynik = minimax(
            kopia_stanu,
            glebokosc - 1,
            float('-inf'),
            float('inf'),
            not czy_tura_max,
            graczSI
        )
        if czy_tura_max:
            if wynik > najlepsza_ocena:
                najlepsza_ocena = wynik
                najlepsze_ruchy = [(rzad, kolumna)]
            elif wynik == najlepsza_ocena:
                najlepsze_ruchy.append((rzad, kolumna))
        else:
            if wynik < najlepsza_ocena:
                najlepsza_ocena = wynik
                najlepsze_ruchy = [(rzad, kolumna)]
            elif wynik == najlepsza_ocena:
                najlepsze_ruchy.append((rzad, kolumna))

    return random.choice(najlepsze_ruchy) if najlepsze_ruchy else None

def znajdz_najlepszy_ruch_z_drzewem(stan_gry: StanGry,
                                    graczSI: int,
                                    glebokosc: Optional[int] = None) -> Tuple[Optional[Tuple[int, int]], Optional[Wezel]]:
    dostepne_ruchy = stan_gry.otrzymaj_mozliwe_ruchy()
    if not dostepne_ruchy:
        return None, None

    if glebokosc is None:
        glebokosc = dobierz_glebokosc(stan_gry.rozmiar_planszy, len(dostepne_ruchy))

    korzen = Wezel(stan_gry)
    korzen.czy_tura_max = (stan_gry.obecny_gracz == graczSI)
    najlepsza_ocena = float('-inf') if korzen.czy_tura_max else float('inf')
    najlepszy_ruch: Optional[Tuple[int, int]] = None

    for rzad, kolumna in dostepne_ruchy:
        kopia_stanu = copy.deepcopy(stan_gry)
        kopia_stanu.wykonaj_ruch(rzad, kolumna)
        ocena, wezel_dziecka = minimax_z_drzewem(
            kopia_stanu,
            glebokosc - 1,
            float('-inf'),
            float('inf'),
            not korzen.czy_tura_max,
            glebokosc,
            graczSI
        )
        wezel_dziecka.ruch = (rzad, kolumna)
        korzen.dzieci.append(wezel_dziecka)

        if korzen.czy_tura_max and ocena > najlepsza_ocena:
            najlepsza_ocena = ocena
            najlepszy_ruch = (rzad, kolumna)
        elif not korzen.czy_tura_max and ocena < najlepsza_ocena:
            najlepsza_ocena = ocena
            najlepszy_ruch = (rzad, kolumna)

    korzen.wynik = najlepsza_ocena
    return najlepszy_ruch, korzen
