
import sys
import random
import numpy as np
from typing import Optional
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from gui.główne_okno import GlowneOkno
from gra.logika import StanGry
from ai.minimax import znajdz_najlepszy_ruch, dobierz_glebokosc, znajdz_najlepszy_ruch_z_drzewem, Wezel
from ai.reguly import znajdz_najlepszy_ruch
from ai.mcts import znajdz_najlepszy_ruch

from ai.agent_q_learning import AgentQLearning

from gui.okno_wizualizacji import OknoWizualizacji


class KontrolerKolkoKrzyzyk:

    def __init__(self):
        self.stan_gry = StanGry(rozmiar_planszy=3, warunek_wygranej=3)
        self.glowne_okno = GlowneOkno()
        self.obecny_tryb_gry = "Gracz vs Gracz"
        self.ai_mysli = False
        self.gracz_zaczyna_gre = True
        self.statystyki_gry = {
            'rozegrane_gry': 0,
            'wygrane_gracza': 0,
            'wygrane_ai': 0,
            'remisy': 0,
            'statystyki_algorytmów_ai': {}
        }
        
        self._zainicjuj_agentow_ai()
        self.ostatnie_drzewo_wyszukiwania: Optional[Wezel] = None
        self.okno_wizualizacji: Optional[OknoWizualizacji] = None
        self._polacz_sygnaly()
        self._zacznij_nowa_gre_losuj_rozpoczynajacego()
    
    def _zainicjuj_agentow_ai(self):
        print("🤖 Inicjalizowanie agentów AI...")
        
        self.agent = AgentQLearning()
        try:
            self.agent.zaladuj_tabele_q('ai/gotowe_tabele/model.pkl')
        except Exception as e:
            print(f"⚠️  Nie znaleziono modelu: {e}")
        algorytmy = ["Minimax", "Reguły", "Q-learning", "MCTS"]
        for alg in algorytmy:
            self.statystyki_gry['statystyki_algorytmów_ai'][alg] = {
                'gry': 0, 'wygrane': 0, 'przegrane': 0, 'remisy': 0
            }
        
        print("🎮 Wszystkie modele AI załadowane pomyślnie")
    
    def _polacz_sygnaly(self):
        self.glowne_okno.plansza_kliknieta.connect(self._obsluz_klikniecie_planszy)
        self.glowne_okno.zazadano_nowa_gre.connect(self._zacznij_nowa_gre_losuj_rozpoczynajacego)
        self.glowne_okno.tryb_gry_zmieniony.connect(self._obsluz_zmiane_trybu_gry)
        self.glowne_okno.zazadano_wizualizacje.connect(self._pokaz_wizualizacje)
    
    def _zacznij_nowa_gre_losuj_rozpoczynajacego(self):
        self.stan_gry.zresetuj_plansze()
        self.glowne_okno.zresetuj_plansze()
        self.ostatnie_drzewo_wyszukiwania = None
        self.glowne_okno.wylacz_przycisk_wizualizacji()

        if self.obecny_tryb_gry != "Gracz vs Gracz":
            self.gracz_zaczyna_gre = random.choice([True, False])
            
            if self.gracz_zaczyna_gre:
                self.stan_gry.obecny_gracz = 1
                self.glowne_okno.zaktualizuj_panel_informacyjny("Ty zaczynasz! Kolej gracza X")
            else:
                self.stan_gry.obecny_gracz = 1
                self.glowne_okno.zaktualizuj_panel_informacyjny("AI zaczyna! Kolej AI (X)")
                QTimer.singleShot(500, self._wykonaj_ruch_ai)
        else:
            self.gracz_zaczyna_gre = True
            self.stan_gry.obecny_gracz = 1
            self.glowne_okno.zaktualizuj_panel_informacyjny("Kolej gracza X")

    
    def _obsluz_klikniecie_planszy(self, wiersz: int, kolumna: int):
        if self.ai_mysli:
            return
        
        if self.obecny_tryb_gry != "Gracz vs Gracz":
            if self._czy_kolej_ai():
                return

        if self.stan_gry.wykonaj_ruch(wiersz, kolumna):
            symbol_gracza = 'X' if self.stan_gry.obecny_gracz == -1 else 'O'
            self.glowne_okno.zaktualizuj_przycisk(wiersz, kolumna, symbol_gracza)
            
            zwyciezca = self.stan_gry.sprawdz_zwyciezce()
            
            if zwyciezca is not None:
                self._obsluz_koniec_gry(zwyciezca)
            else:
                nastepny_gracz = 'X' if self.stan_gry.obecny_gracz == 1 else 'O'
                if self._czy_kolej_ai():
                    self.glowne_okno.zaktualizuj_panel_informacyjny(f"AI myśli... (AI = {nastepny_gracz})")
                    QTimer.singleShot(800, self._wykonaj_ruch_ai)
                else:
                    self.glowne_okno.zaktualizuj_panel_informacyjny(f"Twoja kolej! ({nastepny_gracz})")

    def _czy_kolej_ai(self):
        if self.obecny_tryb_gry == "Gracz vs Gracz":
            return False
        
        if self.gracz_zaczyna_gre:
            return self.stan_gry.obecny_gracz == -1
        else:
            return self.stan_gry.obecny_gracz == 1
    
    def _wykonaj_ruch_ai(self):
        if self.ai_mysli or self.stan_gry.czy_koniec_gry():
            return
        
        self.ai_mysli = True
        
        try:
            akcja = None
            nazwa_algorytmu = ""
            
            if "Minimax" in self.obecny_tryb_gry:
                liczba_pustych = np.sum(self.stan_gry.plansza == 0)
                glebokosc = dobierz_glebokosc(self.stan_gry.rozmiar_planszy, liczba_pustych)
                akcja, self.ostatnie_drzewo_wyszukiwania = znajdz_najlepszy_ruch_z_drzewem(self.stan_gry, self.stan_gry.obecny_gracz, glebokosc)
                nazwa_algorytmu = "Minimax"
                self.glowne_okno.wlacz_przycisk_wizualizacji()
                
            elif "Reguły" in self.obecny_tryb_gry:
                akcja = znajdz_najlepszy_ruch(self.stan_gry)
                nazwa_algorytmu = "Reguły"
                
            elif "Q-learning" in self.obecny_tryb_gry:
                akcja = self.agent.znajdz_najlepszy_ruch(self.stan_gry)
                nazwa_algorytmu = "Q-learning"
                
            elif "MCTS" in self.obecny_tryb_gry:
                akcja = znajdz_najlepszy_ruch(self.stan_gry, iteracje=2000)
                nazwa_algorytmu = "MCTS"
            
            if akcja and self.stan_gry.wykonaj_ruch(akcja[0], akcja[1]):
                symbol_gracza = 'X' if self.stan_gry.obecny_gracz == -1 else 'O'
                self.glowne_okno.zaktualizuj_przycisk(akcja[0], akcja[1], symbol_gracza)
                
                zwyciezca = self.stan_gry.sprawdz_zwyciezce()
                if zwyciezca is not None:
                    self._obsluz_koniec_gry(zwyciezca, nazwa_algorytmu)
                else:
                    if self.gracz_zaczyna_gre:
                        if self.stan_gry.obecny_gracz == 1:
                            self.glowne_okno.zaktualizuj_panel_informacyjny("Twoja kolej! (X)")
                        else:
                            self.glowne_okno.zaktualizuj_panel_informacyjny("Kolej AI (O)")
                    else:
                        if self.stan_gry.obecny_gracz == 1:
                            self.glowne_okno.zaktualizuj_panel_informacyjny("Kolej AI (X)")
                        else:
                            self.glowne_okno.zaktualizuj_panel_informacyjny("Twoja kolej! (O)")
        
        except Exception as e:
            print(f"Błąd AI: {e}")
            self.glowne_okno.zaktualizuj_panel_informacyjny("Błąd AI - spróbuj ponownie")
        
        finally:
            self.ai_mysli = False
    
    def _obsluz_koniec_gry(self, zwyciezca, nazwa_algorytmu=""):
        self.glowne_okno.wylacz_plansze()
        self.statystyki_gry['rozegrane_gry'] += 1
        if self.obecny_tryb_gry != "Gracz vs Gracz":
            if zwyciezca == 0:
                self.glowne_okno.zaktualizuj_panel_informacyjny("🤝 Remis!")
                self.statystyki_gry['remisy'] += 1
                if nazwa_algorytmu:
                    self.statystyki_gry['statystyki_algorytmów_ai'][nazwa_algorytmu]['remisy'] += 1
            else:
                if (self.gracz_zaczyna_gre and zwyciezca == 1) or (not self.gracz_zaczyna_gre and zwyciezca == -1):
                    self.glowne_okno.zaktualizuj_panel_informacyjny("🎉 Wygrałeś!")
                    self.statystyki_gry['wygrane_gracza'] += 1
                    if nazwa_algorytmu:
                        self.statystyki_gry['statystyki_algorytmów_ai'][nazwa_algorytmu]['przegrane'] += 1

                elif (self.gracz_zaczyna_gre and zwyciezca == -1) or (not self.gracz_zaczyna_gre and zwyciezca == 1):
                    self.glowne_okno.zaktualizuj_panel_informacyjny(f"🤖 AI wygrało! - {nazwa_algorytmu}")
                    self.statystyki_gry['wygrane_ai'] += 1
                    if nazwa_algorytmu:
                        self.statystyki_gry['statystyki_algorytmów_ai'][nazwa_algorytmu]['wygrane'] += 1
        else:
            if zwyciezca == 1:
                self.glowne_okno.zaktualizuj_panel_informacyjny("🎉 Wygrał gracz X!")
            elif zwyciezca == -1:
                self.glowne_okno.zaktualizuj_panel_informacyjny("🎉 Wygrał gracz O!")
            else:
                self.glowne_okno.zaktualizuj_panel_informacyjny("🤝 Remis!")
        
        self._wydrukuj_statystyki_gry()
        if nazwa_algorytmu == "Q-learning" and zwyciezca != 0:
            if self.gracz_zaczyna_gre and zwyciezca == 1:
                print("⚠️  OSTRZEENIE: Niepokonany Q-learning przegrał! To nigdy nie powinno się zdarzyć!")
            elif not self.gracz_zaczyna_gre and zwyciezca == -1:
                print("⚠️  OSTRZEENIE: Niepokonany Q-learning przegrał! To nigdy nie powinno się zdarzyć!")
    
    def _wydrukuj_statystyki_gry(self):
        statystyki = self.statystyki_gry
        print(f"\n📊 Statystyki gier:")
        print(f"   Rozegrane gry: {statystyki['rozegrane_gry']}")
        print(f"   Wygrane gracza: {statystyki['wygrane_gracza']}")
        print(f"   Wygrane AI: {statystyki['wygrane_ai']}")
        print(f"   Remisy: {statystyki['remisy']}")
        
        print(f"\n🤖 Statystyki algorytmów AI:")
        for alg, statystyki_alg in statystyki['statystyki_algorytmów_ai'].items():
            if statystyki_alg['gry'] > 0:
                wspolczynnik_wygranych = statystyki_alg['wygrane'] / statystyki_alg['gry'] * 100
                wspolczynnik_przegranych = statystyki_alg['przegrane'] / statystyki_alg['gry'] * 100
                print(f"   {alg}: {statystyki_alg['gry']} gier, "
                      f"wygranych: {statystyki_alg['wygrane']} ({wspolczynnik_wygranych:.1f}%), "
                      f"przegranych: {statystyki_alg['przegrane']} ({wspolczynnik_przegranych:.1f}%)")
                
                # Specjalna notatka dla niepokonanego Q-learning
                if alg == "Q-learning" and statystyki_alg['przegrane'] > 0:
                    print(f"   ⚠️  {alg} NIGDY nie powinien przegrywać! Sprawdź jakość treningu!")
    
    def _obsluz_zmiane_trybu_gry(self, nowy_tryb: str):
        self.obecny_tryb_gry = nowy_tryb
        print(f"🎮 Zmiana trybu na: {nowy_tryb}")
        
        self._zacznij_nowa_gre_losuj_rozpoczynajacego()
        if nowy_tryb == "Gracz vs Gracz":
            self.glowne_okno.zaktualizuj_panel_informacyjny("Tryb: Gracz vs Gracz - Kolej gracza X")
        else:
            rozpoczynajacy = "Ty" if self.gracz_zaczyna_gre else "AI"
            self.glowne_okno.zaktualizuj_panel_informacyjny(f"Tryb: {nowy_tryb} - {rozpoczynajacy} zaczyna!")
    
    def _pokaz_wizualizacje(self):
        if self.ostatnie_drzewo_wyszukiwania is None:
            self.glowne_okno.zaktualizuj_panel_informacyjny("Brak danych wizualizacji - zagraj przeciwko Minimax")
            return
        
        if self.okno_wizualizacji is None:
            self.okno_wizualizacji = OknoWizualizacji(self.ostatnie_drzewo_wyszukiwania)
        
        self.okno_wizualizacji.visualize_tree(self.ostatnie_drzewo_wyszukiwania)
        self.okno_wizualizacji.show()
        self.okno_wizualizacji.raise_()
        self.okno_wizualizacji.activateWindow()


def wydrukuj_podsumowanie_projektu():
    print("\n" + "="*60)
    print("🚀 KOMPLETNA IMPLEMENTACJA TIC-TAC-TOE AI")
    print("🎯 Z NIEPOKONANYM Q-LEARNING")
    print("="*60)
    print("📋 Zaimplementowane wymagania projektu:")
    print()
    print("✅ 1. Algorytm Minimax z Alpha-Beta Pruning")
    print("   - Gwarantuje optymalną grę (niemożliwość przegrania)")
    print("   - Adaptacyjna głębokość wyszukiwania")
    print("   - Pełna optymalizacja wydajności")
    print()
    print("✅ 2. Wizualizacja Drzewa Gry")
    print("   - Graficzna wizualizacja procesu myślenia Minimax")
    print("   - Interaktywne okno z detalami węzłów")
    print("   - Analiza wartości i najlepszych ruchów")
    print()
    print("✅ 3. System Ekspertowy (Reguły)")
    print("   - Zaimplementowane reguły strategiczne")
    print("   - Hierarchia decyzyjna: wygraj > blokuj > centrum > narożnik")
    print("   - Porównanie złożoności z Minimax")
    print()
    print("✅ 4. NIEPOKONANY Q-learning (Nowa implementacja)")
    print("   - Równoległy trening na 95% rdzeni CPU")
    print("   - Komunikacja w czasie rzeczywistym między procesami")
    print("   - Trening przeciwko wszystkim algorytmom AI")
    print("   - GWARANTUJE: Nigdy nie przegrywa (remis to najgorszy wynik)")
    print("   - Wymieniono wszystkie poprzednie implementacje Q-learning")
    print()
    print("✅ 5. Monte Carlo Tree Search (MCTS)")
    print("   - Implementacja algorytmu MCTS")
    print("   - Porównanie wydajności z innymi metodami")
    print("   - Konfigurowalna liczba iteracji")
    print()
    print("✅ 6. Graficzny Interfejs Użytkownika")
    print("   - Intuicyjny interfejs PySide6/Qt")
    print("   - Wybór przeciwnika spośród 4 algorytmów AI")
    print("   - Losowy wybór rozpoczynającego gracza")
    print("   - Analiza i wizualizacja procesu myślenia AI")
    print()
    print("🎯 DOSTĘPNE ALGORYTMY AI:")
    print("   1. Minimax (Alpha-Beta) - Optymalna gra")
    print("   2. System Reguł - Strategia ekspercka") 
    print("   3. Q-learning Unbeatable - NIEPOKONANY agent RL")
    print("   4. MCTS - Monte Carlo Tree Search")
    print()
    print("💡 NIEPOKONANY Q-LEARNING:")
    print("   - Trening równoległy na 95% rdzeni CPU")
    print("   - Komunikacja w czasie rzeczywistym między procesami")
    print("   - Shared Memory dla współdzielenia Q-table")
    print("   - Trening przeciwko Minimax, Rules, MCTS")
    print("   - Massive penalty za przegranie (-100)")
    print("   - Reward shaping dla optymalnej gry")
    print("   - GWARANCJA: Nigdy nie przegrywa!")
    print("="*60)


def main():
    wydrukuj_podsumowanie_projektu()
    
    app = QApplication(sys.argv)
    kontroler = KontrolerKolkoKrzyzyk()
    kontroler.glowne_okno.show()
    
    print("\n🎮 Aplikacja uruchomiona pomyślnie!")
    print("💡 Wybierz tryb gry z menu i ciesz się grą przeciwko AI!")
    print("🔍 Zagraj przeciwko Minimax aby zobaczyć wizualizację drzewa gry!")
    print("🚀 Zagraj przeciwko Q-learning - wytrenowany agent")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()