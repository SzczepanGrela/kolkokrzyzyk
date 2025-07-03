from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QComboBox, QLabel
)
from PySide6.QtCore import Qt, Signal
from typing import List, Tuple


class GlowneOkno(QMainWindow):
    plansza_kliknieta = Signal(int, int)
    zazadano_nowa_gre = Signal()
    tryb_gry_zmieniony = Signal(str)
    rozmiar_planszy_zmieniony = Signal(str)
    zazadano_wizualizacje = Signal()
    
    def __init__(self, rozmiar_planszy: int = 3):
        super().__init__()
        self.przyciski_planszy: List[List[QPushButton]] = []
        self.obecny_rozmiar_planszy = rozmiar_planszy
        self.przygotuj_interfejs_uzytkownika()
        
    def przygotuj_interfejs_uzytkownika(self):
        self.setWindowTitle("AI - Kolko i Krzyzyk")
        self.setMinimumSize(500, 600)
        
        centralny_widzet = QWidget()
        self.setCentralWidget(centralny_widzet)
        glowny_uklad = QVBoxLayout(centralny_widzet)
        
        self._stworz_panel_sterowania(glowny_uklad)
        self._stworz_plansze_gry(glowny_uklad)
        self._stworz_panel_informacyjny(glowny_uklad)
        
    def _stworz_panel_sterowania(self, uklad_rodzic: QVBoxLayout):
        panel_sterowania = QHBoxLayout()
        uklad_rodzic.addLayout(panel_sterowania)
        
        etykieta_trybu = QLabel("Tryb gry:")
        panel_sterowania.addWidget(etykieta_trybu)
        
        self.tryby_gry_lista_rozwijana = QComboBox()
        game_modes = [
            "Gracz vs Gracz",
            "Gracz vs Minimax",
            "Gracz vs Reguły", 
            "Gracz vs Q-learning",
            "Gracz vs MCTS"
        ]
        
        self.tryby_gry_lista_rozwijana.addItems(game_modes)
        self.tryby_gry_lista_rozwijana.currentTextChanged.connect(self.tryb_gry_zmieniony.emit)
        panel_sterowania.addWidget(self.tryby_gry_lista_rozwijana)
        
        etykieta_rozmiaru = QLabel("Rozmiar planszy: 3x3 (kólko i krzyżyk)")
        panel_sterowania.addWidget(etykieta_rozmiaru)
        
        panel_sterowania.addStretch()
        
        self.przycisk_nowej_gry = QPushButton("Nowa Gra")
        self.przycisk_nowej_gry.clicked.connect(self.zazadano_nowa_gre.emit)
        panel_sterowania.addWidget(self.przycisk_nowej_gry)
        
        self.przycisk_wizualizacji = QPushButton("Wizualizuj ostatni ruch AI")
        self.przycisk_wizualizacji.clicked.connect(self.zazadano_wizualizacje.emit)
        self.przycisk_wizualizacji.setEnabled(False)
        panel_sterowania.addWidget(self.przycisk_wizualizacji)
        
    def _stworz_plansze_gry(self, uklad_rodzic: QVBoxLayout):
        self.widzet_planszy = QWidget()
        self.widzet_planszy.setMaximumSize(400, 400)
        uklad_rodzic.addWidget(self.widzet_planszy, alignment=Qt.AlignCenter)
        self._przebuduj_plansze(3)
        
    def _przebuduj_plansze(self, rozmiar: int):
        if hasattr(self, 'uklad_planszy'):
            while self.uklad_planszy.count():
                element = self.uklad_planszy.takeAt(0)
                if element.widget():
                    element.widget().deleteLater()
            self.uklad_planszy.deleteLater()
            
        self.przyciski_planszy = []
        self.uklad_planszy = QGridLayout(self.widzet_planszy)
        self.uklad_planszy.setSpacing(2)
        
        rozmiar_przycisku = 100
        
        for rzad in range(rozmiar):
            rzad_przyciskow = []
            for kolumna in range(rozmiar):
                przycisk = QPushButton()
                przycisk.setFixedSize(rozmiar_przycisku, rozmiar_przycisku)
                przycisk.setStyleSheet("""
                    QPushButton {
                        font-size: 36px;
                        font-weight: bold;
                        background-color: #f0f0f0;
                        border: 2px solid #888;
                        border-radius: 8px;
                    }
                    QPushButton:hover:enabled {
                        background-color: #e0e0e0;
                        border-color: #666;
                    }
                    QPushButton:disabled {
                        background-color: #d0d0d0;
                        color: #333;
                    }
                """)
                
                przycisk.clicked.connect(lambda checked, r=rzad, c=kolumna: self.plansza_kliknieta.emit(r, c))
                
                self.uklad_planszy.addWidget(przycisk, rzad, kolumna)
                rzad_przyciskow.append(przycisk)
                
            self.przyciski_planszy.append(rzad_przyciskow)
            
    def _stworz_panel_informacyjny(self, uklad_rodzic: QVBoxLayout):
        self.panel_informacyjny = QLabel("Twoja kolej")
        self.panel_informacyjny.setAlignment(Qt.AlignCenter)
        self.panel_informacyjny.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333333;
                padding: 10px;
                background-color: #f0f0f0;
                border: 2px solid #888888;
                border-radius: 8px;
            }
        """)
        uklad_rodzic.addWidget(self.panel_informacyjny)
        

        
    def zaktualizuj_przycisk(self, rzad: int, kolumna: int, tekst: str):
        if 0 <= rzad < len(self.przyciski_planszy) and 0 <= kolumna < len(self.przyciski_planszy[0]):
            self.przyciski_planszy[rzad][kolumna].setText(tekst)
            self.przyciski_planszy[rzad][kolumna].setEnabled(False)
            
    def zaktualizuj_panel_informacyjny(self, wiadomosc: str):
        self.panel_informacyjny.setText(wiadomosc)
        
    def zresetuj_plansze(self):
        for rzad in self.przyciski_planszy:
            for przycisk in rzad:
                przycisk.setText("")
                przycisk.setEnabled(True)
                
    def wylacz_plansze(self):
        for rzad in self.przyciski_planszy:
            for przycisk in rzad:
                przycisk.setEnabled(False)
                
    def wlacz_plansze(self):
        for rzad in self.przyciski_planszy:
            for przycisk in rzad:
                if przycisk.text() == "":
                    przycisk.setEnabled(True)
                    
    def wlacz_przycisk_wizualizacji(self):
        self.przycisk_wizualizacji.setEnabled(True)
        
    def wylacz_przycisk_wizualizacji(self):
        self.przycisk_wizualizacji.setEnabled(False)