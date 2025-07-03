from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsTextItem, QGraphicsLineItem, QLabel,
    QPushButton, QScrollArea, QSlider
)
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPen, QBrush, QColor, QFont
from typing import Dict, Tuple
from ai.minimax import Wezel


class OknoWizualizacji(QMainWindow):
    def __init__(self, tree_root: Wezel):
        super().__init__()
        self.tree_root = tree_root
        self.node_positions: Dict[Wezel, Tuple[float, float]] = {}
        self.node_width = 120
        self.node_height = 80
        self.level_height = 150
        self.node_spacing = 20
        
        self.setup_ui()
        self.draw_tree()
        
    def setup_ui(self):
        self.setWindowTitle("Wizualizacja Drzewa Przeszukiwania Minimax")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        info_label = QLabel("Drzewo Przeszukiwania Minimax - Niebieski: Maksymalizujacy, Czerwony: Minimalizujacy, Zielony: Najlepsza Sciezka")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f0f0f0;")
        layout.addWidget(info_label)
        
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        
        self.graphics_view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.graphics_view.setRenderHint(self.graphics_view.renderHints())
        
        layout.addWidget(self.graphics_view)
        
        button_layout = QHBoxLayout()
        
        zoom_in_btn = QPushButton("Przybliz")
        zoom_in_btn.clicked.connect(self.zoom_in)
        button_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("Oddal")
        zoom_out_btn.clicked.connect(self.zoom_out)
        button_layout.addWidget(zoom_out_btn)
        
        fit_btn = QPushButton("Dopasuj do Okna")
        fit_btn.clicked.connect(self.fit_to_window)
        button_layout.addWidget(fit_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Zamknij")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
    def calculate_tree_layout(self, node: Wezel, depth: int = 0, x_offset: float = 0) -> float:
        if not node.dzieci:
            x = x_offset
            y = depth * self.level_height
            self.node_positions[node] = (x, y)
            return self.node_width
        
        child_x = x_offset
        total_width = 0
        
        for child in node.dzieci:
            child_width = self.calculate_tree_layout(child, depth + 1, child_x)
            child_x += child_width + self.node_spacing
            total_width += child_width + self.node_spacing
        
        total_width -= self.node_spacing
        
        if node.dzieci:
            first_child_x = self.node_positions[node.dzieci[0]][0]
            last_child_x = self.node_positions[node.dzieci[-1]][0]
            center_x = (first_child_x + last_child_x) / 2
        else:
            center_x = x_offset
            
        y = depth * self.level_height
        self.node_positions[node] = (center_x, y)
        
        return max(total_width, self.node_width)
    
    def draw_tree(self):
        if not self.tree_root:
            return
            
        self.calculate_tree_layout(self.tree_root)
        best_path = self.find_best_path()
        self.draw_edges(self.tree_root, best_path)
        self.draw_nodes(self.tree_root, best_path)
        self.update_scene_bounds()
    
    def find_best_path(self) -> set:
        best_path = set()
        
        def trace_best_path(node: Wezel):
            best_path.add(node)
            
            if node.dzieci:
                if node.czy_tura_max:
                    best_child = max(node.dzieci, key=lambda c: c.wynik if c.wynik is not None else float('-inf'))
                else:
                    best_child = min(node.dzieci, key=lambda c: c.wynik if c.wynik is not None else float('inf'))
                
                trace_best_path(best_child)
        
        if self.tree_root:
            trace_best_path(self.tree_root)
            
        return best_path
    
    def draw_nodes(self, node: Wezel, best_path: set):
        if node not in self.node_positions:
            return
            
        x, y = self.node_positions[node]
        
        rect = QGraphicsRectItem(x - self.node_width/2, y - self.node_height/2, 
                                self.node_width, self.node_height)
        
        if node in best_path:
            rect.setBrush(QBrush(QColor(144, 238, 144)))
            rect.setPen(QPen(QColor(34, 139, 34), 3))
        elif node.czy_tura_max:
            rect.setBrush(QBrush(QColor(173, 216, 230)))
            rect.setPen(QPen(QColor(0, 100, 200), 2))
        else:
            rect.setBrush(QBrush(QColor(255, 182, 193)))
            rect.setPen(QPen(QColor(200, 0, 0), 2))
            
        self.graphics_scene.addItem(rect)
        
        text_lines = []
        
        if node.ruch:
            text_lines.append(f"Ruch: {node.ruch}")
        else:
            text_lines.append("Korzeń")
            
        if node.wynik is not None:
            text_lines.append(f"Wynik: {node.wynik}")
            
        if node.czy_tura_max is not None:
            text_lines.append("MAKS" if node.czy_tura_max else "MIN")
        
        if node.alfa is not None and node.beta is not None:
            text_lines.append(f"α={node.alfa:.1f} β={node.beta:.1f}")
            
        text = "\n".join(text_lines)
        text_item = QGraphicsTextItem(text)
        text_item.setPos(x - self.node_width/2 + 5, y - self.node_height/2 + 5)
        
        font = QFont("Arial", 9)
        text_item.setFont(font)
        
        self.graphics_scene.addItem(text_item)
        
        for child in node.dzieci:
            self.draw_nodes(child, best_path)
    
    def draw_edges(self, node: Wezel, best_path: set):
        if node not in self.node_positions:
            return
            
        x1, y1 = self.node_positions[node]
        
        for child in node.dzieci:
            if child not in self.node_positions:
                continue
                
            x2, y2 = self.node_positions[child]
            
            line = QGraphicsLineItem(x1, y1 + self.node_height/2, x2, y2 - self.node_height/2)
            
            if node in best_path and child in best_path:
                line.setPen(QPen(QColor(34, 139, 34), 3))
            else:
                line.setPen(QPen(QColor(128, 128, 128), 1))
                
            self.graphics_scene.addItem(line)
            
            self.draw_edges(child, best_path)
    
    def update_scene_bounds(self):
        if not self.node_positions:
            return
            
        min_x = min(pos[0] for pos in self.node_positions.values()) - self.node_width
        max_x = max(pos[0] for pos in self.node_positions.values()) + self.node_width
        min_y = min(pos[1] for pos in self.node_positions.values()) - self.node_height
        max_y = max(pos[1] for pos in self.node_positions.values()) + self.node_height
        
        self.graphics_scene.setSceneRect(min_x, min_y, max_x - min_x, max_y - min_y)
    
    def zoom_in(self):
        self.graphics_view.scale(1.2, 1.2)
    
    def zoom_out(self):
        self.graphics_view.scale(0.8, 0.8)
    
    def fit_to_window(self):
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
    
    def visualize_tree(self, tree_root: Wezel):
        self.tree_root = tree_root
        self.node_positions.clear()
        self.graphics_scene.clear()
        self.draw_tree()