import cv2
import numpy as np
import tensorflow as tf
import time
import folium
from folium.plugins import HeatMap
import os
import random
from datetime import datetime
from collections import deque, Counter

# --- CONFIGURACIÓN DE LA MISIÓN ---
MODEL_PATH = "models/plant_doctor_thor.tflite"
OUTPUT_DIR = "output"
CONFIDENCE_THRESHOLD = 0.60 

CLASS_NAMES = ['deficiencia', 'fusario', 'sanas']

# COLORES HUD (BGR para OpenCV)
COLOR_DEFICIENCIA = (0, 255, 255) # Amarillo
COLOR_FUSARIO = (0, 0, 255)       # Rojo
COLOR_SANAS = (0, 255, 0)         # Verde
COLOR_NEUTRO = (255, 255, 255)    # Blanco (Sin detección)

# Colores Hex para el mapa
HEX_COLORS = {'deficiencia': '#FFFF00', 'fusario': '#FF0000', 'sanas': '#00FF00'}

# GPS Simulado
START_LAT = 10.1565
START_LON = -67.9899 

class DroneBrain:
    def __init__(self):
        self.load_model()
        self.flight_log = []
        self.current_lat = START_LAT
        self.current_lon = START_LON
        
        # --- ESTABILIZADOR DE IMAGEN (BUFFER) ---
        # Guardaremos las últimas 10 predicciones para sacar el promedio
        # Esto elimina el parpadeo
        self.prediction_buffer = deque(maxlen=10) 
        
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    def load_model(self):
        print("🚁 Cargando sistemas de IA a bordo...")
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("✅ Sistemas listos.")

    def preprocess(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (224, 224))
        # EfficientNet espera 0-255 float32
        input_data = np.array(resized, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        return input_data

    def infer(self, frame):
        input_data = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        idx = np.argmax(output_data)
        confidence = output_data[idx]
        label = CLASS_NAMES[idx]
        
        return label, confidence

    def get_stabilized_prediction(self, label, confidence):
        """
        Usa votación democrática de los últimos frames para decidir qué mostrar.
        """
        # Solo añadimos al buffer si la confianza es decente
        if confidence > 0.4:
            self.prediction_buffer.append(label)
        
        if len(self.prediction_buffer) < 3:
            return "ESCANEO...", 0.0

        # Votación: ¿Cuál es la clase más frecuente en el buffer?
        most_common = Counter(self.prediction_buffer).most_common(1)[0][0]
        return most_common, confidence

    def update_gps(self):
        self.current_lat += 0.00005 + random.uniform(-0.00001, 0.00001)
        self.current_lon += random.uniform(-0.00001, 0.00001)

    def draw_hud(self, frame, label, conf):
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        
        # Determinar color según amenaza
        if label == "fusario":
            color = COLOR_FUSARIO
            alert_text = "FUSARIUM ALERT!"
        elif label == "deficiencia":
            color = COLOR_DEFICIENCIA
            alert_text = "WARNING: DEFICIENCY"
        elif label == "sanas":
            color = COLOR_SANAS
            alert_text = "HEALTHY"
        else:
            color = COLOR_NEUTRO
            alert_text = "SEARCHING..."

        # 1. Mira de Apuntado (Target Box) en el centro
        box_size = 100
        top_left = (center_x - box_size, center_y - box_size)
        bottom_right = (center_x + box_size, center_y + box_size)
        
        # Grosor dinámico: Si detecta algo malo, la caja se hace más gruesa
        thickness = 4 if label in ["fusario", "deficiencia"] else 2
        
        cv2.rectangle(frame, top_left, bottom_right, color, thickness)
        
        # Líneas de mira (Crosshair)
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), color, 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), color, 1)

        # 2. Texto de Estado (Arriba)
        # Fondo negro semitransparente para el texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Texto grande
        cv2.putText(frame, f"{alert_text} ({conf*100:.0f}%)", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 3. Telemetría GPS (Abajo)
        cv2.putText(frame, f"LAT: {self.current_lat:.6f} | LON: {self.current_lon:.6f}", 
                    (30, height - 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)

        return frame

    def generate_map(self):
        print("🗺️ Generando Mapa...")
        if not self.flight_log: return

        center_lat = sum(x['lat'] for x in self.flight_log) / len(self.flight_log)
        center_lon = sum(x['lon'] for x in self.flight_log) / len(self.flight_log)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=19, tiles='OpenStreetMap')

        for point in self.flight_log:
            folium.CircleMarker(
                location=[point['lat'], point['lon']],
                radius=3,
                color=HEX_COLORS[point['label']],
                fill=True,
                fill_opacity=0.7,
                popup=f"{point['label'].upper()}"
            ).add_to(m)
        
        heat_data = [[p['lat'], p['lon']] for p in self.flight_log if p['label'] != 'sanas']
        if heat_data: HeatMap(heat_data, radius=15).add_to(m)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/mission_report_{timestamp}.html"
        m.save(filename)
        print(f"✅ Reporte: {filename}")

    def fly(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return

        print("🎥 VUELO INICIADO (Presiona 'Q' para salir)")
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Simulación GPS constante
            self.update_gps()

            # Inferencia cada frame (o cada N frames si va lento)
            raw_label, raw_conf = self.infer(frame)
            
            # SUAVIZADO: Usamos el buffer para decidir qué mostrar
            display_label, display_conf = self.get_stabilized_prediction(raw_label, raw_conf)
            
            # DIBUJAR HUD ESTILO MILITAR
            frame = self.draw_hud(frame, display_label, display_conf)

            # LOGGING: Solo guardamos puntos si estamos muy seguros
            if raw_conf > CONFIDENCE_THRESHOLD:
                self.flight_log.append({
                    'lat': self.current_lat, 'lon': self.current_lon,
                    'label': raw_label, 'conf': raw_conf * 100
                })

            cv2.imshow('Drone HUD - Plant Doctor', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.generate_map()

if __name__ == "__main__":
    drone = DroneBrain()
    drone.fly()