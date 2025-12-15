import cv2
import numpy as np
import tensorflow as tf
import time
import folium
from folium.plugins import HeatMap
import os
import random
from datetime import datetime

# --- CONFIGURACIÓN DE LA MISIÓN ---
MODEL_PATH = "models/plant_doctor_thor.tflite"
OUTPUT_DIR = "output"
CONFIDENCE_THRESHOLD = 0.60  # Solo marcar si estamos seguros

# Clases (Mismo orden que entrenamiento)
CLASS_NAMES = ['deficiencia', 'fusario', 'sanas']
# Colores: Deficiencia (Amarillo), Fusario (Rojo), Sanas (Verde)
COLORS = [(0, 255, 255), (0, 0, 255), (0, 255, 0)] 
# Colores Hex para el mapa
HEX_COLORS = {'deficiencia': '#FFFF00', 'fusario': '#FF0000', 'sanas': '#00FF00'}

# Simulación GPS (Empezamos en coordenadas de una granja ficticia)
START_LAT = 10.1565
START_LON = -67.9899 

class DroneBrain:
    def __init__(self):
        self.load_model()
        self.flight_log = [] # Aquí guardaremos los "breadcrumbs"
        self.current_lat = START_LAT
        self.current_lon = START_LON
        
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    def load_model(self):
        print("🚁 Cargando sistemas de IA a bordo...")
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3] # (224, 224)
        print("✅ Sistemas listos. Esperando video...")

    def preprocess(self, frame):
        # 1. Convertir BGR (OpenCV) a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 2. Redimensionar a 224x224
        resized = cv2.resize(rgb_frame, (224, 224))
        # 3. EfficientNet espera 0-255 float32
        input_data = np.array(resized, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        return input_data

    def infer(self, frame):
        input_data = self.preprocess(frame)
        
        # Inferencia TFLite
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        idx = np.argmax(output_data)
        confidence = output_data[idx]
        label = CLASS_NAMES[idx]
        
        return label, confidence, idx

    def update_gps(self):
        # Simula que el drone vuela en línea recta con un poco de ruido
        self.current_lat += 0.00005 + random.uniform(-0.00001, 0.00001)
        self.current_lon += random.uniform(-0.00001, 0.00001)

    def generate_map(self):
        print("🗺️ Generando Mapa de Calor de la Misión...")
        if not self.flight_log:
            print("⚠️ No hay datos de vuelo.")
            return

        # Centro del mapa
        center_lat = sum(x['lat'] for x in self.flight_log) / len(self.flight_log)
        center_lon = sum(x['lon'] for x in self.flight_log) / len(self.flight_log)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=19, tiles='OpenStreetMap')

        # 1. Puntos individuales
        for point in self.flight_log:
            folium.CircleMarker(
                location=[point['lat'], point['lon']],
                radius=3,
                color=HEX_COLORS[point['label']],
                fill=True,
                fill_opacity=0.7,
                popup=f"{point['label'].upper()} ({point['conf']:.0f}%)"
            ).add_to(m)

        # 2. Capa de Calor (Heatmap) - Solo para enfermedades
        heat_data = [[p['lat'], p['lon']] for p in self.flight_log if p['label'] != 'sanas']
        if heat_data:
            HeatMap(heat_data, radius=15).add_to(m)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/mission_report_{timestamp}.html"
        m.save(filename)
        print(f"✅ Reporte guardado: {filename}")

    def fly(self):
        # Abrir cámara (0 suele ser la webcam)
        cap = cv2.VideoCapture(0) 
        
        if not cap.isOpened():
            print("❌ Error: No se detecta cámara.")
            return

        print("🎥 INICIANDO VUELO DE RECONOCIMIENTO (Presiona 'Q' para aterrizar)")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Procesar cada 5 frames para no saturar y simular velocidad de movimiento
            if frame_count % 5 == 0:
                self.update_gps()
                
                label, conf, idx = self.infer(frame)
                
                # HUD (Head-Up Display)
                color = COLORS[idx]
                label_text = f"{label.upper()} {conf*100:.1f}%"
                
                # Dibujar rectángulo y texto
                cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1) # Fondo negro
                cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"GPS: {self.current_lat:.4f}, {self.current_lon:.4f}", (20, 450), 
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

                # Registrar dato si la confianza es buena
                if conf > CONFIDENCE_THRESHOLD:
                    self.flight_log.append({
                        'lat': self.current_lat,
                        'lon': self.current_lon,
                        'label': label,
                        'conf': conf * 100
                    })
            
            cv2.imshow('Drone View - Plant Doctor AI', frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.generate_map()

if __name__ == "__main__":
    drone = DroneBrain()
    drone.fly()