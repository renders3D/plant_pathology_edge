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

# 🛠️ MODO DE PRUEBA (IMPORTANTE)
# Asignar True si se está probando fuera de un cultivo (ideal para pruebas en interiores).
# Asignar False si se va a volar sobre un cultivo real (activa el filtro de vegetación).
TEST_MODE = False 

CLASS_NAMES = ['deficiencia', 'fusario', 'sanas']

# COLORES HUD
COLOR_DEFICIENCIA = (0, 255, 255) # Amarillo
COLOR_FUSARIO = (0, 0, 255)       # Rojo
COLOR_SANAS = (0, 255, 0)         # Verde
COLOR_NEUTRO = (200, 200, 200)    # Gris
COLOR_TEXTO = (255, 255, 255)

HEX_COLORS = {'deficiencia': '#FFFF00', 'fusario': '#FF0000', 'sanas': '#00FF00'}

VEGETATION_THRESHOLD = 0.20 
START_LAT = 10.1565
START_LON = -67.9899 

class DroneBrain:
    def __init__(self):
        self.load_model()
        self.flight_log = []
        self.current_lat = START_LAT
        self.current_lon = START_LON
        self.prediction_buffer = deque(maxlen=10) 
        
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    def load_model(self):
        print("🚁 Cargando sistemas de IA a bordo...")
        if not os.path.exists(MODEL_PATH):
            print(f"❌ ERROR CRÍTICO: No encuentro el modelo en {MODEL_PATH}")
            print("   Asegúrate de copiar el archivo .tflite en la carpeta 'models'.")
            exit()
            
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("✅ Sistemas listos.")

    def preprocess(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (224, 224))
        input_data = np.array(resized, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        return input_data

    def check_vegetation_presence(self, frame):
        if TEST_MODE: return True, 1.0 # Bypass para pruebas en interior

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_veg = np.array([20, 40, 40])
        upper_veg = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_veg, upper_veg)
        
        height, width = frame.shape[:2]
        total_pixels = height * width
        vegetation_pixels = cv2.countNonZero(mask)
        ratio = vegetation_pixels / total_pixels
        return ratio > VEGETATION_THRESHOLD, ratio

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
        if confidence > 0.4:
            self.prediction_buffer.append(label)
        if len(self.prediction_buffer) < 3:
            return "ESCANEO...", 0.0
        most_common = Counter(self.prediction_buffer).most_common(1)[0][0]
        return most_common, confidence

    def update_gps(self):
        self.current_lat += 0.00005 + random.uniform(-0.00001, 0.00001)
        self.current_lon += random.uniform(-0.00001, 0.00001)

    def draw_hud(self, frame, label, conf, is_scanning=False):
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        
        if is_scanning:
            color = COLOR_NEUTRO
            alert_text = "BUSCANDO CULTIVO..." if not TEST_MODE else "MODO TEST: ESCANEANDO"
            thickness = 1
            conf_display = ""
        else:
            if label == "fusario":
                color = COLOR_FUSARIO
                alert_text = "¡ALERTA FUSARIO!"
            elif label == "deficiencia":
                color = COLOR_DEFICIENCIA
                alert_text = "PRECAUCIÓN: DEFICIENCIA"
            elif label == "sanas":
                color = COLOR_SANAS
                alert_text = "CULTIVO SANO"
            else:
                color = COLOR_NEUTRO
                alert_text = "ANALIZANDO..."
            
            thickness = 4 if label in ["fusario", "deficiencia"] else 2
            conf_display = f"({conf*100:.0f}%)"

        # DIBUJO
        box_size = 80
        top_left = (center_x - box_size, center_y - box_size)
        bottom_right = (center_x + box_size, center_y + box_size)
        
        cv2.rectangle(frame, top_left, bottom_right, color, thickness)
        
        # Miras
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), color, 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), color, 1)

        # Panel Superior
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"{alert_text} {conf_display}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Panel Inferior
        cv2.putText(frame, f"LAT: {self.current_lat:.6f} | LON: {self.current_lon:.6f}", 
                    (30, height - 30), cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_TEXTO, 1)

        if TEST_MODE:
            cv2.putText(frame, "TEST MODE ON", (width - 200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return frame

    def generate_map(self):
        print("\n🗺️  Finalizando Misión. Generando reporte...")
        
        if not self.flight_log:
            print("⚠️  AVISO: No se registraron datos válidos durante el vuelo.")
            print("   (Posibles causas: Confianza baja del modelo o filtro de vegetación activo en interior)")
            print("   Intenta poner TEST_MODE = True si estás en interior.")
            return

        print(f"📊 Procesando {len(self.flight_log)} puntos de datos...")

        center_lat = sum(x['lat'] for x in self.flight_log) / len(self.flight_log)
        center_lon = sum(x['lon'] for x in self.flight_log) / len(self.flight_log)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=19, tiles='OpenStreetMap')

        # Marcadores
        for point in self.flight_log:
            folium.CircleMarker(
                location=[point['lat'], point['lon']],
                radius=3,
                color=HEX_COLORS[point['label']],
                fill=True,
                fill_opacity=0.7,
                popup=f"{point['label'].upper()} ({point['conf']:.0f}%)"
            ).add_to(m)
        
        # Mapa de Calor (Solo para problemas)
        heat_data = [[p['lat'], p['lon']] for p in self.flight_log if p['label'] != 'sanas']
        if heat_data: 
            HeatMap(heat_data, radius=15).add_to(m)
            print(f"🔥 Mapa de calor generado con {len(heat_data)} puntos críticos.")
        else:
            print("✅ No se detectaron enfermedades críticas para el mapa de calor.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/mission_report_{timestamp}.html"
        # Ruta absoluta para que la encuentres fácil
        abs_path = os.path.abspath(filename)
        
        m.save(filename)
        print(f"✅ REPORTE GENERADO EXITOSAMENTE:")
        print(f"👉 {abs_path}")

    def fly(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): 
            print("❌ No se puede abrir la cámara.")
            return

        print("🎥 VUELO INICIADO (Apunta a plantas/fotos). Presiona 'Q' para finalizar.")
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Corrección del bug de width/height
            height, width = frame.shape[:2]

            self.update_gps()

            # Filtro
            has_plants, ratio = self.check_vegetation_presence(frame)

            if has_plants:
                raw_label, raw_conf = self.infer(frame)
                display_label, display_conf = self.get_stabilized_prediction(raw_label, raw_conf)
                
                frame = self.draw_hud(frame, display_label, display_conf, is_scanning=False)

                # LOGGING: Guardar si la confianza es alta
                if raw_conf > CONFIDENCE_THRESHOLD:
                    self.flight_log.append({
                        'lat': self.current_lat, 'lon': self.current_lon,
                        'label': raw_label, 'conf': raw_conf * 100
                    })
            else:
                self.prediction_buffer.clear()
                frame = self.draw_hud(frame, None, 0, is_scanning=True)
                
                cv2.putText(frame, f"VEG: {ratio*100:.1f}% (Bajo)", (width-250, height-30), 
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (100, 100, 100), 2)

            cv2.imshow('Drone HUD - Plant Doctor', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.generate_map()

if __name__ == "__main__":
    drone = DroneBrain()
    drone.fly()