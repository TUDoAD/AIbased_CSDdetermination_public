import numpy as np
import pandas as pd
import os
from PIL import Image
import openpyxl
import math 
import matplotlib.pyplot as plt
import datetime
import cv2
import time
import psutil
from ultralytics import YOLO

input_folder = "Input"
model = YOLO("Current/SModel8.pt")#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<MODEL
results_list = []

def rotating_calipers(points):
    def dist(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    n = len(points)
    if n == 0:
        return 0
    
    k = 1
    max_diameter = 0

    for i in range(n):
        while True:
            current_dist = dist(points[i], points[k])
            next_dist = dist(points[i], points[(k + 1) % n])
            if next_dist > current_dist:
                k = (k + 1) % n
            else:
                break
        max_diameter = max(max_diameter, dist(points[i], points[k]))
    
    return max_diameter

def calculate_feret_diameter(contour):
    # Berechnung der konvexen Hülle
    hull = cv2.convexHull(contour)
    hull_points = hull[:, 0, :]  # Entfernen unnötiger Dimensionen

    # Feret-Durchmesser mit Rotating Calipers berechnen
    return rotating_calipers(hull_points)

for file in os.listdir(input_folder):
    results = None 
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, file)
        image = Image.open(image_path)
        width, height = image.size

        # Startzeitpunkt der Verarbeitung
        start_time = time.time()

        # Startspeichermessung
        process = psutil.Process(os.getpid())
        memory_start = process.memory_info().rss

        # Durchführen der Objekterkennung
        results = model(image, save=True, conf=0.5)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<CONFIDENCE SCORE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Endzeitpunkt der Verarbeitung
        end_time = time.time()
        
        # Endspeichermessung
        memory_end = process.memory_info().rss
        
        # Rechenzeit berechnen
        processing_time = end_time - start_time
        
        # Speichernutzung berechnen
        memory_usage = (memory_end - memory_start) / (1024 * 1024)  # in MB

        if results is not None:
            for result in results:
                # Filtern der Ergebnisse nach Klasse 0
                class_ids = result.boxes.cls.cpu().numpy()
                mask_indices = np.where(class_ids == 0)[0]

                if result is not None and hasattr(result, 'masks') and result.masks is not None:
                    for idx in mask_indices:
                        contour = result.masks.xyn[idx]
                        contour = [(x * width, y * height) for x, y in contour] # Umwandeln normierter Koordinaten
                        contour = np.array(contour, dtype=np.float32)

                        # Berechnen des Feret-Durchmessers
                        feret_diameter = calculate_feret_diameter(contour)
                        adjusted_diameter = feret_diameter / 1.934346#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<CALIBRIERUNG FERET DURCHMESSER in PIXEL PRO MIKROMETER!!!!!!!!!!!!!!!!!!

                        # Berechnen der Fläche der Maske in Pixel
                        area = cv2.contourArea(contour)
                        adjusted_area = area / (1.934346)**2#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<CALIBRIERUNG FLÄCHE in PIXEL PRO MIKROMETER!!!!!!!!!!!!!!!!!!
                        # Hinzufügen der Ergebnisse zur Liste
                        results_list.append([file, feret_diameter, adjusted_diameter, area, adjusted_area, processing_time, memory_usage])
        else:
            # Falls keine Ergebnisse vorhanden sind, Dummy-Einträge hinzufügen
            results_list.append([file, None, None, None, None, processing_time, memory_usage])

# Speichern in einer Excel-Datei
df = pd.DataFrame(results_list, columns=['Bildname', 'Feret-Durchmesser', 'CrystalSizeMM', 'Fläche in Pixel', 'Area'])#MM STEHT FÜR MIKROMETER!!!!!!!!!!!!! NICHT MILIMETER

# Erstellen des Output-Ordners, falls er nicht existiert
output_folder = "Output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"Feret_Diameters_segmentation_{timestamp}.xlsx"
output_path = os.path.join(output_folder, filename)

# Speichern der Datei
df.to_excel(output_path, index=False)
print(f"Results saved to '{output_path}'")
