from ultralytics import YOLO
import numpy as np
import pandas as pd
import os
from PIL import Image
import openpyxl
import math 
import time
import psutil

def find_pos(item, lst): 
    return [i for (y, i) in zip(lst, range(len(lst))) if item == y] 

# Listen für die Ergebnisse und Bildnamen
CrystalSizeP = []
CrystalSizeMM = []
image_names = []
processing_times = []
memory_usages = []

# Umrechnungsfaktor von Pixel zu Millimeter
PixelToMM = 1934.346 / 1000

# Pfad zum Eingabeordner
input_folder = "Input"

# Model laden
model = YOLO("E:/Masterarbeit/Yolo/Model/Current/Model14.pt")

# Durchlaufen der Bilder
for file in os.listdir(input_folder):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):  
        image_path = os.path.join(input_folder, file)
        image = Image.open(image_path)
        
        # Startzeitpunkt der Verarbeitung
        start_time = time.time()

        # Startspeichermessung
        process = psutil.Process(os.getpid())
        memory_start = process.memory_info().rss

        # Durchführen der Objekterkennung
        results = model(image, save=True, conf=0.9)
        
        # Endzeitpunkt der Verarbeitung
        end_time = time.time()
        
        # Endspeichermessung
        memory_end = process.memory_info().rss
        
        # Rechenzeit und Speicherverbrauch berechnen und hinzufügen
        processing_time = end_time - start_time
        memory_usage = (memory_end - memory_start) / (1024 * 1024)  # in MB
        
        found_object = False

        for r in results:
            a = r.cpu().boxes.cls.numpy()
            b = r.cpu().boxes.xywh.numpy()

            # Wenn Diagonal
            pos = find_pos(1, a)
            for i in pos:
                size = math.sqrt((b[i,2]**2) + (b[i,3]**2))
                CrystalSizeP.append(size)
                image_names.append(file)  # Bildname hinzufügen
                processing_times.append(processing_time)
                memory_usages.append(memory_usage)
                found_object = True

            # Wenn Kristall
            pos = find_pos(0, a)
            for i in pos:
                if b[i,2] > b[i,3]:
                    CrystalSizeP.append(b[i,2])
                else:
                    CrystalSizeP.append(b[i,3])
                image_names.append(file)  # Bildname hinzufügen
                processing_times.append(processing_time)
                memory_usages.append(memory_usage)
                found_object = True
        
        # Falls keine Objekte gefunden wurden, Dummy-Werte hinzufügen
        if not found_object:
            CrystalSizeP.append(None)
            image_names.append(file)
            processing_times.append(processing_time)
            memory_usages.append(memory_usage)

# Umwandlung in Millimeter
CrystalSizeMM = [x / PixelToMM if x is not None else None for x in CrystalSizeP]     

# Erstellen eines DataFrames
df = pd.DataFrame({
    "Bildname": image_names,
    "CrystalSizeP": CrystalSizeP,
    "CrystalSizeMM": CrystalSizeMM,
    "ProcessingTime": processing_times,  # Rechenzeit hinzufügen
    "MemoryUsageMB": memory_usages  # Speicherverbrauch hinzufügen
})

# Erstellen des Ausgabeordners, falls nicht vorhanden
output_folder = "Output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Speichern des DataFrames als Excel-Datei
output_path = os.path.join(output_folder, "crystalsizes.xlsx")
df.to_excel(output_path, index=False)
