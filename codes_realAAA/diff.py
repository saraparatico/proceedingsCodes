import csv
import math
from math import sqrt

# Specifica il percorso del file CSV che desideri aprire
percorso_file = "./diffPost.csv"
valori = []
# Apre il file in modalità lettura
with open(percorso_file, newline='') as file_csv:
    # Legge il contenuto del file CSV utilizzando il modulo csv.reader
    lettore_csv = csv.reader(file_csv)

    for i, riga in enumerate(lettore_csv):
        if i == 0:
            print(riga)
            break

    next(lettore_csv)
    # Itera sulle righe del file CSV e stampa ogni riga
    for i, riga in enumerate(lettore_csv):
        print("at", i, ":", riga)
        # Itera sui valori all'interno di ogni riga e aggiungi alla lista dei valori
        for valore in riga:
            valori.append(float(valore))  # Converti il valore in float e aggiungi alla lista

    # Calcola la media dei valori
    media = sum(valori)

    print("La somma di tutte le differenze nel file CSV è:", media)

quadrato = media**2
divisione = quadrato/len(valori)
RMSE = sqrt(divisione)
print("RMSE:", RMSE)