import requests
from LLM import LLM_ans
from time import sleep
import pandas as pd

umbrales = pd.read_csv(r"Datos_extra/umbrales.csv")

while True:
    r = requests.get("http://192.168.100.2:5000/predict")
    pred = r.json()
    esp = pred["label"]
    prob = pred["confidence"]
    umbral = umbrales.loc[umbrales.Clase == esp, "Umbral"].iloc[0]
    if umbral <= prob:
        LLM_ans(esp)
    sleep(5)