# Rilevamento della Stanchezza con Python, Mediapipe e OpenCV

Questo progetto utilizza **Mediapipe**, **OpenCV** e **PyAutoGUI** per rilevare segnali di stanchezza da webcam, come:

- Occhi chiusi prolungati
- Sbadigli
- Sguardo basso prolungato
- PERCLOS > 70% negli ultimi 30 secondi

In caso di rilevamento, il programma:
- Mostra un messaggio di allerta
- Cattura uno screenshot automatico
- Riproduce un suono di avviso (`beep.wav`)

---

## Requisiti

- Python 3.8+
- Webcam funzionante

### Librerie richieste
Installa le dipendenze con:

pip install opencv-python mediapipe pyautogui

### Avvio del programma

python tired_detection.py