"""
Transla{w}tion - Prototyp einer Gebärdensprache-Übersetzungssoftware
======================================================================

Plattform: Windows 10/11 und macOS, jede aktuelle Python-Version (auch
3.13+) - es wird bewusst KEIN mediapipe verwendet, da dessen alte
Landmark-API auf neuen Python-Versionen/Betriebssystemen unzuverlässig
ist bzw. inzwischen entfernt wurde.

Die Handerkennung basiert stattdessen ausschließlich auf OpenCV:
Hautfarben-Segmentierung (YCrCb) + Konturanalyse + "Convexity Defects",
um die Anzahl ausgestreckter Finger zu zählen. Das ist weniger präzise
als ein trainiertes ML-Modell, funktioniert aber überall identisch und
ohne Versionskonflikte - für einen Prototyp ein fairer Kompromiss.

Bedienung der Handerkennung:
- Die Hand in die eingeblendete gelbe Box halten.
- Einmalig 'C' drücken, während die flache Handfläche im kleinen
  Kalibrierungs-Quadrat in der Mitte der Box liegt (bei möglichst
  neutralem Hintergrund). Das kalibriert die Hautfarbe für Kamera und
  Lichtverhältnisse und verbessert die Erkennung deutlich.
- Erkannt wird die Anzahl ausgestreckter Finger (0-5); jede Zahl ist
  einem Wort zugeordnet (siehe GESTURE_WORDS weiter unten).

INSTALLATION (beide Plattformen, keine mediapipe-Abhängigkeit mehr)
---------------------------------------------------------------------
    pip install opencv-python numpy PyQt5 sounddevice soundfile \
                gtts SpeechRecognition deep-translator ollama

Hinweise:
- Für die KI-Satzkorrektur wird optional Ollama verwendet
  (https://ollama.com). Ist Ollama nicht installiert / nicht gestartet,
  läuft die App trotzdem weiter - es wird dann nur eine einfache
  Wiederholungs-Bereinigung statt einer KI-Korrektur durchgeführt.
  Falls gewünscht: `ollama pull gemma3:1b` und Ollama einmal starten.
- gTTS, SpeechRecognition (Google) und deep-translator benötigen eine
  Internetverbindung.
- macOS fragt beim ersten Start automatisch nach Kamera- und
  Mikrofonzugriff (Systemeinstellungen -> Datenschutz). Diese Anfragen
  müssen bestätigt werden, sonst bleibt das Kamerabild schwarz.
- Windows: Falls die Kamera von einer anderen Anwendung blockiert wird,
  kann das Öffnen fehlschlagen - die App zeigt dazu eine verständliche
  Fehlermeldung an, statt sich aufzuhängen.
"""
import sys
import os
import platform
import tempfile
import math
from datetime import datetime

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from gtts import gTTS
from deep_translator import GoogleTranslator

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTextEdit, QFrame, QMessageBox, QCheckBox, QComboBox,
                             QSizePolicy, QSpacerItem, QRadioButton, QButtonGroup, QProgressBar)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QUrl, QMarginsF
from PyQt5.QtGui import QImage, QPixmap, QFont, QFontDatabase, QResizeEvent, QMovie, QTextDocument, QPageSize, QPageLayout
from PyQt5.QtPrintSupport import QPrinter, QPrintPreviewDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

# Ollama ist optional: Läuft der Dienst nicht (oder ist er nicht
# installiert), soll die App trotzdem benutzbar bleiben.
try:
    from ollama import chat, ChatResponse
    OLLAMA_IMPORT_OK = True
except ImportError:
    OLLAMA_IMPORT_OK = False

# ---------------------------------------------------------------------------
# Gestenerkennung (reines OpenCV, ohne mediapipe)
# ---------------------------------------------------------------------------
# Erkannt wird nur die Anzahl ausgestreckter Finger (0 bis 5). Jede Zahl
# ist pro Sprache einem Wort zugeordnet. Weitere Wörter lassen sich nicht
# beliebig ergänzen (max. 6 Zustände: 0-5 Finger), aber die Zuordnung
# selbst ist hier zentral und leicht austauschbar.
GESTURE_WORDS = {
    "deutsch": {
        0: "Nein",
        1: "Ich",
        2: "Hallo",
        3: "Gut",
        4: "nicht",
        5: "Bitte",
    },
    "amerikanisch": {
        0: "no",
        1: "I",
        2: "hello",
        3: "good",
        4: "not",
        5: "please",
    },
}

def _dist(p1, p2):
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))

def count_fingers(contour):
    """Zählt ausgestreckte Finger einer Handkontur über Convexity Defects.

    Gibt (anzahl_finger, solidity) zurück. 'solidity' (Kontur-/Hüllflächen-
    Verhältnis) hilft, eine Faust (0 Finger, kompakte Form) von einem
    einzelnen ausgestreckten Finger (1, länglich) zu unterscheiden, da
    beide Fälle keine Convexity Defects erzeugen.
    """
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return 0, 1.0

    # WICHTIG: hull_indices NICHT sortieren!
    # convexityDefects erwartet die Indizes in der Reihenfolge des Convex Hulls
    # (entlang der Kontur). np.sort zerstört diese Reihenfolge und führt
    # unter manchen OpenCV-Versionen zu kaputten defects-Arrays.

    hull_points = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull_points)
    contour_area = cv2.contourArea(contour)
    solidity = float(contour_area) / hull_area if hull_area > 0 else 1.0

    defects = None
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        defects = None

    finger_gaps = 0
    if defects is not None and len(defects) > 0:
        for i in range(defects.shape[0]):
            # Robust gegen unterschiedliche Shapes: (N,1,4) oder (N,4)
            if defects.ndim == 3:
                defect = defects[i, 0]
            else:
                defect = defects[i]

            # Manche OpenCV-Versionen / kaputte Contours liefern Skalare
            if not hasattr(defect, '__len__') or len(defect) < 4:
                continue

            s, e, f, d = defect
            # Indizes müssen gültig sein
            if s < 0 or e < 0 or f < 0:
                continue
            if s >= len(contour) or e >= len(contour) or f >= len(contour):
                continue

            start = contour[s][0]
            end = contour[e][0]
            far = contour[f][0]
            a = _dist(end, start)
            b = _dist(far, start)
            c = _dist(end, far)
            if b * c == 0:
                continue
            cos_angle = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.acos(cos_angle)
            # Nur "spitze" Winkel (< ~86°) zwischen zwei ausgestreckten
            # Fingern zählen, und nur deutliche Einkerbungen (d ist in
            # OpenCV-Fixpunkt-Skalierung, daher der hohe Schwellenwert).
            if angle <= math.pi / 2.1 and d > 9000:
                finger_gaps += 1

    if finger_gaps > 0:
        fingers = min(finger_gaps + 1, 5)
    else:
        fingers = 0 if solidity > 0.88 else 1

    return fingers, solidity

# ---------------------------------------------------------------------------
# Hilfsfunktionen für Plattform-Kompatibilität
# ---------------------------------------------------------------------------

def get_default_font_family():
    """Wählt eine auf dem jeweiligen Betriebssystem garantiert vorhandene, gut lesbare
    Schriftart, statt fest 'Arial' zu erzwingen."""
    available = set(QFontDatabase().families())
    system = platform.system()
    if system == "Windows":
        preferred = ["Segoe UI", "Calibri", "Arial"]
    elif system == "Darwin":
        preferred = ["Helvetica Neue", "Helvetica", "Arial"]
    else:
        preferred = ["Arial", "DejaVu Sans", "Noto Sans"]
    for name in preferred:
        if name in available:
            return name
    return QFont().defaultFamily()

def open_camera(index=0):
    """Öffnet die Kamera mit dem für das jeweilige Betriebssystem passenden Backend.
    Ohne das richtige Backend bleibt das Kamerabild unter Windows und macOS
    häufig schwarz oder das Öffnen dauert sehr lange bzw. schlägt fehl.
    Versucht bei Fehlschlag alternative Backends und macht einen kurzen Warm-up."""
    system = platform.system()
    backends = []
    if system == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif system == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for backend in backends:
        try:
            cap = cv2.VideoCapture(index, backend)
        except Exception:
            continue
        if cap is not None and cap.isOpened():
            # Stabile Auflösung setzen (viele Webcams starten sonst mit 0x0 oder ungewöhnlichen Werten)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Kurzer Warm-up: ein paar Frames verwerfen, bis die Kamera liefert
            ret = False
            for _ in range(8):
                ret, _ = cap.read()
                if ret:
                    break
            if ret:
                return cap
            try:
                cap.release()
            except Exception:
                pass
    # Letzter Fallback ohne explizites Backend
    try:
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            return cap
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Audioaufnahme
# ---------------------------------------------------------------------------
class AudioRecorder(QThread):
    recording_finished_signal = pyqtSignal(str)
    processing_started_signal = pyqtSignal()
    processing_finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.recording = False
        self.sample_rate = 44100
        self.channels = 1
        self.recognizer = sr.Recognizer()
        self.current_language = "deutsch"
        # Temp-Verzeichnis des Betriebssystems statt Skriptverzeichnis:
        # Läuft die App aus einem geschützten Ordner (z. B. "Programme" unter
        # Windows oder einer .app unter macOS), darf dort oft nicht geschrieben
        # werden. tempfile.gettempdir() ist auf beiden Plattformen garantiert
        # beschreibbar.
        self.temp_dir = tempfile.gettempdir()
        self._frames = []
        self._stream = None

    def run(self):
        # Hält für dieses QThread-Objekt eine Qt-Ereignisschleife am Leben,
        # damit Signale sauber verarbeitet werden. Die eigentliche Aufnahme
        # läuft über einen sounddevice-Callback und blockiert diesen Thread nicht.
        self.exec_()

    def set_language(self, language):
        self.current_language = language

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Sounddevice-Status: {status}")
        self._frames.append(indata.copy())

    def toggle_recording(self):
        if not self.recording:
            self._frames = []
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='int16',
                    callback=self._audio_callback,
                )
                self._stream.start()
                self.recording = True
                print("Aufnahme gestartet")
            except Exception as e:
                self.error_signal.emit(f"Mikrofon konnte nicht gestartet werden: {e}")
        else:
            self.recording = False
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception as e:
                self.error_signal.emit(f"Fehler beim Beenden der Aufnahme: {e}")
            finally:
                self._stream = None

            if self._frames:
                self.processing_started_signal.emit()
                try:
                    recording = np.concatenate(self._frames, axis=0)
                    temp_file_path = os.path.join(self.temp_dir, "translawtion_temp_audio.wav")
                    sf.write(temp_file_path, recording, self.sample_rate)
                    self.process_audio(temp_file_path)
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung der Audioaufnahme: {e}")
                    self.recording_finished_signal.emit(f"Fehler bei der Verarbeitung der Audioaufnahme: {e}")
                finally:
                    self.processing_finished_signal.emit()
            self._frames = []

    def process_audio(self, audio_file):
        try:
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
                lang_code = "de-DE" if self.current_language == "deutsch" else "en-US"
                text = self.recognizer.recognize_google(audio_data, language=lang_code)
                self.recording_finished_signal.emit(text)
        except sr.UnknownValueError:
            msg = "Kein Sprachinhalt erkannt." if self.current_language == "deutsch" else "No speech detected."
            self.recording_finished_signal.emit(msg)
        except sr.RequestError as e:
            msg = f"Spracherkennungsdienst nicht erreichbar (Internetverbindung prüfen): {e}"
            self.recording_finished_signal.emit(msg)
        except Exception as e:
            print(f"Fehler bei der Spracherkennung: {e}")
            self.recording_finished_signal.emit(f"Fehler bei der Spracherkennung: {e}")

    def stop(self):
        if self.recording:
            self.toggle_recording()
        self.quit()
        self.wait()

# ---------------------------------------------------------------------------
# Video- und Gestenerkennungs-Thread (reines OpenCV)
# ---------------------------------------------------------------------------
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QPixmap)
    recording_finished_signal = pyqtSignal(list)
    debug_info_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    calibration_done_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.recording = False
        self.recognized_words = []
        self.last_word = None
        self.sign_language = "deutsch"
        self.show_keypoints = False
        self.current_word = None
        self.cap = open_camera(0)
        self._failed_reads = 0

        # Standard-Hautfarbbereich (YCrCb) - grobe Voreinstellung, die
        # durch Kalibrierung (Taste 'C') pro Nutzer/Lichtverhältnis
        # deutlich verbessert werden kann.
        self.lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_skin = np.array([255, 178, 133], dtype=np.uint8)
        self.calibrated = False
        self._calibration_requested = False

    def set_sign_language(self, language):
        self.sign_language = language

    def set_show_keypoints(self, show):
        self.show_keypoints = show

    def request_calibration(self):
        self._calibration_requested = True

    def run(self):
        if self.cap is None or not self.cap.isOpened():
            self.error_signal.emit(
                "Kamera konnte nicht geöffnet werden. Bitte prüfen Sie, ob eine "
                "Kamera angeschlossen ist, keine andere Anwendung sie blockiert "
                "und der Kamerazugriff für diese App erlaubt ist."
            )
            return

        while self._run_flag:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self._failed_reads += 1
                    if self._failed_reads > 100:
                        self.error_signal.emit(
                            "Die Kamera liefert kein Bild mehr. Bitte Anwendung neu starten."
                        )
                        break
                    continue
                self._failed_reads = 0

                # Leere oder ungültige Frames überspringen
                if frame is None or frame.size == 0:
                    continue

                frame = cv2.flip(frame, 1)  # Spiegelbild wirkt für Nutzer intuitiver
                h, w = frame.shape[:2]
                if h < 10 or w < 10:
                    continue

                # Erkennungsbox: rechter Bildbereich, dort soll die Hand
                # platziert werden.
                x1, y1 = int(w * 0.55), int(h * 0.08)
                x2, y2 = int(w * 0.97), int(h * 0.88)
                roi = frame[y1:y2, x1:x2]

                current_word = None
                detected_fingers = None

                if roi.size > 0:
                    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

                    if self._calibration_requested:
                        self._calibration_requested = False
                        rh, rw = ycrcb.shape[:2]
                        sx1, sx2 = int(rw * 0.35), int(rw * 0.65)
                        sy1, sy2 = int(rh * 0.35), int(rh * 0.65)
                        sample = ycrcb[sy1:sy2, sx1:sx2].reshape(-1, 3)
                        if sample.size > 0:
                            mean = sample.mean(axis=0)
                            std = sample.std(axis=0) + 5  # Mindeststreuung gegen zu enge Grenzen
                            self.lower_skin = np.clip(mean - 2.3 * std, 0, 255).astype(np.uint8)
                            self.upper_skin = np.clip(mean + 2.3 * std, 0, 255).astype(np.uint8)
                            self.calibrated = True
                            self.calibration_done_signal.emit(True)

                    mask = cv2.inRange(ycrcb, self.lower_skin, self.upper_skin)
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        min_area = roi.shape[0] * roi.shape[1] * 0.04
                        if cv2.contourArea(largest) > min_area:
                            fingers, solidity = count_fingers(largest)
                            detected_fingers = fingers

                            if self.recording:
                                current_word = GESTURE_WORDS.get(self.sign_language, {}).get(fingers)

                            if self.show_keypoints:
                                offset = np.array([[x1, y1]])
                                shifted = largest + offset
                                cv2.drawContours(frame, [shifted], -1, (0, 220, 0), 2)
                                hull_pts = cv2.convexHull(largest) + offset
                                cv2.drawContours(frame, [hull_pts], -1, (255, 0, 220), 1)

                # Erkennungsbox und Kalibrierungs-Quadrat einzeichnen
                box_color = (60, 200, 60) if self.calibrated else (0, 200, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cw, ch = x2 - x1, y2 - y1
                csx1, csy1 = x1 + int(cw * 0.35), y1 + int(ch * 0.35)
                csx2, csy2 = x1 + int(cw * 0.65), y1 + int(ch * 0.65)
                if not self.calibrated:
                    cv2.rectangle(frame, (csx1, csy1), (csx2, csy2), (0, 165, 255), 1)

                if self.show_keypoints:
                    info = f"Finger: {detected_fingers if detected_fingers is not None else '-'}"
                    if current_word:
                        info += f" -> {current_word}"
                    if not self.calibrated:
                        info += "  |  Taste 'C' zum Kalibrieren"
                    self.debug_info_signal.emit(info)

                self.current_word = current_word
                if self.recording and current_word and current_word != self.last_word:
                    self.recognized_words.append(current_word)
                    self.last_word = current_word

                # Sichere Konvertierung zu QImage/QPixmap
                # tobytes() + QImage-Konstruktor kopiert die Daten zuverlässig
                # (vermeidet Probleme mit numpy-Puffer-Lebensdauer auf manchen Systemen)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                h2, w2, ch2 = rgb.shape
                bytes_per_line = ch2 * w2
                q_img = QImage(rgb.tobytes(), w2, h2, bytes_per_line, QImage.Format_RGB888)
                if q_img.isNull():
                    continue
                pixmap = QPixmap.fromImage(q_img)
                if not pixmap.isNull():
                    self.change_pixmap_signal.emit(pixmap)

            except Exception as e:
                # Thread darf nicht abstürzen – Fehler loggen und weiterlaufen
                print(f"VideoThread Fehler (wird ignoriert): {e}")
                import traceback
                traceback.print_exc()
                continue

    def stop(self):
        self._run_flag = False
        self.wait()
        if self.cap is not None:
            self.cap.release()

    def toggle_recording(self):
        self.recording = not self.recording
        if not self.recording:
            if self.recognized_words:
                self.recording_finished_signal.emit(self.recognized_words.copy())
            self.recognized_words = []
            self.last_word = None

# ---------------------------------------------------------------------------
# Stylesheet (rein optisch - "Aufhübschen" der Oberfläche)
# ---------------------------------------------------------------------------
APP_STYLESHEET = """
QMainWindow {
    background-color: #f4f6f9;
}
QFrame#panel {
    background-color: #ffffff;
    border: 1px solid #e1e5eb;
    border-radius: 10px;
}
QLabel#headerTitle {
    font-size: 22px;
    font-weight: 600;
    color: #1f2d3d;
}
QLabel#headerSubtitle {
    font-size: 13px;
    color: #6b7785;
}
QRadioButton {
    padding: 6px 10px;
    border-radius: 6px;
    color: #1f2d3d;
}
QRadioButton::indicator {
    width: 14px;
    height: 14px;
}
QPushButton {
    background-color: #2f6fed;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #255ac7;
}
QPushButton:pressed {
    background-color: #1c4699;
}
QPushButton:disabled {
    background-color: #c6ccd6;
    color: #8891a0;
}
QPushButton#secondaryButton {
    background-color: #eef1f6;
    color: #1f2d3d;
}
QPushButton#secondaryButton:hover {
    background-color: #dfe4ec;
}
QComboBox, QTextEdit {
    border: 1px solid #d7dce3;
    border-radius: 6px;
    padding: 4px;
    background-color: white;
}
QCheckBox {
    color: #1f2d3d;
}
QProgressBar {
    border: 1px solid #d7dce3;
    border-radius: 6px;
    background-color: #eef1f6;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #2f6fed;
    border-radius: 6px;
}
QLabel#controlHint {
    color: #2f6fed;
    font-weight: 600;
}
"""

# ---------------------------------------------------------------------------
# Hauptfenster
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transla{w}tion")
        self.setGeometry(100, 100, 1280, 760)
        self.setMinimumSize(1000, 620)

        self.player = QMediaPlayer()
        self.player.error.connect(self._on_player_error)

        self.default_font_family = get_default_font_family()
        self.base_font_size = 12
        self.font = QFont()
        self.font.setPointSize(self.base_font_size)
        self.font.setFamily(self.default_font_family)

        self.translator = GoogleTranslator(source='auto')

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # --- Kopfbereich ---------------------------------------------------
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        self.logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            if not logo_pixmap.isNull():
                self.logo_label.setPixmap(logo_pixmap.scaledToHeight(56, Qt.SmoothTransformation))
                header_layout.addWidget(self.logo_label)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(0)
        self.header_title = QLabel("Transla{w}tion")
        self.header_title.setObjectName("headerTitle")
        self.header_subtitle = QLabel("Prototyp: Echtzeit-Übersetzung von Gebärdensprache")
        self.header_subtitle.setObjectName("headerSubtitle")
        title_layout.addWidget(self.header_title)
        title_layout.addWidget(self.header_subtitle)
        header_layout.addLayout(title_layout)
        header_layout.addStretch(1)
        main_layout.addLayout(header_layout)

        user_selection_layout = QHBoxLayout()
        self.user_group = QButtonGroup()
        self.user_a_radio = QRadioButton("Person A (Gebärdensprache)")
        self.user_a_radio.setChecked(True)
        self.user_b_radio = QRadioButton("Person B (Sprache)")
        self.user_group.addButton(self.user_a_radio)
        self.user_group.addButton(self.user_b_radio)
        self.user_group.buttonClicked.connect(self.switch_user_mode)

        user_selection_layout.addWidget(self.user_a_radio)
        user_selection_layout.addWidget(self.user_b_radio)
        user_selection_layout.addStretch(1)
        main_layout.addLayout(user_selection_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # --- Linkes Panel (Video) ------------------------------------------
        left_panel = QFrame()
        left_panel.setObjectName("panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #10141c;
            border: 1px solid #ccc;
            border-radius: 8px;
        """)
        self.video_label.setMinimumSize(640, 440)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setToolTip("Hier wird das Kamerabild angezeigt")
        left_layout.addWidget(self.video_label)

        self.debug_info_label = QLabel()
        self.debug_info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.debug_info_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 150);
            color: white;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 14px;
        """)
        self.debug_info_label.setVisible(False)
        left_layout.addWidget(self.debug_info_label)

        self.keypoints_checkbox = QCheckBox("Kontur & Erkennung anzeigen (Debug)")
        self.keypoints_checkbox.setFont(self.font)
        self.keypoints_checkbox.setToolTip("Zeigt die erkannte Handkontur, Fingeranzahl und den aktuellen Kalibrierungsstatus")
        self.keypoints_checkbox.stateChanged.connect(self.toggle_keypoints)
        left_layout.addWidget(self.keypoints_checkbox)

        self.control_hint = QLabel("Hand in die Box halten. 'C' kalibriert, 'Q' startet/beendet die Aufnahme")
        self.control_hint.setObjectName("controlHint")
        self.control_hint.setAlignment(Qt.AlignCenter)
        self.control_hint.setFont(self.font)
        self.control_hint.setToolTip("Tastaturkürzel: C = Hautfarbe kalibrieren, Q = Aufnahme starten/beenden")
        left_layout.addWidget(self.control_hint)

        language_selection_layout = QHBoxLayout()

        self.speaker_a_layout = QVBoxLayout()
        self.speaker_a_label = QLabel("Sprecher A:")
        self.speaker_a_label.setFont(self.font)
        self.speaker_a_layout.addWidget(self.speaker_a_label)

        self.language_combo_a = QComboBox()
        self.language_combo_a.addItem("Deutsch")
        self.language_combo_a.addItem("American Sign Language")
        self.language_combo_a.currentIndexChanged.connect(self.change_sign_language)
        self.language_combo_a.setFont(self.font)
        self.language_combo_a.setToolTip("Wählen Sie die Gebärdensprache für Sprecher A aus")
        self.speaker_a_layout.addWidget(self.language_combo_a)

        language_selection_layout.addLayout(self.speaker_a_layout)

        self.speaker_b_layout = QVBoxLayout()
        self.speaker_b_label = QLabel("Sprecher B:")
        self.speaker_b_label.setFont(self.font)
        self.speaker_b_layout.addWidget(self.speaker_b_label)

        self.language_combo_b = QComboBox()
        self.language_combo_b.addItem("Deutsch")
        self.language_combo_b.addItem("Englisch")
        self.language_combo_b.currentIndexChanged.connect(self.change_spoken_language)
        self.language_combo_b.setFont(self.font)
        self.language_combo_b.setToolTip("Wählen Sie die Sprache für Sprecher B aus")
        self.speaker_b_layout.addWidget(self.language_combo_b)

        language_selection_layout.addLayout(self.speaker_b_layout)
        left_layout.addLayout(language_selection_layout)
        content_layout.addWidget(left_panel, 2)

        # --- Rechtes Panel (Chat) -------------------------------------------
        right_panel = QFrame()
        right_panel.setObjectName("panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setFont(self.font)
        self.chat_area.setToolTip("Hier wird der Chatverlauf angezeigt")
        right_layout.addWidget(self.chat_area, 1)

        self.progress_label = QLabel("Audio wird verarbeitet...")
        self.progress_label.setVisible(False)
        self.progress_label.setFont(self.font)
        right_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)
        right_layout.addWidget(self.progress_bar)

        self.animation_label = QLabel()
        self.animation_label.setVisible(False)
        self.animation_label.setAlignment(Qt.AlignCenter)
        gif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processing.gif")
        self.processing_animation = None
        if os.path.exists(gif_path):
            self.processing_animation = QMovie(gif_path)
            self.animation_label.setMovie(self.processing_animation)
        right_layout.addWidget(self.animation_label)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        self.minus_button = QPushButton("A-")
        self.minus_button.setObjectName("secondaryButton")
        self.minus_button.setFixedSize(40, 32)
        self.minus_button.clicked.connect(self.decrease_font_size)
        self.minus_button.setToolTip("Text verkleinern (Strg + -)")
        button_layout.addWidget(self.minus_button)

        self.plus_button = QPushButton("A+")
        self.plus_button.setObjectName("secondaryButton")
        self.plus_button.setFixedSize(40, 32)
        self.plus_button.clicked.connect(self.increase_font_size)
        self.plus_button.setToolTip("Text vergrößern (Strg + +)")
        button_layout.addWidget(self.plus_button)

        button_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.accept_button = QPushButton("Akzeptieren")
        self.accept_button.clicked.connect(self.accept_result)
        self.accept_button.setEnabled(False)
        self.accept_button.setFont(self.font)
        self.accept_button.setToolTip("Akzeptieren Sie den vorgeschlagenen Text")
        button_layout.addWidget(self.accept_button)

        self.reject_button = QPushButton("Ablehnen")
        self.reject_button.setObjectName("secondaryButton")
        self.reject_button.clicked.connect(self.reject_result)
        self.reject_button.setEnabled(False)
        self.reject_button.setFont(self.font)
        self.reject_button.setToolTip("Lehnen Sie den vorgeschlagenen Text ab")
        button_layout.addWidget(self.reject_button)

        self.clear_button = QPushButton("Chat löschen")
        self.clear_button.setObjectName("secondaryButton")
        self.clear_button.clicked.connect(self.clear_chat)
        self.clear_button.setFont(self.font)
        self.clear_button.setToolTip("Löscht den gesamten Chatverlauf")
        button_layout.addWidget(self.clear_button)

        self.print_button = QPushButton("Drucken")
        self.print_button.setObjectName("secondaryButton")
        self.print_button.clicked.connect(self.print_protocol)
        self.print_button.setFont(self.font)
        self.print_button.setToolTip("Druckt das Protokoll")
        button_layout.addWidget(self.print_button)

        right_layout.addLayout(button_layout)

        self.speech_checkbox = QCheckBox("Sprachausgabe aktivieren")
        self.speech_checkbox.setChecked(True)
        self.speech_checkbox.setFont(self.font)
        self.speech_checkbox.setToolTip("Aktivieren/Deaktivieren der Sprachausgabe")
        right_layout.addWidget(self.speech_checkbox)

        self.target_language_combo = QComboBox()
        self.target_language_combo.addItem("Deutsch")
        self.target_language_combo.addItem("Englisch")
        self.target_language_combo.addItem("Französisch")
        self.target_language_combo.addItem("Spanisch")
        self.target_language_combo.setFont(self.font)
        self.target_language_combo.setToolTip("Wählen Sie die Zielsprache für die Übersetzung aus")
        right_layout.addWidget(self.target_language_combo)

        self.translate_button = QPushButton("Übersetzen")
        self.translate_button.setObjectName("secondaryButton")
        self.translate_button.clicked.connect(self.translate_text)
        self.translate_button.setFont(self.font)
        self.translate_button.setToolTip("Übersetzt den aktuellen Text in die Zielsprache")
        right_layout.addWidget(self.translate_button)

        content_layout.addWidget(right_panel, 1)
        main_layout.addLayout(content_layout)

        # --- Hintergrund-Threads --------------------------------------------
        self.audio_recorder = AudioRecorder()
        self.audio_recorder.recording_finished_signal.connect(self.process_audio_recording)
        self.audio_recorder.processing_started_signal.connect(self.show_processing_indicator)
        self.audio_recorder.processing_finished_signal.connect(self.hide_processing_indicator)
        self.audio_recorder.error_signal.connect(self.show_error_message)
        self.audio_recorder.start()

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.recording_finished_signal.connect(self.process_recording)
        self.thread.debug_info_signal.connect(self.update_debug_info)
        self.thread.error_signal.connect(self.show_error_message)
        self.thread.calibration_done_signal.connect(self.on_calibration_done)
        self.thread.start()

        self.current_language = "deutsch"
        self.current_user = "A"
        self.current_translation = None
        self.current_llm_output = None
        self.last_spoken_text = None

        self.update_gui_language()
        self.add_chat_message(
            "System",
            "Willkommen! Halten Sie Ihre Hand in die gelbe Box im Kamerabild und "
            "drücken Sie 'C', um die Handerkennung auf Ihre Hautfarbe und Beleuchtung "
            "zu kalibrieren. Wählen Sie danach Person A oder B aus und drücken Sie "
            "'Q', um eine Aufnahme zu starten."
        )

        if not OLLAMA_IMPORT_OK:
            self.add_chat_message(
                "System",
                "Hinweis: Ollama wurde nicht gefunden. Die KI-Satzkorrektur ist "
                "deaktiviert, erkannte Wörter werden stattdessen nur bereinigt "
                "aneinandergereiht."
            )

    # -- Fehleranzeige --------------------------------------------------
    def show_error_message(self, message):
        self.add_chat_message("System", message)
        QMessageBox.warning(self, "Transla{w}tion", message)

    def on_calibration_done(self, ok):
        if ok:
            msg = ("Handerkennung kalibriert." if self.current_language == "deutsch"
                   else "Hand detection calibrated.")
            self.add_chat_message("System", msg)

    def _on_player_error(self, error):
        if error != QMediaPlayer.NoError:
            print(f"Wiedergabefehler: {self.player.errorString()}")

    def toggle_keypoints(self, state):
        self.thread.set_show_keypoints(state == Qt.Checked)
        self.debug_info_label.setVisible(state == Qt.Checked)

        if state == Qt.Checked:
            self.add_chat_message("System", "Debug-Ansicht aktiviert (Kontur, Fingeranzahl, Kalibrierungsstatus)")
        else:
            self.add_chat_message("System", "Debug-Ansicht ausgeblendet")
            self.debug_info_label.clear()

    def update_debug_info(self, info):
        self.debug_info_label.setText(info)

    def update_gui_language(self):
        if self.current_user == "A":
            language = "deutsch" if self.language_combo_a.currentIndex() == 0 else "amerikanisch"
        else:
            language = "deutsch" if self.language_combo_b.currentIndex() == 0 else "englisch"

        self.current_language = language
        self.update_ui_language()

    def show_processing_indicator(self):
        self.progress_label.setVisible(True)
        self.progress_bar.setVisible(True)
        if self.processing_animation is not None:
            self.animation_label.setVisible(True)
            self.processing_animation.start()

        if self.current_language == "deutsch":
            self.progress_label.setText("Audio wird verarbeitet...")
        else:
            self.progress_label.setText("Processing audio...")

    def hide_processing_indicator(self):
        self.progress_label.setVisible(False)
        self.progress_bar.setVisible(False)
        if self.processing_animation is not None:
            self.animation_label.setVisible(False)
            self.processing_animation.stop()

    def switch_user_mode(self, button):
        self.current_user = "A" if button == self.user_a_radio else "B"
        self.update_gui_language()

        if self.current_user == "A":
            self.control_hint.setText("Hand in die Box halten, dann 'Q' für Gebärdenaufnahme starten/beenden")
            self.add_chat_message("System", "Modus: Person A (Gebärdensprache). Drücken Sie 'Q' für Gebärdenaufnahme.")
        else:
            self.control_hint.setText("Drücken Sie 'Q', um Sprachaufnahme zu starten/beenden")
            self.add_chat_message("System", "Modus: Person B (Sprache). Drücken Sie 'Q' für Sprachaufnahme.")

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.adjust_font_size()

    def adjust_font_size(self):
        base_size = max(8, min(16, int(self.height() / 50)))
        self.base_font_size = base_size
        self.update_font_sizes()

    def update_font_sizes(self):
        self.font.setPointSize(self.base_font_size)

        self.control_hint.setFont(self.font)
        self.language_combo_a.setFont(self.font)
        self.language_combo_b.setFont(self.font)
        self.chat_area.setFont(self.font)
        self.accept_button.setFont(self.font)
        self.reject_button.setFont(self.font)
        self.clear_button.setFont(self.font)
        self.speech_checkbox.setFont(self.font)
        self.minus_button.setFont(self.font)
        self.plus_button.setFont(self.font)
        self.user_a_radio.setFont(self.font)
        self.user_b_radio.setFont(self.font)
        self.target_language_combo.setFont(self.font)
        self.translate_button.setFont(self.font)
        self.progress_label.setFont(self.font)
        self.print_button.setFont(self.font)
        self.keypoints_checkbox.setFont(self.font)
        self.speaker_a_label.setFont(self.font)
        self.speaker_b_label.setFont(self.font)
        self.debug_info_label.setFont(self.font)

        chat_font = QFont(self.default_font_family, self.base_font_size + 2)
        self.chat_area.setFont(chat_font)

    def increase_font_size(self):
        self.base_font_size = min(24, self.base_font_size + 2)
        self.update_font_sizes()

    def decrease_font_size(self):
        self.base_font_size = max(8, self.base_font_size - 2)
        self.update_font_sizes()

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                self.increase_font_size()
            elif event.key() == Qt.Key_Minus:
                self.decrease_font_size()
            return

        if event.key() == Qt.Key_C:
            self.thread.request_calibration()
            return

        if event.key() == Qt.Key_Q:
            if self.current_user == "A":
                self.thread.toggle_recording()
                if self.thread.recording:
                    if self.current_language == "deutsch":
                        self.control_hint.setText("Gebärdenaufnahme läuft... Drücken Sie 'Q' zum Beenden")
                        self.add_chat_message("System", "Gebärdenaufnahme gestartet...")
                    else:
                        self.control_hint.setText("Sign language recording in progress... Press 'Q' to stop")
                        self.add_chat_message("System", "Sign language recording started...")
                    self.control_hint.setStyleSheet("color: #d64545; font-weight: bold;")
                else:
                    if self.current_language == "deutsch":
                        self.control_hint.setText("Hand in die Box halten, dann 'Q' für Gebärdenaufnahme starten")
                    else:
                        self.control_hint.setText("Hold your hand in the box, then press 'Q' to start recording")
                    self.control_hint.setStyleSheet("")
            else:
                if not self.audio_recorder.recording:
                    self.audio_recorder.toggle_recording()
                    if self.current_language == "deutsch":
                        self.control_hint.setText("Sprachaufnahme läuft... Drücken Sie 'Q' zum Beenden")
                        self.add_chat_message("System", "Sprachaufnahme gestartet...")
                    else:
                        self.control_hint.setText("Voice recording in progress... Press 'Q' to stop")
                        self.add_chat_message("System", "Voice recording started...")
                    self.control_hint.setStyleSheet("color: #d64545; font-weight: bold;")
                else:
                    self.audio_recorder.toggle_recording()
                    if self.current_language == "deutsch":
                        self.control_hint.setText("Drücken Sie 'Q', um Sprachaufnahme zu starten")
                    else:
                        self.control_hint.setText("Press 'Q' to start voice recording")
                    self.control_hint.setStyleSheet("")

    def process_audio_recording(self, text):
        if self.current_user == "B" and text:
            self.add_chat_message("Sie", text)
            self.current_llm_output = text
            self.accept_button.setEnabled(True)
            self.reject_button.setEnabled(True)

    def change_sign_language(self):
        language = "deutsch" if self.language_combo_a.currentIndex() == 0 else "amerikanisch"
        self.thread.set_sign_language(language)

        if self.current_user == "A":
            self.update_gui_language()

        if language == "deutsch":
            self.add_chat_message("System", "Gebärdensprache geändert auf Deutsch")
        else:
            self.add_chat_message("System", "Sign language changed to American Sign Language")

    def change_spoken_language(self):
        language = "deutsch" if self.language_combo_b.currentIndex() == 0 else "englisch"
        self.audio_recorder.set_language(language)

        if self.current_user == "B":
            self.update_gui_language()

        if language == "deutsch":
            self.add_chat_message("System", "Sprache für Sprecher B geändert auf Deutsch")
        else:
            self.add_chat_message("System", "Language for speaker B changed to English")

    def update_ui_language(self):
        if self.current_language == "deutsch":
            if self.current_user == "A":
                self.control_hint.setText("Hand in die Box halten, dann 'Q' für Gebärdenaufnahme starten/beenden")
            else:
                self.control_hint.setText("Drücken Sie 'Q', um Sprachaufnahme zu starten/beenden")
            self.accept_button.setText("Akzeptieren")
            self.reject_button.setText("Ablehnen")
            self.clear_button.setText("Chat löschen")
            self.speech_checkbox.setText("Sprachausgabe aktivieren")
            self.minus_button.setToolTip("Text verkleinern (Strg + -)")
            self.plus_button.setToolTip("Text vergrößern (Strg + +)")
            self.user_a_radio.setText("Person A (Gebärdensprache)")
            self.user_b_radio.setText("Person B (Sprache)")
            self.translate_button.setText("Übersetzen")
            self.translate_button.setToolTip("Übersetzt den aktuellen Text in die Zielsprache")
            self.progress_label.setText("Audio wird verarbeitet...")
            self.print_button.setText("Drucken")
            self.print_button.setToolTip("Druckt das Protokoll")
            self.language_combo_a.setItemText(0, "Deutsch")
            self.language_combo_a.setItemText(1, "American Sign Language")
            self.language_combo_b.setItemText(0, "Deutsch")
            self.language_combo_b.setItemText(1, "Englisch")
            self.target_language_combo.setItemText(0, "Deutsch")
            self.target_language_combo.setItemText(1, "Englisch")
            self.target_language_combo.setItemText(2, "Französisch")
            self.target_language_combo.setItemText(3, "Spanisch")
            self.keypoints_checkbox.setText("Kontur & Erkennung anzeigen (Debug)")
            self.keypoints_checkbox.setToolTip("Zeigt die erkannte Handkontur, Fingeranzahl und den aktuellen Kalibrierungsstatus")
            self.header_subtitle.setText("Prototyp: Echtzeit-Übersetzung von Gebärdensprache")
        else:
            if self.current_user == "A":
                self.control_hint.setText("Hold your hand in the box, then press 'Q' to start/stop recording")
            else:
                self.control_hint.setText("Press 'Q' to start/stop voice recording")
            self.accept_button.setText("Accept")
            self.reject_button.setText("Reject")
            self.clear_button.setText("Clear Chat")
            self.speech_checkbox.setText("Enable Speech Output")
            self.minus_button.setToolTip("Decrease text size (Ctrl + -)")
            self.plus_button.setToolTip("Increase text size (Ctrl + +)")
            self.user_a_radio.setText("Person A (Sign Language)")
            self.user_b_radio.setText("Person B (Voice)")
            self.translate_button.setText("Translate")
            self.translate_button.setToolTip("Translate the current text to target language")
            self.progress_label.setText("Processing audio...")
            self.print_button.setText("Print")
            self.print_button.setToolTip("Print the protocol")
            self.language_combo_a.setItemText(0, "German")
            self.language_combo_a.setItemText(1, "American Sign Language")
            self.language_combo_b.setItemText(0, "German")
            self.language_combo_b.setItemText(1, "English")
            self.target_language_combo.setItemText(0, "German")
            self.target_language_combo.setItemText(1, "English")
            self.target_language_combo.setItemText(2, "French")
            self.target_language_combo.setItemText(3, "Spanish")
            self.keypoints_checkbox.setText("Show contour & detection (Debug)")
            self.keypoints_checkbox.setToolTip("Shows the detected hand contour, finger count and calibration status")
            self.header_subtitle.setText("Prototype: Real-time sign language translation")

    def update_image(self, pixmap):
        if pixmap is None or pixmap.isNull():
            return
        # Aktuelle Label-Größe verwenden; Fallback falls noch nicht layoutet
        lw = max(self.video_label.width(), 640)
        lh = max(self.video_label.height(), 440)
        scaled = pixmap.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def process_recording(self, words):
        if self.current_user == "A" and words:
            output = " ".join(words)
            self.add_chat_message("Sie", output)

            llm_out = self.process_with_ollama(output)
            self.current_llm_output = llm_out
            if self.current_language == "deutsch":
                self.add_chat_message("System (Vorschlag)", llm_out)
            else:
                self.add_chat_message("System (Suggestion)", llm_out)

            self.accept_button.setEnabled(True)
            self.reject_button.setEnabled(True)

    def add_chat_message(self, sender, message):
        timestamp = datetime.now().strftime("%H:%M")
        if sender in ("Sie", "You"):
            sender = f"Person {self.current_user}"
        self.chat_area.append(f"<b>{timestamp} - {sender}:</b> {message}<br>")
        self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())

    def accept_result(self):
        if self.current_llm_output:
            if self.current_language == "deutsch":
                self.add_chat_message("System", f"Akzeptiert: {self.current_llm_output}")
            else:
                self.add_chat_message("System", f"Accepted: {self.current_llm_output}")

            if self.speech_checkbox.isChecked():
                self.speak_text(self.current_llm_output)

            self.current_llm_output = None
            self.accept_button.setEnabled(False)
            self.reject_button.setEnabled(False)

    def reject_result(self):
        if self.current_llm_output:
            if self.current_language == "deutsch":
                self.add_chat_message("System", f"Abgelehnt: {self.current_llm_output}")
            else:
                self.add_chat_message("System", f"Rejected: {self.current_llm_output}")
            self.current_llm_output = None
            self.accept_button.setEnabled(False)
            self.reject_button.setEnabled(False)

    def clear_chat(self):
        self.chat_area.clear()
        if self.current_language == "deutsch":
            self.add_chat_message("System", "Chat wurde gelöscht. Wählen Sie Person A oder B aus und drücken Sie 'Q', um eine Aufnahme zu starten.")
        else:
            self.add_chat_message("System", "Chat cleared. Select Person A or B and press 'Q' to start recording.")

    def speak_text(self, text):
        try:
            language = 'de' if self.current_language == "deutsch" else 'en'
            temp_file_path = os.path.join(
                tempfile.gettempdir(),
                f"translawtion_speech_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp3"
            )
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_file_path)
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_file_path)))
            self.player.play()
            QTimer.singleShot(6000, lambda: os.path.exists(temp_file_path) and os.remove(temp_file_path))
        except Exception as e:
            if self.current_language == "deutsch":
                self.add_chat_message("System", f"Fehler bei der Sprachausgabe: {str(e)}")
            else:
                self.add_chat_message("System", f"Error in speech output: {str(e)}")

    def translate_text(self):
        if self.current_llm_output:
            try:
                target_lang = self.target_language_combo.currentText().lower()
                lang_map = {
                    "deutsch": "de", "german": "de",
                    "englisch": "en", "english": "en",
                    "französisch": "fr", "french": "fr",
                    "spanisch": "es", "spanish": "es",
                }
                target_lang_code = lang_map.get(target_lang, "en")

                translation = GoogleTranslator(source='auto', target=target_lang_code).translate(self.current_llm_output)
                self.current_translation = translation

                if self.current_language == "deutsch":
                    self.add_chat_message("System (Übersetzung)", f"Übersetzung ({target_lang}): {self.current_translation}")
                else:
                    self.add_chat_message("System (Translation)", f"Translation ({target_lang}): {self.current_translation}")

                if self.speech_checkbox.isChecked():
                    self.speak_translation(self.current_translation, target_lang_code)
            except Exception as e:
                if self.current_language == "deutsch":
                    self.add_chat_message("System", f"Fehler bei der Übersetzung: {str(e)}")
                else:
                    self.add_chat_message("System", f"Translation error: {str(e)}")

    def speak_translation(self, text, language_code):
        try:
            temp_file_path = os.path.join(
                tempfile.gettempdir(),
                f"translawtion_translation_speech_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp3"
            )
            tts = gTTS(text=text, lang=language_code, slow=False)
            tts.save(temp_file_path)
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_file_path)))
            self.player.play()
            QTimer.singleShot(6000, lambda: os.path.exists(temp_file_path) and os.remove(temp_file_path))
        except Exception as e:
            if self.current_language == "deutsch":
                self.add_chat_message("System", f"Fehler bei der Sprachausgabe der Übersetzung: {str(e)}")
            else:
                self.add_chat_message("System", f"Error in translation speech output: {str(e)}")

    def _naive_sentence_cleanup(self, input_string: str) -> str:
        """Einfacher Ersatz für die KI-Korrektur, falls Ollama nicht verfügbar
        ist: entfernt direkt aufeinanderfolgende Wortwiederholungen."""
        words = input_string.split()
        cleaned = []
        for w in words:
            if not cleaned or cleaned[-1].lower() != w.lower():
                cleaned.append(w)
        if not cleaned:
            return input_string
        sentence = " ".join(cleaned)
        return sentence[0].upper() + sentence[1:] + ("." if not sentence.endswith((".", "!", "?")) else "")

    def process_with_ollama(self, input_string: str) -> str:
        if not OLLAMA_IMPORT_OK:
            return self._naive_sentence_cleanup(input_string)

        model = "gemma3:1b"
        temperature = 0
        context_window_size = 8096
        seed = 42

        if self.current_language == "deutsch":
            messages = [
                {"role": "system",
                 "content": "You are an ultimate correction AI. You are provided multiple words that were tracked during an audio recording. Your goal is to form the sentence that was meant to say by the speaker. Usually the audio recording contains repeated words. You will reduce the repeated words and form the sentence. You strictly stick to the original content. You improve grammar and expression."},
                {"role": "user",
                 "content": "Du bist eine ultimative Korrektur-KI. Dir werden mehrere Wörter zur Verfügung gestellt, die während einer Audioaufnahme erfasst wurden. Dein Ziel ist es, den Satz zu bilden, den der Sprecher eigentlich sagen wollte. Normalerweise enthält die Audioaufnahme wiederholte Wörter. Du wirst die wiederholten Wörter reduzieren und den Satz bilden. Du hältst dich strikt an das Original und erfindest keine neuen Inhalte. Du verbesserst die Grammatik und den Ausdruck."},
                {"role": "user", "content": "hallo ich hallo bin zeuge"},
                {"role": "assistant", "content": "Hallo! Ich bin Zeuge."},
                {"role": "user", "content": "ich ich ich ich nicht nicht nicht gut gut"},
                {"role": "assistant", "content": "Ich finde das nicht gut."},
                {"role": "user", "content": f"{input_string}"}
            ]
        else:
            messages = [
                {"role": "system",
                 "content": "You are an ultimate correction AI. You are provided multiple words that were tracked during a sign language recording. Your goal is to form the sentence that the signer intended to communicate. Typically, the recording contains repeated words. You will reduce the repeated words and form the sentence. You strictly adhere to the original content. You improve grammar and expression."},
                {"role": "user",
                 "content": "You are an ultimate correction AI. You are provided with multiple words that were tracked during a sign language recording. Your goal is to form the sentence that the signer intended to communicate. Typically, the recording contains repeated words. You will reduce the repeated words and form the sentence. You strictly adhere to the original content. You improve grammar and expression."},
                {"role": "user", "content": "hello I hello am witness"},
                {"role": "assistant", "content": "Hello! I am a witness."},
                {"role": "user", "content": "I I I I not not not good good"},
                {"role": "assistant", "content": "I don't think that's good."},
                {"role": "user", "content": f"{input_string}"}
            ]

        try:
            response: ChatResponse = chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_ctx": context_window_size,
                    "seed": seed
                },
            )
            return response['message']['content']
        except Exception as e:
            fallback = self._naive_sentence_cleanup(input_string)
            note = (f"[Ollama nicht erreichbar - einfache Bereinigung verwendet: {e}]"
                    if self.current_language == "deutsch"
                    else f"[Ollama unavailable - used simple cleanup: {e}]")
            print(note)
            return fallback

    def print_protocol(self):
        printer = QPrinter(QPrinter.HighResolution)
        page_size = QPageSize(QPageSize.A4)
        printer.setPageSize(page_size)
        page_layout = QPageLayout(
            page_size,
            QPageLayout.Portrait,
            QMarginsF(15, 15, 15, 15)
        )
        printer.setPageLayout(page_layout)
        preview_dialog = QPrintPreviewDialog(printer, self)
        preview_dialog.setWindowTitle("Druckvorschau")
        preview_dialog.paintRequested.connect(self.print_preview)
        preview_dialog.exec_()

    def print_preview(self, printer):
        document = QTextDocument()
        document.setDefaultFont(QFont(self.default_font_family, 18))
        html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: {self.default_font_family}, sans-serif;
                    font-size: 28pt;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    font-size: 32pt;
                    font-weight: bold;
                }}
                .message {{
                    margin-bottom: 15px;
                    page-break-inside: avoid;
                }}
            </style>
        </head>
        <body>
            <div class="header">Protokoll der Transla{{w}}tion</div>
            {self.chat_area.toHtml()}
        </body>
        </html>
        """
        document.setHtml(html_content)
        printer.setResolution(120)
        document.print_(printer)

    def closeEvent(self, event):
        self.thread.stop()
        self.audio_recorder.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)

    font = QFont()
    font.setFamily(get_default_font_family())
    font.setPointSize(14)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())