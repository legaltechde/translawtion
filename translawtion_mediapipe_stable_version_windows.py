import sys
import cv2
import mediapipe as mp
import numpy as np
import os
import tempfile
import sounddevice as sd
import soundfile as sf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTextEdit, QFrame, QMessageBox, QCheckBox, QComboBox,
                             QSizePolicy, QSpacerItem, QRadioButton, QButtonGroup, QProgressBar, QDialog)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QUrl, QEvent, QMarginsF
from PyQt5.QtGui import QImage, QPixmap, QFont, QFontDatabase, QResizeEvent, QMovie, QTextDocument, QPageSize, QPageLayout
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog, QPrintPreviewDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from ollama import chat, ChatResponse
from datetime import datetime
from gtts import gTTS
import platform
import queue
import threading
import speech_recognition as sr
from deep_translator import GoogleTranslator

# MediaPipe Hands und Face Mesh initialisieren
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Konfiguration für die verschiedenen MediaPipe-Modelle
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

class AudioRecorder(QThread):
    recording_finished_signal = pyqtSignal(str)
    processing_started_signal = pyqtSignal()
    processing_finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.recording = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 44100
        self.channels = 1
        self.recognizer = sr.Recognizer()
        self.current_language = "deutsch"
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.recording_thread = None

    def set_language(self, language):
        self.current_language = language

    def run(self):
        while self._run_flag:
            if self.recording:
                try:
                    # Audio aufnehmen
                    recording = sd.rec(int(5 * self.sample_rate), samplerate=self.sample_rate,
                                      channels=self.channels, dtype='int16')
                    sd.wait()  # Wartet, bis die Aufnahme beendet ist
                    self.audio_queue.put(recording)
                except Exception as e:
                    print(f"Fehler bei der Audioaufnahme: {str(e)}")
                    self.recording_finished_signal.emit(f"Fehler bei der Audioaufnahme: {str(e)}")

    def toggle_recording(self):
        """Startet oder stoppt die Audioaufnahme für Person B"""
        if not self.recording:
            # Aufnahme starten
            self.recording = True
            self.audio_queue = queue.Queue()  # Neue Queue für jede Aufnahme
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
            print("Aufnahme gestartet")
        else:
            # Aufnahme stoppen und sofort verarbeiten
            self.recording = False
            if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                self.recording_thread.join()  # Auf das Ende des Aufnahme-Threads warten

            if not self.audio_queue.empty():
                try:
                    self.processing_started_signal.emit()
                    recording = self.audio_queue.get()
                    temp_file_path = os.path.join(self.script_dir, "temp_audio.wav")
                    sf.write(temp_file_path, recording, self.sample_rate)
                    self.process_audio(temp_file_path)
                    os.remove(temp_file_path)
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung der Audioaufnahme: {str(e)}")
                    self.recording_finished_signal.emit(f"Fehler bei der Verarbeitung der Audioaufnahme: {str(e)}")
                finally:
                    self.processing_finished_signal.emit()

    def record_audio(self):
        """Nimmt Audio auf und speichert es in der Queue"""
        try:
            print("Aufnahme-Thread gestartet")
            recording = sd.rec(int(5 * self.sample_rate), samplerate=self.sample_rate,
                              channels=self.channels, dtype='int16')
            while self.recording:
                # Hier könnte man eine Schleife für kontinuierliche Aufnahme implementieren
                # Für diese Anwendung reicht eine einzelne Aufnahme
                sd.wait()
            self.audio_queue.put(recording)
            print("Aufnahme beendet und in Queue gespeichert")
        except Exception as e:
            print(f"Fehler in Aufnahme-Thread: {str(e)}")
            self.audio_queue.put(None)  # Signalisiere Fehler

    def process_audio(self, audio_file):
        """Verarbeitet die Audioaufnahme und gibt den Text zurück"""
        try:
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)

                if self.current_language == "deutsch":
                    text = self.recognizer.recognize_google(audio_data, language="de-DE")
                else:
                    text = self.recognizer.recognize_google(audio_data, language="en-US")

                self.recording_finished_signal.emit(text)
        except Exception as e:
            print(f"Fehler bei der Spracherkennung: {str(e)}")
            self.recording_finished_signal.emit(f"Fehler bei der Spracherkennung: {str(e)}")

    def stop(self):
        self._run_flag = False
        self.wait()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QPixmap)
    recording_finished_signal = pyqtSignal(list)
    debug_info_signal = pyqtSignal(str)  # Neues Signal für Debug-Info

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.recording = False
        self.recognized_words = []
        self.last_word = None
        self.cap = cv2.VideoCapture(0)
        self.sign_language = "deutsch"  # Standardmäßig Deutsch
        self.show_keypoints = False  # Standardmäßig Keypoints ausblenden
        self.current_word = None  # Aktuelles erkanntes Wort speichern

    def set_sign_language(self, language):
        self.sign_language = language

    def set_show_keypoints(self, show):
        self.show_keypoints = show

    def run(self):
        while self._run_flag:
            ret, cv_img = self.cap.read()
            if not ret:
                continue

            # Bild für MediaPipe vorbereiten
            image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Verarbeitung mit verschiedenen MediaPipe-Modellen
            hand_results = hands.process(image)
            face_results = face_mesh.process(image)
            pose_results = pose.process(image)

            # Zeichne Keypoints, wenn aktiviert
            if self.show_keypoints:
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            cv_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            cv_img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())

                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        cv_img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Bild für die Anzeige konvertieren
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            self.change_pixmap_signal.emit(pixmap)

            # Nur verarbeiten, wenn Aufnahme aktiv ist
            if self.recording and hand_results.multi_hand_landmarks:
                current_word = None

                for hand_landmarks in hand_results.multi_hand_landmarks:
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                    # Abstand zwischen Daumen und Zeigefinger berechnen
                    distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

                    if self.sign_language == "deutsch":
                        # Erkennung von "Ich" (Zeigefinger auf die Brust tippen)
                        if (index_tip.y < wrist.y and
                            middle_tip.y > wrist.y and
                            ring_tip.y > wrist.y and
                            pinky_tip.y > wrist.y):
                            current_word = "Ich"

                        # Erkennung von "Tat" (Faust)
                        elif (thumb_tip.y > index_tip.y and
                              thumb_tip.y > middle_tip.y and
                              thumb_tip.y > ring_tip.y and
                              thumb_tip.y > pinky_tip.y and
                              distance < 0.05):
                            current_word = "Tat"

                        # Erkennung von "nicht" (Handfläche nach außen)
                        elif (index_tip.x < wrist.x and
                              middle_tip.x < wrist.x and
                              ring_tip.x < wrist.x and
                              pinky_tip.x < wrist.x):
                            current_word = "nicht"

                        # Erkennung von "begangen" (Hände nach vorne schieben)
                        elif (index_tip.y < wrist.y and
                              middle_tip.y < wrist.y and
                              ring_tip.y < wrist.y and
                              pinky_tip.y < wrist.y):
                            current_word = "begangen"

                    elif self.sign_language == "amerikanisch":
                        # Erkennung von "I" (Zeigefinger auf die Brust tippen)
                        if (index_tip.y < wrist.y and
                            middle_tip.y > wrist.y and
                            ring_tip.y > wrist.y and
                            pinky_tip.y > wrist.y):
                            current_word = "I"

                        # Erkennung von "did" (Faust)
                        elif (thumb_tip.y > index_tip.y and
                              thumb_tip.y > middle_tip.y and
                              thumb_tip.y > ring_tip.y and
                              thumb_tip.y > pinky_tip.y and
                              distance < 0.05):
                            current_word = "did"

                        # Erkennung von "not" (Handfläche nach außen)
                        elif (index_tip.x < wrist.x and
                              middle_tip.x < wrist.x and
                              ring_tip.x < wrist.x and
                              pinky_tip.x < wrist.x):
                            current_word = "not"

                        # Erkennung von "commit" (Hände nach vorne schieben)
                        elif (index_tip.y < wrist.y and
                              middle_tip.y < wrist.y and
                              ring_tip.y < wrist.y and
                              pinky_tip.y < wrist.y):
                            current_word = "commit"

                # Aktuelles Wort speichern und Debug-Info senden
                self.current_word = current_word
                if self.show_keypoints and current_word:
                    self.debug_info_signal.emit(f"Erkannt: {current_word}")

                # Nur hinzufügen, wenn das aktuelle Wort anders ist als das letzte
                if current_word and current_word != self.last_word:
                    self.recognized_words.append(current_word)
                    self.last_word = current_word

    def stop(self):
        self._run_flag = False
        self.cap.release()
        self.wait()

    def toggle_recording(self):
        self.recording = not self.recording
        if not self.recording:
            # Sende die erkannten Wörter, wenn Aufnahme beendet wird
            if self.recognized_words:
                self.recording_finished_signal.emit(self.recognized_words.copy())
            self.recognized_words = []
            self.last_word = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handzeichen-Erkennung")
        self.setGeometry(100, 100, 1200, 700)  # Größeres Fenster für bessere Skalierung

        # Medienplayer für Sprachausgabe
        self.player = QMediaPlayer()

        # Schriftart für bessere Lesbarkeit
        self.base_font_size = 12
        self.font = QFont()
        self.font.setPointSize(self.base_font_size)
        self.font.setFamily("Arial")

        # Übersetzer initialisieren
        self.translator = GoogleTranslator(source='auto')

        # Hauptwidget und Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Logo hinzufügen
        self.logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            if not logo_pixmap.isNull():
                self.logo_label.setPixmap(logo_pixmap.scaledToWidth(200, Qt.SmoothTransformation))
                self.logo_label.setAlignment(Qt.AlignCenter)
                main_layout.addWidget(self.logo_label)

        # Benutzerauswahl (Person A oder B)
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
        main_layout.addLayout(user_selection_layout)

        # Hauptbereich
        content_layout = QHBoxLayout()

        # Linkes Panel für die Kamera
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 5px;
        """)
        self.video_label.setMinimumSize(640, 480)  # Mindestgröße festlegen
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Skalierung ermöglichen
        self.video_label.setToolTip("Hier wird das Kamerabild angezeigt")
        left_layout.addWidget(self.video_label)

        # Label für Debug-Info hinzufügen
        self.debug_info_label = QLabel()
        self.debug_info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.debug_info_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 128);
            color: white;
            padding: 5px;
            border-radius: 3px;
            font-size: 14px;
            margin-top: 5px;
        """)
        self.debug_info_label.setVisible(False)  # Standardmäßig unsichtbar
        left_layout.addWidget(self.debug_info_label)

        # Checkbox für Tracking-Keypoints hinzufügen
        self.keypoints_checkbox = QCheckBox("Tracking-Keypoints anzeigen (Debug)")
        self.keypoints_checkbox.setFont(self.font)
        self.keypoints_checkbox.setToolTip("Aktiviert/Deaktiviert die Anzeige der Hand-Tracking-Keypoints")
        self.keypoints_checkbox.stateChanged.connect(self.toggle_keypoints)
        left_layout.addWidget(self.keypoints_checkbox)

        self.control_hint = QLabel("Drücken Sie 'Q', um Aufnahme zu starten/beenden")
        self.control_hint.setAlignment(Qt.AlignCenter)
        self.control_hint.setFont(self.font)
        self.control_hint.setToolTip("Tastaturkürzel für Aufnahme: Q-Taste")
        left_layout.addWidget(self.control_hint)

        # Layout für Sprachauswahlen
        language_selection_layout = QHBoxLayout()

        # Layout für Sprecher A
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

        # Layout für Sprecher B
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

        # Rechtes Panel für den Chat
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # Chat-Bereich
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("""
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            font-family: Arial;
            font-size: 14px;
        """)
        self.chat_area.setFont(self.font)
        self.chat_area.setToolTip("Hier wird der Chatverlauf angezeigt")
        right_layout.addWidget(self.chat_area, 1)

        # Fortschrittsanzeige für die Audioverarbeitung
        self.progress_label = QLabel("Audio wird verarbeitet...")
        self.progress_label.setVisible(False)
        self.progress_label.setFont(self.font)
        right_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        right_layout.addWidget(self.progress_bar)

        # Animation für die Audioverarbeitung
        self.processing_animation = QMovie(os.path.join(os.path.dirname(__file__), "processing.gif"))
        self.animation_label = QLabel()
        self.animation_label.setMovie(self.processing_animation)
        self.animation_label.setVisible(False)
        self.animation_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.animation_label)

        # Button-Bereich
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        # Plus/Minus Buttons für Textgröße
        self.minus_button = QPushButton("A-")
        self.minus_button.setFixedSize(40, 30)
        self.minus_button.clicked.connect(self.decrease_font_size)
        self.minus_button.setToolTip("Text verkleinern (Strg + -)")
        button_layout.addWidget(self.minus_button)

        self.plus_button = QPushButton("A+")
        self.plus_button.setFixedSize(40, 30)
        self.plus_button.clicked.connect(self.increase_font_size)
        self.plus_button.setToolTip("Text vergrößern (Strg + +)")
        button_layout.addWidget(self.plus_button)

        # Leerraum hinzufügen
        button_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.accept_button = QPushButton("Akzeptieren")
        self.accept_button.clicked.connect(self.accept_result)
        self.accept_button.setEnabled(False)
        self.accept_button.setFont(self.font)
        self.accept_button.setToolTip("Akzeptieren Sie den vorgeschlagenen Text")
        button_layout.addWidget(self.accept_button)

        self.reject_button = QPushButton("Ablehnen")
        self.reject_button.clicked.connect(self.reject_result)
        self.reject_button.setEnabled(False)
        self.reject_button.setFont(self.font)
        self.reject_button.setToolTip("Lehnen Sie den vorgeschlagenen Text ab")
        button_layout.addWidget(self.reject_button)

        self.clear_button = QPushButton("Chat löschen")
        self.clear_button.clicked.connect(self.clear_chat)
        self.clear_button.setFont(self.font)
        self.clear_button.setToolTip("Löscht den gesamten Chatverlauf")
        button_layout.addWidget(self.clear_button)

        # Druck-Button hinzufügen
        self.print_button = QPushButton("Drucken")
        self.print_button.clicked.connect(self.print_protocol)
        self.print_button.setFont(self.font)
        self.print_button.setToolTip("Druckt das Protokoll")
        button_layout.addWidget(self.print_button)

        right_layout.addLayout(button_layout)

        # Sprachausgabe-Optionen
        self.speech_checkbox = QCheckBox("Sprachausgabe aktivieren")
        self.speech_checkbox.setChecked(True)
        self.speech_checkbox.setFont(self.font)
        self.speech_checkbox.setToolTip("Aktivieren/Deaktivieren der Sprachausgabe")
        right_layout.addWidget(self.speech_checkbox)

        # Zielsprache für Übersetzung
        self.target_language_combo = QComboBox()
        self.target_language_combo.addItem("Deutsch")
        self.target_language_combo.addItem("Englisch")
        self.target_language_combo.addItem("Französisch")
        self.target_language_combo.addItem("Spanisch")
        self.target_language_combo.setFont(self.font)
        self.target_language_combo.setToolTip("Wählen Sie die Zielsprache für die Übersetzung aus")
        right_layout.addWidget(self.target_language_combo)

        # Übersetzungsbutton
        self.translate_button = QPushButton("Übersetzen")
        self.translate_button.clicked.connect(self.translate_text)
        self.translate_button.setFont(self.font)
        self.translate_button.setToolTip("Übersetzt den aktuellen Text in die Zielsprache")
        right_layout.addWidget(self.translate_button)

        content_layout.addWidget(right_panel, 1)

        main_layout.addLayout(content_layout)

        # Audio-Recorder initialisieren
        self.audio_recorder = AudioRecorder()
        self.audio_recorder.recording_finished_signal.connect(self.process_audio_recording)
        self.audio_recorder.processing_started_signal.connect(self.show_processing_indicator)
        self.audio_recorder.processing_finished_signal.connect(self.hide_processing_indicator)
        self.audio_recorder.start()

        # Video-Thread starten
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.recording_finished_signal.connect(self.process_recording)
        self.thread.debug_info_signal.connect(self.update_debug_info)  # Neues Signal verbinden
        self.thread.start()

        # Aktuelle Sprache speichern
        self.current_language = "deutsch"
        self.current_user = "A"  # Standardmäßig Person A
        self.current_translation = None
        self.current_llm_output = None
        self.last_spoken_text = None  # Speichert den zuletzt gesprochenen Text

        # GUI-Sprache initialisieren
        self.update_gui_language()

        # Chat initialisieren
        self.add_chat_message("System", "Willkommen! Wählen Sie Person A oder B aus und drücken Sie 'Q', um eine Aufnahme zu starten.")

    def toggle_keypoints(self, state):
        """Aktiviert/Deaktiviert die Anzeige der Tracking-Keypoints"""
        self.thread.set_show_keypoints(state == Qt.Checked)
        self.debug_info_label.setVisible(state == Qt.Checked)  # Debug-Info nur anzeigen, wenn Debug-Modus aktiviert ist

        if state == Qt.Checked:
            self.add_chat_message("System", "Tracking-Keypoints werden angezeigt (Debug-Modus)")
        else:
            self.add_chat_message("System", "Tracking-Keypoints werden ausgeblendet")
            self.debug_info_label.clear()

    def update_debug_info(self, info):
        """Aktualisiert die Debug-Info im Kamera-Feld"""
        self.debug_info_label.setText(info)

    def update_gui_language(self):
        """Aktualisiert die GUI-Sprache basierend auf dem aktuellen Sprecher und dessen Spracheinstellung"""
        # Sprache des aktuellen Sprechers ermitteln
        if self.current_user == "A":
            # Sprecher A verwendet Gebärdensprache
            language = "deutsch" if self.language_combo_a.currentIndex() == 0 else "amerikanisch"
        else:
            # Sprecher B verwendet gesprochene Sprache
            language = "deutsch" if self.language_combo_b.currentIndex() == 0 else "englisch"

        self.current_language = language
        self.update_ui_language()

    def show_processing_indicator(self):
        """Zeigt die Verarbeitungsindikatoren an"""
        self.progress_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.animation_label.setVisible(True)
        self.processing_animation.start()

        if self.current_language == "deutsch":
            self.progress_label.setText("Audio wird verarbeitet...")
        else:
            self.progress_label.setText("Processing audio...")

    def hide_processing_indicator(self):
        """Versteckt die Verarbeitungsindikatoren"""
        self.progress_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.animation_label.setVisible(False)
        self.processing_animation.stop()

    def switch_user_mode(self, button):
        """Wechselt zwischen Person A und Person B Modus"""
        self.current_user = "A" if button == self.user_a_radio else "B"

        # GUI-Sprache basierend auf dem ausgewählten Sprecher aktualisieren
        self.update_gui_language()

        if self.current_user == "A":
            self.control_hint.setText("Drücken Sie 'Q', um Gebärdenaufnahme zu starten/beenden")
            self.add_chat_message("System", f"Modus: Person A (Gebärdensprache). Drücken Sie 'Q' für Gebärdenaufnahme.")
        else:
            self.control_hint.setText("Drücken Sie 'Q', um Sprachaufnahme zu starten/beenden")
            self.add_chat_message("System", f"Modus: Person B (Sprache). Drücken Sie 'Q' für Sprachaufnahme.")

    def resizeEvent(self, event: QResizeEvent):
        """Behandelt das Ändern der Fenstergröße"""
        super().resizeEvent(event)
        self.adjust_font_size()

    def adjust_font_size(self):
        """Passt die Schriftgröße basierend auf der Fenstergröße an"""
        # Basisgröße basierend auf Fensterhöhe berechnen
        base_size = max(8, min(16, int(self.height() / 50)))
        self.base_font_size = base_size

        # Schriftgröße für alle Elemente aktualisieren
        self.update_font_sizes()

    def update_font_sizes(self):
        """Aktualisiert die Schriftgrößen aller UI-Elemente"""
        self.font.setPointSize(self.base_font_size)

        # Alle UI-Elemente aktualisieren
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

        # Chat-Bereich speziell anpassen
        chat_font = QFont()
        chat_font.setPointSize(self.base_font_size + 2)
        chat_font.setFamily("Arial")
        self.chat_area.setStyleSheet(f"""
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            font-family: Arial;
            font-size: {self.base_font_size + 2}px;
        """)

    def increase_font_size(self):
        """Vergrößert die Schriftgröße in gröberen Schritten"""
        self.base_font_size = min(24, self.base_font_size + 2)  # Schrittweite von 2 statt 1
        self.update_font_sizes()

    def decrease_font_size(self):
        """Verkleinert die Schriftgröße in gröberen Schritten"""
        self.base_font_size = max(8, self.base_font_size - 2)  # Schrittweite von 2 statt 1
        self.update_font_sizes()

    def keyPressEvent(self, event):
        # Tastaturkürzel für Textvergrößerung/-verkleinerung
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                self.increase_font_size()
            elif event.key() == Qt.Key_Minus:
                self.decrease_font_size()
        elif event.key() == Qt.Key_Q:
            if self.current_user == "A":
                self.thread.toggle_recording()
                if self.thread.recording:
                    if self.current_language == "deutsch":
                        self.control_hint.setText("Gebärdenaufnahme läuft... Drücken Sie 'Q' zum Beenden")
                        self.add_chat_message("System", "Gebärdenaufnahme gestartet...")
                    else:
                        self.control_hint.setText("Sign language recording in progress... Press 'Q' to stop")
                        self.add_chat_message("System", "Sign language recording started...")
                    self.control_hint.setStyleSheet("color: red; font-weight: bold;")
                else:
                    if self.current_language == "deutsch":
                        self.control_hint.setText("Drücken Sie 'Q', um Gebärdenaufnahme zu starten")
                        self.control_hint.setStyleSheet("color: black; font-weight: normal;")
                    else:
                        self.control_hint.setText("Press 'Q' to start sign language recording")
                        self.control_hint.setStyleSheet("color: black; font-weight: normal;")
            else:
                # Für Person B: Aufnahme starten/stoppen
                if not self.audio_recorder.recording:
                    self.audio_recorder.toggle_recording()
                    if self.current_language == "deutsch":
                        self.control_hint.setText("Sprachaufnahme läuft... Drücken Sie 'Q' zum Beenden")
                        self.add_chat_message("System", "Sprachaufnahme gestartet...")
                    else:
                        self.control_hint.setText("Voice recording in progress... Press 'Q' to stop")
                        self.add_chat_message("System", "Voice recording started...")
                    self.control_hint.setStyleSheet("color: red; font-weight: bold;")
                else:
                    self.audio_recorder.toggle_recording()
                    if self.current_language == "deutsch":
                        self.control_hint.setText("Drücken Sie 'Q', um Sprachaufnahme zu starten")
                        self.control_hint.setStyleSheet("color: black; font-weight: normal;")
                    else:
                        self.control_hint.setText("Press 'Q' to start voice recording")
                        self.control_hint.setStyleSheet("color: black; font-weight: normal;")

    def process_audio_recording(self, text):
        """Verarbeitet die Audioaufnahme von Person B und zeigt den Text direkt an"""
        if self.current_user == "B":
            self.add_chat_message("Sie", text)  # Wird in add_chat_message zu "Person B" umgewandelt
            self.current_llm_output = text

            # Buttons aktivieren
            self.accept_button.setEnabled(True)
            self.reject_button.setEnabled(True)

    def change_sign_language(self):
        """Ändert die Gebärdensprache und aktualisiert die GUI-Sprache"""
        language = "deutsch" if self.language_combo_a.currentIndex() == 0 else "amerikanisch"
        self.thread.set_sign_language(language)

        # GUI-Sprache aktualisieren, wenn Sprecher A ausgewählt ist
        if self.current_user == "A":
            self.update_gui_language()

        # Nachricht im Chat
        if language == "deutsch":
            self.add_chat_message("System", "Gebärdensprache geändert auf Deutsch")
        else:
            self.add_chat_message("System", "Sign language changed to American Sign Language")

    def change_spoken_language(self):
        """Ändert die gesprochene Sprache und aktualisiert die GUI-Sprache"""
        language = "deutsch" if self.language_combo_b.currentIndex() == 0 else "englisch"
        self.audio_recorder.set_language(language)

        # GUI-Sprache aktualisieren, wenn Sprecher B ausgewählt ist
        if self.current_user == "B":
            self.update_gui_language()

        # Nachricht im Chat
        if language == "deutsch":
            self.add_chat_message("System", "Sprache für Sprecher B geändert auf Deutsch")
        else:
            self.add_chat_message("System", "Language for speaker B changed to English")

    def update_ui_language(self):
        if self.current_language == "deutsch":
            self.setWindowTitle("Transla{w}tion")
            if self.current_user == "A":
                self.control_hint.setText("Drücken Sie 'Q', um Gebärdenaufnahme zu starten/beenden")
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
            self.keypoints_checkbox.setText("Tracking-Keypoints anzeigen (Debug)")
            self.keypoints_checkbox.setToolTip("Aktiviert/Deaktiviert die Anzeige der Hand-Tracking-Keypoints")
        else:
            self.setWindowTitle("Transla{w}tion")
            if self.current_user == "A":
                self.control_hint.setText("Press 'Q' to start/stop sign language recording")
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
            self.keypoints_checkbox.setText("Show tracking keypoints (Debug)")
            self.keypoints_checkbox.setToolTip("Enable/Disable display of hand tracking keypoints")

    def update_image(self, pixmap):
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def process_recording(self, words):
        """Verarbeitet die Gebärdenaufnahme von Person A"""
        if self.current_user == "A" and words:
            output = " ".join(words)
            self.add_chat_message("Sie", output)  # Wird in add_chat_message zu "Person A" umgewandelt

            # LLM-Verarbeitung
            llm_out = self.process_with_ollama(output)
            self.current_llm_output = llm_out
            if self.current_language == "deutsch":
                self.add_chat_message("System (Vorschlag)", llm_out)
            else:
                self.add_chat_message("System (Suggestion)", llm_out)

            # Buttons aktivieren
            self.accept_button.setEnabled(True)
            self.reject_button.setEnabled(True)

    def add_chat_message(self, sender, message):
        timestamp = datetime.now().strftime("%H:%M")
        # Anpassen des Senders basierend auf dem aktuellen Benutzer
        if sender == "Sie" or sender == "You":
            sender = f"Person {self.current_user}"
        self.chat_area.append(f"<b>{timestamp} - {sender}:</b> {message}<br>")
        self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())

    def accept_result(self):
        if self.current_llm_output:
            if self.current_language == "deutsch":
                self.add_chat_message("System", f"Akzeptiert: {self.current_llm_output}")
            else:
                self.add_chat_message("System", f"Accepted: {self.current_llm_output}")

            # Sprachausgabe, wenn aktiviert
            if self.speech_checkbox.isChecked():
                self.speak_text(self.current_llm_output)

            # Zurücksetzen der Variablen
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
            # Sprache basierend auf der aktuellen Einstellung
            language = 'de' if self.current_language == "deutsch" else 'en'

            # Temporäre Audiodatei erstellen
            temp_file_path = os.path.join(
                self.audio_recorder.script_dir,
                f"temp_speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            )
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_file_path)

            # Audio abspielen
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_file_path)))
            self.player.play()

            # Datei nach dem Abspielen löschen
            QTimer.singleShot(5000, lambda: os.remove(temp_file_path))  # 5 Sekunden Wartezeit

        except Exception as e:
            if self.current_language == "deutsch":
                self.add_chat_message("System", f"Fehler bei der Sprachausgabe: {str(e)}")
            else:
                self.add_chat_message("System", f"Error in speech output: {str(e)}")

    def translate_text(self):
        """Übersetzt den aktuellen Text in die Zielsprache"""
        if self.current_llm_output:
            try:
                # Zielsprache bestimmen
                target_lang = self.target_language_combo.currentText().lower()
                if target_lang == "deutsch":
                    target_lang_code = "de"
                elif target_lang == "englisch":
                    target_lang_code = "en"
                elif target_lang == "französisch":
                    target_lang_code = "fr"
                elif target_lang == "spanisch":
                    target_lang_code = "es"
                else:
                    target_lang_code = "en"  # Standardmäßig Englisch

                # Text übersetzen
                translation = GoogleTranslator(source='auto', target=target_lang_code).translate(self.current_llm_output)
                self.current_translation = translation

                # Übersetzung anzeigen
                if self.current_language == "deutsch":
                    self.add_chat_message("System (Übersetzung)", f"Übersetzung ({target_lang}): {self.current_translation}")
                else:
                    self.add_chat_message("System (Translation)", f"Translation ({target_lang}): {self.current_translation}")

                # Sprachausgabe der Übersetzung, wenn aktiviert
                if self.speech_checkbox.isChecked():
                    self.speak_translation(self.current_translation, target_lang_code)

            except Exception as e:
                if self.current_language == "deutsch":
                    self.add_chat_message("System", f"Fehler bei der Übersetzung: {str(e)}")
                else:
                    self.add_chat_message("System", f"Translation error: {str(e)}")

    def speak_translation(self, text, language_code):
        """Gibt die Übersetzung als Sprache aus"""
        try:
            # Temporäre Audiodatei erstellen
            temp_file_path = os.path.join(
                self.audio_recorder.script_dir,
                f"temp_translation_speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            )

            tts = gTTS(text=text, lang=language_code, slow=False)
            tts.save(temp_file_path)

            # Audio abspielen
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_file_path)))
            self.player.play()

            # Datei nach dem Abspielen löschen
            QTimer.singleShot(5000, lambda: os.remove(temp_file_path))  # 5 Sekunden Wartezeit

        except Exception as e:
            if self.current_language == "deutsch":
                self.add_chat_message("System", f"Fehler bei der Sprachausgabe der Übersetzung: {str(e)}")
            else:
                self.add_chat_message("System", f"Error in translation speech output: {str(e)}")

    def process_with_ollama(self, input_string: str) -> str:
        # Parameter
        model = "gemma3:1b"
        temperature = 0
        context_window_size = 8096
        seed = 42

        # Sprachabhängige Nachrichten
        if self.current_language == "deutsch":
            messages = [
                {"role": "system",
                 "content": "You are an ultimate correction AI. You are provided multiple words that were tracked during an audio recording. Your goal is to form the sentence that was meant to say by the speaker. Usually the audio recording contains repeated words. You will reduce the repeated words and form the sentence. You strictly stick to the original content. You improve grammar and expression."},
                {"role": "user",
                 "content": "Du bist eine ultimative Korrektur-KI. Dir werden mehrere Wörter zur Verfügung gestellt, die während einer Audioaufnahme erfasst wurden. Dein Ziel ist es, den Satz zu bilden, den der Sprecher eigentlich sagen wollte. Normalerweise enthält die Audioaufnahme wiederholte Wörter. Du wirst die wiederholten Wörter reduzieren und den Satz bilden. Du hältst dich strikt an das Original und erfindest keine neuen Inhalte. Du verbesserst die Grammatik und den Ausdruck."},
                {"role": "user", "content": "hallo ich hallo bin zeuge"},
                {"role": "assistant", "content": "Hallo! Ich bin Zeuge."},
                {"role": "user", "content": "ich ich ich ich tat tat tat tat ich ich nicht begangen begangen begangen"},
                {"role": "assistant", "content": "Ich habe die Tat nicht begangen."},
                {"role": "user", "content": f"{input_string}"}
            ]
        else:  # Amerikanisch/Englisch
            messages = [
                {"role": "system",
                 "content": "You are an ultimate correction AI. You are provided multiple words that were tracked during a sign language recording. Your goal is to form the sentence that the signer intended to communicate. Typically, the recording contains repeated words. You will reduce the repeated words and form the sentence. You strictly adhere to the original content. You improve grammar and expression."},
                {"role": "user",
                 "content": "You are an ultimate correction AI. You are provided with multiple words that were tracked during a sign language recording. Your goal is to form the sentence that the signer intended to communicate. Typically, the recording contains repeated words. You will reduce the repeated words and form the sentence. You strictly adhere to the original content. You improve grammar and expression."},
                {"role": "user", "content": "hello I hello am witness"},
                {"role": "assistant", "content": "Hello! I am a witness."},
                {"role": "user", "content": "I I I I did did did did I I not commit commit commit"},
                {"role": "assistant", "content": "I did not commit the act."},
                {"role": "user", "content": f"{input_string}"}
            ]

        # Anfrage an das Ollama-Modell
        response: ChatResponse = chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_ctx": context_window_size,
                "seed": seed
            }
        )

        # Die Antwort aus der Antwortstruktur extrahieren und zurückgeben
        return response['message']['content']

    def print_protocol(self):
        """Öffnet den Druckdialog für das Protokoll mit A4-Formatierung"""
        # Erstellen eines Druckdialogs
        printer = QPrinter(QPrinter.HighResolution)

        # A4-Seitenformat einstellen
        page_size = QPageSize(QPageSize.A4)
        printer.setPageSize(page_size)

        # Seitenränder einstellen (in Millimetern)
        page_layout = QPageLayout(
            page_size,
            QPageLayout.Portrait,
            QMarginsF(15, 15, 15, 15)  # Links, Oben, Rechts, Unten
        )
        printer.setPageLayout(page_layout)

        # Druckvorschau-Dialog anzeigen
        preview_dialog = QPrintPreviewDialog(printer, self)
        preview_dialog.setWindowTitle("Druckvorschau")

        # Signal für das Malen der Vorschau verbinden
        preview_dialog.paintRequested.connect(self.print_preview)

        # Dialog anzeigen
        preview_dialog.exec_()

    def print_preview(self, printer):
        """Erstellt die Druckvorschau mit angepasster Formatierung"""
        # Dokument erstellen
        document = QTextDocument()
        document.setDefaultFont(QFont("Arial", 18))  # Standardschriftart und -größe

        # HTML-Inhalt mit angepasster Skalierung
        html_content = f"""
        <html>
        <head>
            <style>
                @media print {{
                    body {{
                        zoom: 1.5;  /* Skalierung für Druckansicht */
                    }}
                }}
                body {{
                    font-family: Arial, sans-serif;
                    font-size: 28pt;  /* Deutlich größere Schriftgröße */
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    font-size: 32pt;  /* Noch größere Überschrift */
                    font-weight: bold;
                }}
                .message {{
                    margin-bottom: 15px;
                    page-break-inside: avoid;
                }}
                .timestamp {{
                    font-weight: bold;
                    color: #555;
                    font-size: 24pt;  /* Deutliche Größe für Zeitstempel */
                }}
                .sender {{
                    font-weight: bold;
                    font-size: 24pt;  /* Deutliche Größe für Absender */
                }}
                .content {{
                    margin-top: 5px;
                    font-size: 28pt;  /* Hauptinhalt gut lesbar */
                }}
            </style>
        </head>
        <body>
            <div class="header">Protokoll der Transla&#123;w&#125;tion</div>
            {self.chat_area.toHtml()}
        </body>
        </html>
        """

        # HTML in das Dokument laden
        document.setHtml(html_content)

        # Druckauflösung auf Standard setzen
        printer.setResolution(120)  # Mittelwert zwischen Bildschirm- und Druckauflösung
        document.print_(printer)

    def closeEvent(self, event):
        self.thread.stop()
        self.audio_recorder.stop()
        hands.close()
        face_mesh.close()
        pose.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Schriftart für die gesamte Anwendung setzen
    font = QFont()
    font.setPointSize(14)
    font.setFamily("Arial")
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())