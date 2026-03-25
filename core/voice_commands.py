"""
VoiceCommands — Offline voice control using Vosk (runs locally, no internet).
Listens continuously for keywords to switch drills hands-free.

Say things like:
  "knee" → switch to knee bend drill
  "shoulder" → shoulder turn
  "racket" or "lag" → racket lag
  "elbow" → elbow extension
  "follow" → follow-through
  "trophy" → trophy position (serve)
  "serve" → serve drills
  "ready" → ready position
  "split" → split step
  "restart" or "again" → restart current drill
  "pause" → pause/resume
  "quit" or "stop" → exit
"""

import threading
import queue
import os
import json

# Keyword → drill ID mapping
KEYWORD_MAP = [
    # Serve drills (check first — "serve knee" before "knee")
    (['trophy'], 6),
    (['serve knee', 'serve load', 'serve bend'], 7),
    (['serve extension', 'serve reach'], 8),
    (['serve'], 6),  # default serve = trophy position
    # Groundstroke drills
    (['knee', 'knees', 'bend'], 1),
    (['shoulder', 'turn', 'coil'], 2),
    (['racket', 'lag', 'rack'], 3),
    (['elbow', 'extension', 'extend'], 4),
    (['follow', 'finish', 'through'], 5),
    # Fundamentals
    (['ready', 'stance'], 9),
    (['split', 'step'], 10),
    # Control commands
    (['restart', 'again', 'reset', 'redo'], 'restart'),
    (['pause', 'stop', 'wait', 'hold'], 'pause'),
    (['quit', 'exit', 'done', 'close'], 'quit'),
    (['next', 'switch'], 'next_drill'),
]

VOSK_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data', 'models', 'vosk-model-small-en-us-0.15'
)


class VoiceCommands:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.command_queue = queue.Queue()
        self._running = False
        self._thread = None

        if enabled:
            self.start()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _listen_loop(self):
        """Background thread — continuous offline speech recognition with Vosk."""
        try:
            from vosk import Model, KaldiRecognizer
            import pyaudio
        except ImportError as e:
            print(f"  Voice commands: missing dependency — {e}")
            self._running = False
            return

        # Load Vosk model
        if not os.path.exists(VOSK_MODEL_PATH):
            print(f"  Voice commands: model not found at {VOSK_MODEL_PATH}")
            self._running = False
            return

        try:
            model = Model(VOSK_MODEL_PATH)
            recognizer = KaldiRecognizer(model, 16000)
            recognizer.SetWords(True)
        except Exception as e:
            print(f"  Voice commands: model load error — {e}")
            self._running = False
            return

        # Open microphone
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4000,
            )
            stream.start_stream()
        except Exception as e:
            print(f"  Voice commands: mic error — {e}")
            self._running = False
            return

        print("  Voice commands: READY (say drill names to switch)")

        while self._running:
            try:
                data = stream.read(4000, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get('text', '').strip()
                    if text:
                        command = self._match_command(text)
                        if command is not None:
                            self.command_queue.put(command)
                            print(f"  Voice: '{text}' → {command}")
                else:
                    # Partial result — check for keywords in real-time
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get('partial', '').strip()
                    if len(partial_text) > 2:
                        command = self._match_command(partial_text)
                        if command is not None:
                            self.command_queue.put(command)
                            print(f"  Voice: '{partial_text}' → {command}")
                            recognizer.Reset()  # clear partial to avoid double-trigger

            except Exception:
                continue

        stream.stop_stream()
        stream.close()
        pa.terminate()

    def _match_command(self, text):
        """Match spoken text to a drill command."""
        text = text.lower()
        for keywords, command in KEYWORD_MAP:
            for kw in keywords:
                if kw in text:
                    return command
        return None

    def get_command(self):
        """Non-blocking check for a voice command."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._running = False
