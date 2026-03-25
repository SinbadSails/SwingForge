"""
VoiceCommands — Listen for voice commands to switch drills hands-free.
Runs in a background thread, uses speech_recognition with Google API.
No LLM needed — just keyword matching.

Say things like:
  "knee" or "knee bend" → switch to knee bend drill
  "shoulder" → shoulder turn
  "racket" or "lag" → racket lag
  "elbow" or "extension" → elbow extension
  "follow" or "follow through" → follow-through
  "trophy" → trophy position (serve)
  "serve knee" → serve knee load
  "serve extension" or "reach" → serve extension
  "ready" → ready position
  "split" → split step
  "restart" or "again" → restart current drill
  "quit" or "stop" → signal quit
"""

import threading
import queue
import time


# Keyword → drill ID mapping (checked in order, first match wins)
KEYWORD_MAP = [
    # Serve drills (check first because "serve" + "knee" needs to match serve knee, not just knee)
    (['trophy', 'trophy position'], 6),
    (['serve knee', 'serve load', 'serve bend'], 7),
    (['serve extension', 'serve reach', 'serve arm'], 8),
    # Groundstroke drills
    (['knee', 'knee bend', 'knees', 'legs'], 1),
    (['shoulder', 'shoulder turn', 'turn', 'coil', 'unit turn'], 2),
    (['racket', 'lag', 'racket lag', 'wrist'], 3),
    (['elbow', 'extension', 'extend', 'arm'], 4),
    (['follow', 'follow through', 'finish'], 5),
    # Fundamentals
    (['ready', 'ready position', 'stance'], 9),
    (['split', 'split step'], 10),
    # Control commands
    (['restart', 'again', 'reset', 'redo'], 'restart'),
    (['quit', 'stop', 'exit', 'done'], 'quit'),
]


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
        """Background thread that listens for voice commands."""
        try:
            import speech_recognition as sr
        except ImportError:
            print("  Voice commands: speech_recognition not installed")
            self._running = False
            return

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300  # sensitivity
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.5  # seconds of silence before processing

        try:
            mic = sr.Microphone()
        except (OSError, AttributeError):
            print("  Voice commands: No microphone found")
            self._running = False
            return

        print("  Voice commands: Listening... (say drill names to switch)")

        while self._running:
            try:
                with mic as source:
                    # Short listen window so it doesn't block
                    recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=3)

                # Use Google's free speech API (requires internet)
                text = recognizer.recognize_google(audio).lower().strip()

                if text:
                    command = self._match_command(text)
                    if command is not None:
                        self.command_queue.put(command)
                        print(f"  Voice: '{text}' → {command}")

            except Exception:
                # Timeout, no speech detected, API error — all expected, just continue
                continue

    def _match_command(self, text):
        """Match spoken text to a drill command."""
        for keywords, command in KEYWORD_MAP:
            for kw in keywords:
                if kw in text:
                    return command
        return None

    def get_command(self):
        """Non-blocking check for a voice command. Returns drill_id, 'restart', 'quit', or None."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._running = False
