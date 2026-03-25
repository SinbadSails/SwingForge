"""
VoiceCoach — Local TTS coaching cues using pyttsx3.
Speaks coaching tips without blocking the main loop.
"""

import threading
import time
import queue


class VoiceCoach:
    def __init__(self, enabled=True, rate=180):
        self.enabled = enabled
        self.msg_queue = queue.Queue()
        self.last_spoken = {}
        self.cooldown = 3.0  # seconds between same message
        self.rate = rate

        if enabled:
            self._thread = threading.Thread(target=self._speak_loop, daemon=True)
            self._thread.start()

    def _speak_loop(self):
        """Background thread that processes speech queue."""
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', 0.9)

        while True:
            try:
                msg = self.msg_queue.get(timeout=1.0)
                if msg == '__STOP__':
                    break
                engine.say(msg)
                engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS error: {e}")

    def say(self, message, category='general'):
        """Queue a message to be spoken. Respects cooldown per category."""
        if not self.enabled:
            return

        now = time.time()
        if category in self.last_spoken:
            if now - self.last_spoken[category] < self.cooldown:
                return  # skip, too soon

        self.last_spoken[category] = now

        # Clear old messages, only keep latest
        while not self.msg_queue.empty():
            try:
                self.msg_queue.get_nowait()
            except queue.Empty:
                break

        self.msg_queue.put(message)

    def coach_on_angles(self, sync_scores, overall):
        """Generate coaching cues based on current sync scores."""
        if overall > 85:
            self.say("Perfect form! Keep it up!", 'praise')
            return

        # Find worst joint
        if not sync_scores:
            return

        worst = min(sync_scores, key=sync_scores.get)
        score = sync_scores[worst]

        if score < 30:
            cues = {
                'right_hip': "Drive your hips! Rotate through the ball!",
                'right_shoulder': "Turn your shoulders! Full unit turn!",
                'right_elbow': "Extend that elbow through contact!",
                'right_wrist': "Check your wrist lag! Let it snap!",
            }
            self.say(cues.get(worst, "Focus on your form!"), worst)

    def announce_phase(self, phase):
        """Announce swing phase transitions."""
        if phase == 'contact':
            self.say("Contact!", 'phase')

    def announce_score(self, score):
        """Announce overall sync score after a swing."""
        if score > 85:
            self.say(f"Great swing! {int(score)} percent sync!", 'score')
        elif score > 60:
            self.say(f"{int(score)} percent. Getting closer!", 'score')
        else:
            self.say(f"{int(score)} percent. Keep working!", 'score')

    def stop(self):
        if self.enabled:
            self.msg_queue.put('__STOP__')
