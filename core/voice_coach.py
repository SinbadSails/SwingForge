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
        self.cooldown = 8.0  # 8 seconds between same category (was 3 — too spammy)
        self.global_cooldown = 4.0  # minimum 4 seconds between ANY speech
        self.last_any_speech = 0
        self.last_tip_joint = None  # track what we said last to avoid repeats
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
        """Queue a message to be spoken. Respects cooldowns."""
        if not self.enabled:
            return

        now = time.time()

        # Global cooldown — don't talk too often
        if now - self.last_any_speech < self.global_cooldown:
            return

        # Per-category cooldown
        if category in self.last_spoken:
            if now - self.last_spoken[category] < self.cooldown:
                return

        self.last_spoken[category] = now
        self.last_any_speech = now

        # Clear old messages, only keep latest
        while not self.msg_queue.empty():
            try:
                self.msg_queue.get_nowait()
            except queue.Empty:
                break

        self.msg_queue.put(message)

    def coach_on_angles(self, sync_scores, overall):
        """Generate coaching cues. Only speaks if something is genuinely wrong
        and doesn't repeat the same tip twice in a row."""
        if not sync_scores:
            return

        # Don't coach if overall is decent
        if overall > 75:
            return

        # Find worst joint
        worst = min(sync_scores, key=sync_scores.get)
        score = sync_scores[worst]

        # Only speak if it's actually bad (below 40%) and different from last tip
        if score < 40 and worst != self.last_tip_joint:
            self.last_tip_joint = worst
            cues = {
                'right_hip': "Rotate your hips!",
                'right_shoulder': "Turn your shoulders more!",
                'right_elbow': "Extend that elbow!",
                'right_wrist': "More wrist lag!",
            }
            self.say(cues.get(worst, "Focus on form!"), worst)

    def announce_score(self, score):
        """Announce score after a detected swing."""
        if score > 85:
            self.say(f"{int(score)} percent! Great swing!", 'score')
        elif score > 60:
            self.say(f"{int(score)} percent.", 'score')
        else:
            self.say(f"{int(score)} percent. Keep working.", 'score')

    def stop(self):
        if self.enabled:
            self.msg_queue.put('__STOP__')
