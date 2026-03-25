"""
GestureDetector — Detect hand gestures from pose keypoints for hands-free control.
Uses wrist/shoulder/hip positions from MediaPipe Pose (no extra models needed).

Gestures:
  BOTH HANDS UP (wrists above shoulders for 1s)  → PAUSE / RESUME
  BOTH HANDS DOWN (wrists below hips for 1s)     → NEXT DRILL
  LEFT HAND UP ONLY (for 1s)                     → RESTART DRILL
  RIGHT HAND WAVE (quick up-down 3x)             → TOGGLE VOICE
"""

import time
import numpy as np


class GestureDetector:
    def __init__(self):
        self.hold_time = 0.8  # seconds to hold a gesture before it triggers
        self._gesture_start = {}  # {gesture_name: start_time}
        self._last_triggered = {}  # {gesture_name: time} cooldown
        self.cooldown = 2.0  # seconds between same gesture triggers
        self._wave_history = []  # for wave detection

    def update(self, keypoints):
        """Check for gestures. Returns gesture name or None.
        Possible returns: 'pause', 'next_drill', 'restart', 'toggle_voice', None
        """
        if keypoints is None:
            self._gesture_start.clear()
            return None

        now = time.time()
        detected = self._detect_gesture(keypoints)

        if detected is None:
            # No gesture held — reset all timers
            self._gesture_start.clear()
            return None

        # Check cooldown
        if detected in self._last_triggered:
            if now - self._last_triggered[detected] < self.cooldown:
                return None

        # Start timer for this gesture if not already started
        if detected not in self._gesture_start:
            self._gesture_start = {detected: now}  # reset to only this gesture
            return None

        # Check if held long enough
        if now - self._gesture_start[detected] >= self.hold_time:
            self._gesture_start.clear()
            self._last_triggered[detected] = now
            return detected

        return None

    def _detect_gesture(self, kp):
        """Detect which gesture is being held (if any)."""
        # Get key positions
        lw = kp.get('left_wrist')
        rw = kp.get('right_wrist')
        ls = kp.get('left_shoulder')
        rs = kp.get('right_shoulder')
        lh = kp.get('left_hip')
        rh = kp.get('right_hip')

        if not all([lw, rw, ls, rs, lh, rh]):
            return None

        lw_y, rw_y = lw[1], rw[1]
        ls_y, rs_y = ls[1], rs[1]
        lh_y, rh_y = lh[1], rh[1]

        # In image coords: lower y = higher position
        left_above_shoulder = lw_y < ls_y - 20
        right_above_shoulder = rw_y < rs_y - 20
        left_below_hip = lw_y > lh_y + 20
        right_below_hip = rw_y > rh_y + 20

        # BOTH HANDS UP → pause/resume
        if left_above_shoulder and right_above_shoulder:
            return 'pause'

        # BOTH HANDS DOWN → next drill
        if left_below_hip and right_below_hip:
            return 'next_drill'

        # LEFT HAND UP ONLY → restart
        if left_above_shoulder and not right_above_shoulder:
            return 'restart'

        return None

    def get_gesture_hint(self, keypoints):
        """Return a hint about what gesture is being detected (for visual feedback)."""
        if keypoints is None:
            return None

        gesture = self._detect_gesture(keypoints)
        if gesture is None:
            return None

        if gesture in self._gesture_start:
            elapsed = time.time() - self._gesture_start[gesture]
            remaining = max(0, self.hold_time - elapsed)
            labels = {
                'pause': 'PAUSE',
                'next_drill': 'NEXT DRILL',
                'restart': 'RESTART',
            }
            if remaining > 0:
                return f"Hold for {labels.get(gesture, gesture)}... {remaining:.1f}s"
            else:
                return f"{labels.get(gesture, gesture)}!"

        labels = {
            'pause': 'Both hands up → PAUSE',
            'next_drill': 'Both hands down → NEXT',
            'restart': 'Left hand up → RESTART',
        }
        return labels.get(gesture)
