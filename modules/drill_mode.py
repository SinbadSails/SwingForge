"""
Drill Mode — Focused single-metric training with guided sessions.

The user picks a drill (knee bend, shoulder turn, racket lag).
The app shows ONLY that metric in big text, with a target zone.
After each swing, it gives specific feedback on that ONE thing.
After 10 swings, it shows the improvement trend.

Controls:
    [1-3]   — pick drill (1=knee bend, 2=shoulder turn, 3=racket lag)
    [R]     — restart drill (reset to swing 1)
    [V]     — toggle voice
    [Q/ESC] — quit
"""

import cv2
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.pose_engine import PoseEngine, SKELETON_CONNECTIONS
from core.swing_detector import SwingDetector
from core.voice_coach import VoiceCoach
from core.coaching import CoachingEngine
from core.gesture_detector import GestureDetector

# ── DRILL CATEGORIES ──
DRILL_CATEGORIES = {
    'GROUNDSTROKE': [1, 2, 3, 4, 5],
    'SERVE': [6, 7, 8],
    'FUNDAMENTALS': [9, 10],
}

# Drill definitions — each focuses on ONE metric
DRILLS = {
    # ── Groundstroke drills ──
    1: {
        'name': 'Knee Bend',
        'category': 'GROUNDSTROKE',
        'metric': 'knee_angle',
        'phase': 'loading',
        'target': (115, 140),
        'unit': '°',
        'good_direction': 'lower',
        'instruction': 'Focus on bending your knees during the backswing.',
        'tip_too_high': 'Bend deeper! Drop into your legs before the swing.',
        'tip_too_low': 'Careful — too deep. Stay athletic, not squatting.',
        'tip_good': 'Great knee bend! Power from the ground up.',
        'color': (0, 255, 0),
    },
    2: {
        'name': 'Contact Height',
        'category': 'GROUNDSTROKE',
        'metric': 'contact_height_ratio',
        'phase': 'contact',
        'target': (0.95, 1.25),
        'unit': 'x',
        'good_direction': 'higher',
        'instruction': 'Hit the ball at waist-to-chest height. Not too high, not too low.',
        'tip_too_high': 'Hitting too high — let the ball drop more.',
        'tip_too_low': 'Hitting too low — step in and take it earlier.',
        'tip_good': 'Perfect contact height! Right in the strike zone.',
        'color': (255, 200, 0),
    },
    3: {
        'name': 'Racket Lag',
        'category': 'GROUNDSTROKE',
        'metric': 'racket_lag',
        'phase': 'loading',
        'target': (75, 130),
        'unit': '°',
        'good_direction': 'higher',
        'instruction': 'Let the racket lag behind your elbow on the backswing.',
        'tip_too_high': 'Too far behind — losing control.',
        'tip_too_low': 'Let it drop more! Think waiter\'s tray.',
        'tip_good': 'Perfect lag! That\'s where the whip comes from.',
        'color': (0, 200, 255),
    },
    4: {
        'name': 'Elbow Extension',
        'category': 'GROUNDSTROKE',
        'metric': 'elbow_angle',
        'phase': 'contact',
        'target': (125, 160),
        'unit': '°',
        'good_direction': 'higher',
        'instruction': 'Extend your arm through the ball at contact.',
        'tip_too_high': 'Arm is locked out — keep slight bend for control.',
        'tip_too_low': 'Extend more! Reach through the ball at contact.',
        'tip_good': 'Full extension! That maximizes racket head speed.',
        'color': (255, 100, 255),
    },
    5: {
        'name': 'Follow-Through',
        'category': 'GROUNDSTROKE',
        'metric': 'racket_lag',
        'phase': 'follow_through',
        'target': (30, 80),
        'unit': '°',
        'good_direction': 'lower',
        'instruction': 'Finish with your racket over your opposite shoulder.',
        'tip_too_high': 'Arm stopped early — keep following through!',
        'tip_too_low': 'Nice wrap! Full follow-through.',
        'tip_good': 'Complete follow-through! The swing isn\'t over at contact.',
        'color': (100, 255, 100),
    },
    # ── Serve drills ──
    6: {
        'name': 'Trophy Position',
        'category': 'SERVE',
        'metric': 'racket_lag',
        'phase': 'loading',
        'target': (120, 170),
        'unit': '°',
        'good_direction': 'higher',
        'instruction': 'Get your racket arm UP during the serve wind-up. Think trophy pose.',
        'tip_too_high': 'Arm is too far back — elbow should point up.',
        'tip_too_low': 'Reach higher! Racket arm should be above your head.',
        'tip_good': 'Great trophy position! Ready to explode up.',
        'color': (255, 150, 0),
    },
    7: {
        'name': 'Serve Knee Load',
        'category': 'SERVE',
        'metric': 'knee_angle',
        'phase': 'loading',
        'target': (100, 135),
        'unit': '°',
        'good_direction': 'lower',
        'instruction': 'Bend your knees to load before jumping into the serve.',
        'tip_too_high': 'Bend more! You need leg drive for power.',
        'tip_too_low': 'Good depth but don\'t sit too low — you need to explode up.',
        'tip_good': 'Loaded! That leg drive will add miles per hour.',
        'color': (255, 100, 0),
    },
    8: {
        'name': 'Serve Extension',
        'category': 'SERVE',
        'metric': 'elbow_angle',
        'phase': 'contact',
        'target': (155, 178),
        'unit': '°',
        'good_direction': 'higher',
        'instruction': 'Reach UP at the highest point. Full arm extension on serve.',
        'tip_too_high': 'Perfect reach! Almost fully extended.',
        'tip_too_low': 'Reach higher! Contact should be at full stretch.',
        'tip_good': 'Maximum reach! That\'s the power zone for serves.',
        'color': (255, 50, 0),
    },
    # ── Fundamental drills ──
    9: {
        'name': 'Ready Position',
        'category': 'FUNDAMENTALS',
        'metric': 'knee_angle',
        'phase': 'contact',  # use live measurement, not swing-based
        'target': (130, 155),
        'unit': '°',
        'good_direction': 'lower',
        'instruction': 'Hold your ready position: knees bent, weight on balls of feet.',
        'tip_too_high': 'Bend your knees more! Athletic stance.',
        'tip_too_low': 'Too low — you won\'t be able to move quickly.',
        'tip_good': 'Perfect athletic stance! You\'re ready to react.',
        'color': (200, 200, 200),
    },
    10: {
        'name': 'Split Step',
        'category': 'FUNDAMENTALS',
        'metric': 'knee_angle',
        'phase': 'loading',
        'target': (120, 145),
        'unit': '°',
        'good_direction': 'lower',
        'instruction': 'Practice your split step — small hop, then land in ready position.',
        'tip_too_high': 'Land lower! Split step needs more knee bend on landing.',
        'tip_too_low': 'Good depth — now recover quickly to neutral.',
        'tip_good': 'Great split step! Quick feet start with good landing.',
        'color': (150, 150, 255),
    },
}

SWINGS_PER_DRILL = 10


def draw_drill_hud(frame, drill, current_value, swing_num, swing_history,
                    fps, state, feedback_text, feedback_color):
    """Draw the clean drill HUD — ONE big number + target zone + progress."""
    h, w = frame.shape[:2]

    # ── Top banner: drill name + instruction ──
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.putText(frame, f"DRILL: {drill['name'].upper()}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, drill['color'], 2)
    cv2.putText(frame, drill['instruction'], (15, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # ── Big metric display (center-right) ──
    panel_w, panel_h = 280, 160
    px = w - panel_w - 15
    py = 70
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), drill['color'], 2)

    # Current value — BIG
    if current_value is not None and state != 'IDLE':
        val_text = f"{current_value:.0f}{drill['unit']}"
        low, high = drill['target']
        if low <= current_value <= high:
            val_color = (0, 255, 0)  # green = in target
        elif (drill['good_direction'] == 'lower' and current_value > high) or \
             (drill['good_direction'] == 'higher' and current_value < low):
            val_color = (0, 0, 255)  # red = wrong direction
        else:
            val_color = (0, 255, 255)  # yellow = close

        cv2.putText(frame, val_text, (px + 30, py + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, val_color, 3)
    else:
        cv2.putText(frame, "---", (px + 60, py + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (100, 100, 100), 3)

    # Target zone
    low, high = drill['target']
    cv2.putText(frame, f"Target: {low}-{high}{drill['unit']}", (px + 15, py + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # State indicator
    state_text = {'IDLE': 'Swing!', 'SWINGING': 'SWINGING...', 'SCORED': ''}
    cv2.putText(frame, state_text.get(state, ''), (px + 15, py + 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 255, 0), 1)

    # ── Progress bar (bottom) ──
    bar_y = h - 80
    bar_h = 35
    cv2.rectangle(frame, (15, bar_y), (w - 15, bar_y + bar_h), (30, 30, 30), -1)

    # Swing blocks
    block_w = (w - 40) // SWINGS_PER_DRILL
    for i in range(SWINGS_PER_DRILL):
        bx = 20 + i * block_w
        if i < len(swing_history):
            score = swing_history[i]
            low, high = drill['target']
            if low <= score <= high:
                bc = (0, 200, 0)  # green = in target
            else:
                bc = (0, 0, 200)  # red = out of target
            cv2.rectangle(frame, (bx, bar_y + 3), (bx + block_w - 4, bar_y + bar_h - 3), bc, -1)
            cv2.putText(frame, f"{score:.0f}", (bx + 5, bar_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        elif i == len(swing_history):
            # Current swing slot
            cv2.rectangle(frame, (bx, bar_y + 3), (bx + block_w - 4, bar_y + bar_h - 3),
                          drill['color'], 2)
            cv2.putText(frame, f"#{i+1}", (bx + 8, bar_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, drill['color'], 1)
        else:
            cv2.rectangle(frame, (bx, bar_y + 3), (bx + block_w - 4, bar_y + bar_h - 3),
                          (50, 50, 50), 1)

    # Progress text
    cv2.putText(frame, f"Swing {min(swing_num, SWINGS_PER_DRILL)}/{SWINGS_PER_DRILL}",
                (15, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # ── Feedback text (after swing) ──
    if feedback_text:
        fb_y = h - 120
        cv2.rectangle(frame, (15, fb_y - 5), (w - 15, fb_y + 25), (20, 20, 20), -1)
        cv2.putText(frame, feedback_text, (20, fb_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, feedback_color, 2)

    # ── FPS + controls ──
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 80, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
    cv2.putText(frame, "Hands UP=Pause | T-pose=Resume | Say drill name or press [1-0] | [H]ome [Q]uit",
                (15, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)

    return frame


def draw_session_summary(frame, drill, swing_history):
    """Draw the end-of-drill summary screen."""
    h, w = frame.shape[:2]
    frame[:] = (20, 20, 20)  # dark background

    cv2.putText(frame, f"DRILL COMPLETE: {drill['name'].upper()}", (w//2 - 200, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, drill['color'], 2)

    if not swing_history:
        cv2.putText(frame, "No swings recorded.", (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        return frame

    low, high = drill['target']
    in_target = sum(1 for v in swing_history if low <= v <= high)
    avg = np.mean(swing_history)
    best_idx = int(np.argmin(swing_history) if drill['good_direction'] == 'lower'
                    else np.argmax(swing_history))

    # Stats
    y = 110
    cv2.putText(frame, f"Swings: {len(swing_history)}", (60, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 35
    cv2.putText(frame, f"In target zone: {in_target}/{len(swing_history)} ({100*in_target/len(swing_history):.0f}%)",
                (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if in_target > len(swing_history)//2 else (0, 0, 255), 1)
    y += 35
    cv2.putText(frame, f"Average: {avg:.1f}{drill['unit']}  (target: {low}-{high})",
                (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 35
    cv2.putText(frame, f"Best swing: #{best_idx+1} ({swing_history[best_idx]:.0f}{drill['unit']})",
                (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (204, 255, 0), 1)

    # Improvement trend
    if len(swing_history) >= 4:
        first_half = np.mean(swing_history[:len(swing_history)//2])
        second_half = np.mean(swing_history[len(swing_history)//2:])

        if drill['good_direction'] == 'lower':
            improved = second_half < first_half
            change = first_half - second_half
        else:
            improved = second_half > first_half
            change = second_half - first_half

        y += 45
        trend_color = (0, 255, 0) if improved else (0, 0, 255)
        trend_text = f"{'IMPROVED' if improved else 'DECLINED'}: {abs(change):.1f}{drill['unit']} {'better' if improved else 'worse'} in second half"
        cv2.putText(frame, trend_text, (60, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, trend_color, 2)

    # Visual graph of swings
    graph_y = h - 200
    graph_h = 120
    graph_w = w - 120
    cv2.rectangle(frame, (60, graph_y), (60 + graph_w, graph_y + graph_h), (40, 40, 40), -1)

    # Target zone band
    all_vals = swing_history + [low, high]
    vmin, vmax = min(all_vals) - 10, max(all_vals) + 10
    if vmax - vmin < 20:
        vmin -= 10
        vmax += 10

    target_y1 = graph_y + int((1 - (high - vmin) / (vmax - vmin)) * graph_h)
    target_y2 = graph_y + int((1 - (low - vmin) / (vmax - vmin)) * graph_h)
    cv2.rectangle(frame, (60, target_y1), (60 + graph_w, target_y2), (30, 60, 30), -1)

    # Plot swings
    step = graph_w // max(len(swing_history), 1)
    for i, val in enumerate(swing_history):
        x = 60 + i * step + step // 2
        y_pt = graph_y + int((1 - (val - vmin) / (vmax - vmin)) * graph_h)
        in_zone = low <= val <= high
        color = (0, 255, 0) if in_zone else (0, 0, 255)
        cv2.circle(frame, (x, y_pt), 6, color, -1)
        if i > 0:
            prev_val = swing_history[i-1]
            prev_x = 60 + (i-1) * step + step // 2
            prev_y = graph_y + int((1 - (prev_val - vmin) / (vmax - vmin)) * graph_h)
            cv2.line(frame, (prev_x, prev_y), (x, y_pt), (150, 150, 150), 1)

    cv2.putText(frame, "Say 'restart' or drill name | Press [R] or [H]ome or [Q]uit",
                (w//2 - 250, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return frame


def draw_home_screen(frame):
    """Draw the home/menu screen where user picks a drill."""
    h, w = frame.shape[:2]
    frame[:] = (20, 20, 20)

    cv2.putText(frame, "SWINGFORGE DRILL MODE", (w//2 - 180, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (204, 255, 0), 2)
    cv2.putText(frame, "Say a drill name or press a key to start", (w//2 - 200, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    y = 130
    for cat_name, drill_ids in DRILL_CATEGORIES.items():
        cv2.putText(frame, cat_name, (60, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (204, 255, 0), 2)
        y += 30
        for did in drill_ids:
            d = DRILLS[did]
            key_label = str(did) if did < 10 else '0'
            cv2.putText(frame, f"  [{key_label}]  {d['name']}", (80, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, d['color'], 1)
            cv2.putText(frame, f"  target: {d['target'][0]}-{d['target'][1]}{d['unit']}",
                        (350, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            y += 25
        y += 15

    cv2.putText(frame, 'Say: "knee", "shoulder", "trophy", "serve", "elbow", "ready"...',
                (60, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(frame, "Press a number key or say a drill name to begin  |  [Q] Quit",
                (60, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return frame


def run_drill_mode(playing_hand='right'):
    """Main drill mode loop."""
    print("\n  SwingForge Drill Mode — Loading...")
    pose_engine = PoseEngine()
    voice_coach = VoiceCoach(enabled=True, rate=170)
    swing_detector = SwingDetector(fps=30)
    gesture_detector = GestureDetector()

    # Voice commands — Vosk offline recognition (no internet needed)
    from core.voice_commands import VoiceCommands
    voice_cmds = VoiceCommands(enabled=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # State
    current_drill_id = None  # None = home screen
    drill = None
    swing_history = []
    feedback_text = ""
    feedback_color = (200, 200, 200)
    feedback_time = 0
    screen = 'HOME'  # HOME, DRILL, SUMMARY, PAUSED
    live_value = None
    swing_state = 'IDLE'

    frame_count = 0
    fps = 0
    fps_timer = time.time()

    print(f"\n  SwingForge Drill Mode")
    print(f"  ─────────────────────")
    print(f"  Say a drill name or press a number key:")
    print(f"    [1] Knee  [2] Shoulder  [3] Lag  [4] Elbow  [5] Follow")
    print(f"    [6] Trophy  [7] Serve Knee  [8] Serve Ext  [9] Ready  [0] Split")
    print(f"  Hands UP = Pause  |  T-pose = Resume  |  [H] Home  |  [Q] Quit\n")

    voice_coach.say("Drill mode. Say a drill name to begin.")

    cv2.namedWindow('SwingForge Drill', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SwingForge Drill', 1280, 720)

    dropped = 0

    def switch_drill(new_id):
        """Helper to switch to a drill and reset state."""
        nonlocal current_drill_id, drill, swing_history, swing_detector
        nonlocal feedback_text, screen, swing_state, gesture_detector
        current_drill_id = new_id
        drill = DRILLS[new_id]
        swing_history = []
        swing_detector = SwingDetector(fps=30)
        gesture_detector = GestureDetector()
        feedback_text = ""
        screen = 'DRILL'
        swing_state = 'IDLE'
        voice_coach.say(f"{drill['name']}. {drill['instruction']}")
        print(f"  → Drill: {drill['name']}")

    def check_voice_and_keys(key):
        """Check voice commands and keyboard on ANY screen. Returns True if should break."""
        nonlocal screen
        # Voice commands — always checked
        vcmd = voice_cmds.get_command()
        if vcmd == 'quit' or key == ord('q') or key == 27:
            return True
        if vcmd == 'pause' or key == ord(' '):
            if screen == 'DRILL':
                screen = 'PAUSED'
                gesture_detector.is_paused = True
                voice_coach.say("Paused.")
            elif screen == 'PAUSED':
                screen = 'DRILL'
                gesture_detector.is_paused = False
                voice_coach.say("Resumed.")
        elif vcmd == 'restart' or key == ord('r'):
            if drill:
                switch_drill(current_drill_id)
        elif vcmd == 'home' or key == ord('h'):
            screen = 'HOME'
            voice_coach.say("Home screen.")
        elif vcmd == 'next_drill':
            if drill:
                drill_ids = sorted(DRILLS.keys())
                idx = drill_ids.index(current_drill_id) if current_drill_id in drill_ids else 0
                switch_drill(drill_ids[(idx + 1) % len(drill_ids)])
        elif isinstance(vcmd, int) and vcmd in DRILLS:
            switch_drill(vcmd)
        elif isinstance(key, int):
            # Number key pressed
            if key in [ord(str(i)) for i in range(1, 10)] + [ord('0')]:
                new_id = (key - ord('0')) if key != ord('0') else 10
                if new_id in DRILLS:
                    switch_drill(new_id)
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            dropped += 1
            if dropped > 30:
                break
            continue
        dropped = 0

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now

        # ── HOME SCREEN ──
        if screen == 'HOME':
            frame = draw_home_screen(frame)
            cv2.imshow('SwingForge Drill', frame)
            key = cv2.waitKey(1) & 0xFF
            if check_voice_and_keys(key):
                break
            continue

        # ── SUMMARY SCREEN ──
        if screen == 'SUMMARY':
            frame = draw_session_summary(frame, drill, swing_history)
            cv2.imshow('SwingForge Drill', frame)
            key = cv2.waitKey(1) & 0xFF
            if check_voice_and_keys(key):
                break
            continue

        # ── PAUSED SCREEN ──
        if screen == 'PAUSED':
            try:
                pause_kp = pose_engine.extract_keypoints(frame)
            except Exception:
                pause_kp = None

            pause_gesture = gesture_detector.update(pause_kp)
            pause_hint = gesture_detector.get_gesture_hint(pause_kp)

            cv2.rectangle(frame, (w//2 - 140, h//2 - 50), (w//2 + 140, h//2 + 50), (20, 20, 20), -1)
            cv2.rectangle(frame, (w//2 - 140, h//2 - 50), (w//2 + 140, h//2 + 50), (204, 255, 0), 2)
            cv2.putText(frame, "PAUSED", (w//2 - 55, h//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (204, 255, 0), 2)
            cv2.putText(frame, "T-pose or SPACE to resume | Say drill name to switch",
                        (w//2 - 210, h//2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

            if pause_hint:
                cv2.putText(frame, pause_hint, (w//2 - 120, h//2 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (204, 255, 0), 1)

            if pause_gesture == 'resume':
                screen = 'DRILL'
                gesture_detector.is_paused = False
                voice_coach.say("Resumed.")

            cv2.imshow('SwingForge Drill', frame)
            key = cv2.waitKey(1) & 0xFF
            if check_voice_and_keys(key):
                break
            continue

        try:
            user_kp = pose_engine.extract_keypoints(frame)
        except Exception:
            user_kp = None

        angles = pose_engine.get_joint_angles(user_kp, side=playing_hand) if user_kp else None

        # Draw skeleton
        if user_kp:
            for start, end in SKELETON_CONNECTIONS:
                if start in user_kp and end in user_kp:
                    p1 = (int(user_kp[start][0]), int(user_kp[start][1]))
                    p2 = (int(user_kp[end][0]), int(user_kp[end][1]))
                    vis1 = user_kp[start][3] if len(user_kp[start]) > 3 else 1.0
                    vis2 = user_kp[end][3] if len(user_kp[end]) > 3 else 1.0
                    if vis1 > 0.5 and vis2 > 0.5:
                        cv2.line(frame, p1, p2, (0, 255, 127), 3)
            for name, kp in user_kp.items():
                vis = kp[3] if len(kp) > 3 else 1.0
                if vis > 0.5:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 255), -1)

        # Track the focused metric in real-time
        live_value = None
        if angles and drill['metric'] in angles:
            live_value = angles[drill['metric']]

        # Gesture detection — only pause gesture now
        gesture = gesture_detector.update(user_kp)
        gesture_hint = gesture_detector.get_gesture_hint(user_kp)

        if gesture_hint:
            cv2.putText(frame, gesture_hint, (w // 2 - 150, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 255, 0), 2)

        if gesture == 'pause':
            screen = 'PAUSED'
            gesture_detector.is_paused = True
            voice_coach.say("Paused.")
        # (next_drill and restart handled by voice commands via check_voice_and_keys)

        # (Voice commands + keys handled by check_voice_and_keys at end of loop)

        # Swing detection (with walking filter via hip_pos)
        wrist_pos = user_kp['right_wrist'][:2] if user_kp and 'right_wrist' in user_kp else None
        torso_len = None
        hip_pos = None
        if user_kp:
            hy = (user_kp['left_hip'][1] + user_kp['right_hip'][1]) / 2
            hx = (user_kp['left_hip'][0] + user_kp['right_hip'][0]) / 2
            sy = (user_kp['left_shoulder'][1] + user_kp['right_shoulder'][1]) / 2
            torso_len = abs(hy - sy)
            hip_pos = (hx, hy)

        swing_result = swing_detector.update(wrist_pos, frame, user_kp, angles,
                                              torso_length=torso_len, hip_pos=hip_pos)

        if swing_detector.is_swinging:
            state = 'SWINGING'
        elif isinstance(swing_result, dict) and swing_result.get('type') == 'swing_complete':
            state = 'SCORED'
            feedback_time = now

            # Get the metric value for this swing
            # For loading-phase metrics, use the best value from the swing
            contact_a = swing_result.get('contact_angles', {})
            loading_a = swing_result.get('loading_angles', {})

            if drill['phase'] == 'loading' and loading_a and drill['metric'] in loading_a:
                metric_val = loading_a[drill['metric']]
            elif contact_a and drill['metric'] in contact_a:
                metric_val = contact_a[drill['metric']]
            else:
                metric_val = live_value  # fallback

            if metric_val is not None:
                swing_history.append(metric_val)
                swing_num = len(swing_history)

                low, high = drill['target']
                if low <= metric_val <= high:
                    feedback_text = f"Swing #{swing_num}: {metric_val:.0f}{drill['unit']} — {drill['tip_good']}"
                    feedback_color = (0, 255, 0)
                    voice_coach.say(f"Swing {swing_num}. {metric_val:.0f} degrees. {drill['tip_good']}", 'swing')
                elif (drill['good_direction'] == 'lower' and metric_val > high) or \
                     (drill['good_direction'] == 'higher' and metric_val < low):
                    feedback_text = f"Swing #{swing_num}: {metric_val:.0f}{drill['unit']} — {drill['tip_too_low'] if drill['good_direction'] == 'higher' else drill['tip_too_high']}"
                    feedback_color = (0, 0, 255)
                    tip = drill['tip_too_low'] if drill['good_direction'] == 'higher' else drill['tip_too_high']
                    voice_coach.say(f"Swing {swing_num}. {metric_val:.0f} degrees. {tip}", 'swing')
                else:
                    feedback_text = f"Swing #{swing_num}: {metric_val:.0f}{drill['unit']} — Close! Keep adjusting."
                    feedback_color = (0, 255, 255)
                    voice_coach.say(f"Swing {swing_num}. {metric_val:.0f} degrees. Close.", 'swing')

                # Check if drill is complete
                if swing_num >= SWINGS_PER_DRILL:
                    screen = "SUMMARY"
                    in_target = sum(1 for v in swing_history if low <= v <= high)
                    voice_coach.say(
                        f"Drill complete. {in_target} out of {SWINGS_PER_DRILL} in the target zone. "
                        f"Average: {np.mean(swing_history):.0f} degrees.",
                        'summary'
                    )

        elif not swing_detector.is_swinging:
            if now - feedback_time > 3.0:
                state = 'IDLE'
                feedback_text = ""

        # Clear old feedback
        if feedback_text and now - feedback_time > 4.0:
            feedback_text = ""

        # Draw drill HUD
        frame = draw_drill_hud(frame, drill, live_value,
                                len(swing_history) + 1, swing_history,
                                fps, state, feedback_text, feedback_color)

        cv2.imshow('SwingForge Drill', frame)

        key = cv2.waitKey(1) & 0xFF
        if check_voice_and_keys(key):
            break

    # Final summary
    voice_coach.stop()
    if swing_history:
        low, high = drill['target']
        in_target = sum(1 for v in swing_history if low <= v <= high)
        print(f"\n  ── Drill Summary: {drill['name']} ──")
        print(f"  Swings: {len(swing_history)}")
        print(f"  In target: {in_target}/{len(swing_history)}")
        print(f"  Average: {np.mean(swing_history):.1f}{drill['unit']}")
        print(f"  Values: {', '.join(f'{v:.0f}' for v in swing_history)}")

    cap.release()
    cv2.destroyAllWindows()
    pose_engine.release()


if __name__ == '__main__':
    run_drill_mode()
