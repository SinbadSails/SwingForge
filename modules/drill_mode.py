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

# Drill definitions — each focuses on ONE metric
DRILLS = {
    1: {
        'name': 'Knee Bend',
        'metric': 'knee_angle',
        'phase': 'loading',  # measure during loading
        'target': (115, 140),  # ideal range in degrees
        'unit': '°',
        'good_direction': 'lower',  # lower angle = more bend = better
        'instruction': 'Focus on bending your knees during the backswing.',
        'tip_too_high': 'Bend deeper! Drop into your legs before the swing.',
        'tip_too_low': 'Careful — too deep. Stay athletic, not squatting.',
        'tip_good': 'Great knee bend! That loading will give you power.',
        'color': (0, 255, 0),  # green
    },
    2: {
        'name': 'Shoulder Turn',
        'metric': 'shoulder_angle',
        'phase': 'loading',
        'target': (65, 100),
        'unit': '°',
        'good_direction': 'higher',
        'instruction': 'Focus on rotating your shoulders fully during preparation.',
        'tip_too_high': 'You\'re over-rotating. Keep it controlled.',
        'tip_too_low': 'Turn more! Get your non-dominant shoulder pointing at the ball.',
        'tip_good': 'Excellent shoulder turn! Full coil.',
        'color': (255, 200, 0),  # yellow
    },
    3: {
        'name': 'Racket Lag',
        'metric': 'racket_lag',
        'phase': 'loading',
        'target': (75, 130),
        'unit': '°',
        'good_direction': 'higher',
        'instruction': 'Focus on letting the racket lag behind your elbow.',
        'tip_too_high': 'Racket is too far behind — you\'ll lose control.',
        'tip_too_low': 'Let the racket drop more! Think waiter\'s tray position.',
        'tip_good': 'Perfect lag! That\'s where the power comes from.',
        'color': (0, 200, 255),  # cyan
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
    cv2.putText(frame, "[1] Knee  [2] Shoulder  [3] Lag  [R] Restart  [Q] Quit",
                (15, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

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

    cv2.putText(frame, "Press [R] to restart or [Q] to quit",
                (w//2 - 180, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    return frame


def run_drill_mode(playing_hand='right'):
    """Main drill mode loop."""
    print("\n  SwingForge Drill Mode — Loading...")
    pose_engine = PoseEngine()
    voice_coach = VoiceCoach(enabled=True, rate=170)
    swing_detector = SwingDetector(fps=30)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # State
    current_drill_id = 1
    drill = DRILLS[current_drill_id]
    swing_history = []
    feedback_text = ""
    feedback_color = (200, 200, 200)
    feedback_time = 0
    show_summary = False
    live_value = None
    state = 'IDLE'

    frame_count = 0
    fps = 0
    fps_timer = time.time()

    print(f"\n  SwingForge Drill Mode")
    print(f"  ─────────────────────")
    print(f"  [1] Knee Bend  [2] Shoulder Turn  [3] Racket Lag")
    print(f"  [R] Restart  [V] Voice  [Q] Quit\n")

    voice_coach.say(f"Drill mode. Working on {drill['name']}. {drill['instruction']}")

    cv2.namedWindow('SwingForge Drill', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SwingForge Drill', 1280, 720)

    dropped = 0

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

        if show_summary:
            frame = draw_session_summary(frame, drill, swing_history)
            cv2.imshow('SwingForge Drill', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                swing_history = []
                swing_detector = SwingDetector(fps=30)
                show_summary = False
                feedback_text = ""
                voice_coach.say(f"Restarting {drill['name']} drill.")
            elif key == ord('q') or key == 27:
                break
            elif key in [ord('1'), ord('2'), ord('3')]:
                current_drill_id = key - ord('0')
                drill = DRILLS[current_drill_id]
                swing_history = []
                swing_detector = SwingDetector(fps=30)
                show_summary = False
                feedback_text = ""
                voice_coach.say(f"Switching to {drill['name']} drill. {drill['instruction']}")
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

        # Swing detection
        wrist_pos = user_kp['right_wrist'][:2] if user_kp and 'right_wrist' in user_kp else None
        torso_len = None
        if user_kp:
            hy = (user_kp['left_hip'][1] + user_kp['right_hip'][1]) / 2
            sy = (user_kp['left_shoulder'][1] + user_kp['right_shoulder'][1]) / 2
            torso_len = abs(hy - sy)

        swing_result = swing_detector.update(wrist_pos, frame, user_kp, angles,
                                              torso_length=torso_len)

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
                    show_summary = True
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
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            swing_history = []
            swing_detector = SwingDetector(fps=30)
            feedback_text = ""
            voice_coach.say(f"Restarting {drill['name']} drill.")
        elif key == ord('v'):
            voice_coach.enabled = not voice_coach.enabled
            print(f"  Voice: {'ON' if voice_coach.enabled else 'OFF'}")
        elif key in [ord('1'), ord('2'), ord('3')]:
            current_drill_id = key - ord('0')
            drill = DRILLS[current_drill_id]
            swing_history = []
            swing_detector = SwingDetector(fps=30)
            show_summary = False
            feedback_text = ""
            voice_coach.say(f"Switching to {drill['name']}. {drill['instruction']}")
            print(f"  Drill: {drill['name']}")

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
