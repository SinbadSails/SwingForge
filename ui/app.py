"""
SwingForge Gradio UI — Dark-themed tennis swing analysis interface.
Tabs: Upload & Analyze | Live Webcam | Pro Compare | Swing Plane | Ball Flight
"""

import gradio as gr
import cv2
import numpy as np
import json
import os
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.pose_engine import PoseEngine
from core.swing_classifier import SwingClassifier
from core.coaching import CoachingEngine
from core.physics import BallPhysics
from utils import read_video, save_video

# Initialize engines
pose_engine = PoseEngine()
swing_classifier = SwingClassifier()
coaching_engine = CoachingEngine()
ball_physics = BallPhysics()

# Available pros
PROS = {
    'Djokovic (Forehand)': 'djokovic_forehand',
    'Alcaraz (Forehand)': 'alcaraz_forehand',
    'Federer (Serve)': 'federer_serve',
    'Nadal (Forehand)': 'nadal_forehand',
    'Medvedev (Backhand)': 'medvedev_backhand',
}

PRO_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pros')


# ─── MODULE 1: Upload & Analyze ───────────────────────────────────────

def analyze_video(video_path, playing_hand='right'):
    """Full swing analysis pipeline on uploaded video."""
    if video_path is None:
        return None, "No video uploaded.", "{}"

    frames = read_video(video_path)
    if not frames:
        return None, "Could not read video.", "{}"

    fps = 30  # default assumption
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

    # Extract keypoints for all frames
    keypoints_seq = pose_engine.extract_keypoints_batch(frames)

    # Calculate velocities and accelerations
    velocities = pose_engine.get_wrist_velocity(keypoints_seq, fps=fps)
    accelerations = pose_engine.get_wrist_acceleration(velocities, fps=fps)

    # Detect swing phases
    phases = swing_classifier.detect_phases(keypoints_seq, velocities, accelerations)

    # Find contact frame
    contact_idx = swing_classifier.get_contact_frame_index(phases)
    if contact_idx is None:
        # Fallback: use peak velocity frame
        contact_idx = int(np.argmax(velocities))

    # Get angles at contact
    side = playing_hand
    contact_angles = pose_engine.get_joint_angles(keypoints_seq[contact_idx], side=side)

    # Get BEST loading angles from loading phase (not preparation — too early).
    # Falls back to last 30% of preparation if no loading frames detected.
    load_phases = [i for i, p in enumerate(phases) if p == 'loading']
    if not load_phases:
        prep_frames = [i for i, p in enumerate(phases) if p == 'preparation']
        if prep_frames:
            load_phases = prep_frames[int(len(prep_frames) * 0.7):]  # last 30% of prep
    loading_angles = None
    if load_phases:
        best = {'shoulder_angle': 0, 'knee_angle': 180, 'racket_lag': 0}
        for fi in load_phases:
            if keypoints_seq[fi] is not None:
                a = pose_engine.get_joint_angles(keypoints_seq[fi], side=side)
                if a:
                    if a['shoulder_angle'] > best['shoulder_angle']:
                        best['shoulder_angle'] = a['shoulder_angle']
                    if a['knee_angle'] < best['knee_angle']:  # lower = more bent
                        best['knee_angle'] = a['knee_angle']
                    if a['racket_lag'] > best['racket_lag']:
                        best['racket_lag'] = a['racket_lag']
        loading_angles = best

    # Classify stroke
    stroke_type = swing_classifier.classify_stroke(keypoints_seq[contact_idx])

    # Check follow-through
    follow_through = swing_classifier.detect_follow_through_completion(keypoints_seq, phases)

    # Score the swing (phase-aware: contact metrics at contact, loading metrics at loading)
    scores = coaching_engine.score_swing(contact_angles, stroke_type, follow_through, loading_angles)
    angles = contact_angles  # for display

    # Generate coaching report (pass both angle sets for correct phase display)
    report = coaching_engine.generate_coaching_report(
        scores, contact_angles, stroke_type, loading_angles=loading_angles
    )

    # Check for injury warnings
    warnings = coaching_engine.get_injury_warnings(angles, stroke_type)
    if warnings:
        report += "\n\nINJURY WARNINGS:\n" + "\n".join(f"  - {w}" for w in warnings)

    # Annotate video frames
    annotated_frames = []
    for i, frame in enumerate(frames):
        f = frame.copy()
        kp = keypoints_seq[i]

        # Draw skeleton
        pose_engine.draw_skeleton(f, kp, color=(0, 255, 127), thickness=2)

        # Phase label
        phase = phases[i] if i < len(phases) else 'idle'
        color = {
            'idle': (128, 128, 128), 'preparation': (255, 255, 0),
            'loading': (0, 165, 255), 'contact': (0, 0, 255),
            'follow_through': (0, 255, 0),
        }.get(phase, (128, 128, 128))
        cv2.putText(f, f"Phase: {phase.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show angles at contact
        if i == contact_idx and angles:
            y_off = 60
            for key, val in angles.items():
                if key != 'contact_height_ratio':
                    nice = key.replace('_', ' ').title()
                    cv2.putText(f, f"{nice}: {val:.1f} deg", (10, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_off += 22

        # SwingScore overlay
        if scores:
            cv2.putText(f, f"SwingScore: {scores['overall']}/100",
                        (f.shape[1] - 280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (204, 255, 0), 2)

        annotated_frames.append(f)

    # Save annotated video
    out_path = tempfile.mktemp(suffix='.mp4')
    h, w = annotated_frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (w, h))
    for f in annotated_frames:
        writer.write(f)
    writer.release()

    # Build JSON report
    json_report = json.dumps({
        'stroke_type': stroke_type,
        'contact_frame': contact_idx,
        'scores': scores,
        'angles': {k: round(v, 1) for k, v in angles.items()} if angles else {},
        'follow_through_complete': follow_through,
        'phases_summary': {p: phases.count(p) for p in set(phases)},
    }, indent=2)

    return out_path, report, json_report


# ─── MODULE 2: Live Webcam Analysis ──────────────────────────────────

def process_webcam_frame(frame, playing_hand='right'):
    """Process a single webcam frame for real-time analysis.
    Gradio 6 sends frames as RGB numpy arrays (or PIL Image).
    """
    if frame is None:
        return None

    try:
        # Handle different input types from Gradio
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        # Make a copy to draw on (frame is RGB from Gradio)
        output = frame.copy()
        h, w = output.shape[:2]

        # MediaPipe needs BGR input (our extract_keypoints converts BGR->RGB internally)
        frame_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        kp = pose_engine.extract_keypoints(frame_bgr)

        if kp is None:
            # Draw "no pose" message directly on RGB frame
            cv2.putText(output, "No pose detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 50), 2)
            cv2.putText(output, "Step back so camera sees head to knees", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 150), 1)
            return output

        # Draw skeleton in RGB color space (swap B and R for RGB)
        for start, end in [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ]:
            if start in kp and end in kp:
                p1 = (int(kp[start][0]), int(kp[start][1]))
                p2 = (int(kp[end][0]), int(kp[end][1]))
                cv2.line(output, p1, p2, (0, 255, 127), 3)

        # Draw joint dots
        for name, point in kp.items():
            cv2.circle(output, (int(point[0]), int(point[1])), 5, (0, 255, 255), -1)

        # Get angles
        angles = pose_engine.get_joint_angles(kp, side=playing_hand)
        if angles:
            # Dark box for text readability
            cv2.rectangle(output, (5, 5), (300, 170), (20, 20, 20), -1)
            cv2.rectangle(output, (5, 5), (300, 170), (200, 255, 0), 2)

            y_off = 30
            for key, val in angles.items():
                if key != 'contact_height_ratio':
                    nice = key.replace('_', ' ').title()
                    cv2.putText(output, f"{nice}: {val:.1f}", (12, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 0), 2)
                    y_off += 28

        return output

    except Exception as e:
        print(f"Webcam error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return frame


# ─── MODULE 3: Pro Compare ───────────────────────────────────────────

def compare_to_pro(video_path, pro_selection, playing_hand='right'):
    """Analyze video and overlay pro skeleton comparison at contact frame."""
    if video_path is None:
        return None, "No video uploaded."

    frames = read_video(video_path)
    if not frames:
        return None, "Could not read video."

    fps = 30
    keypoints_seq = pose_engine.extract_keypoints_batch(frames)
    velocities = pose_engine.get_wrist_velocity(keypoints_seq, fps=fps)
    accelerations = pose_engine.get_wrist_acceleration(velocities, fps=fps)
    phases = swing_classifier.detect_phases(keypoints_seq, velocities, accelerations)

    contact_idx = swing_classifier.get_contact_frame_index(phases)
    if contact_idx is None:
        contact_idx = int(np.argmax(velocities))

    angles = pose_engine.get_joint_angles(keypoints_seq[contact_idx], side=playing_hand)
    stroke_type = swing_classifier.classify_stroke(keypoints_seq[contact_idx])

    # Load pro data
    pro_key = PROS.get(pro_selection, 'djokovic_forehand')
    pro_path = os.path.join(PRO_DATA_DIR, f"{pro_key}.json")
    pro_data = {}
    if os.path.exists(pro_path):
        with open(pro_path) as f:
            pro_data = json.load(f)

    pro_angles = pro_data.get('contact_angles', {})

    # Build comparison text
    comparison_lines = [f"Comparison: You vs. {pro_data.get('player', pro_selection)}",
                        f"Stroke: {stroke_type}", ""]

    if angles and pro_angles:
        comparison_lines.append(f"{'Metric':<20} {'You':>8} {'Pro':>8} {'Diff':>8}")
        comparison_lines.append("-" * 48)
        for metric in ['elbow_angle', 'hip_rotation', 'shoulder_angle', 'knee_angle', 'racket_lag']:
            if metric in angles and metric in pro_angles:
                diff = angles[metric] - pro_angles[metric]
                nice = metric.replace('_', ' ').title()
                comparison_lines.append(
                    f"{nice:<20} {angles[metric]:>7.1f}° {pro_angles[metric]:>7.1f}° {diff:>+7.1f}°"
                )

    # Generate coaching comparison
    pro_name = pro_key.split('_')[0]
    pro_comp = coaching_engine.compare_to_pro(angles, pro_name, stroke_type)
    scores = coaching_engine.score_swing(angles, stroke_type)
    coaching_text = coaching_engine.generate_coaching_report(scores, angles, stroke_type, pro_comp)

    comparison_lines.append("")
    comparison_lines.append(coaching_text)

    # Annotate contact frame with dual skeleton
    contact_frame = frames[contact_idx].copy()
    pose_engine.draw_skeleton(contact_frame, keypoints_seq[contact_idx],
                               color=(0, 255, 127), thickness=2)

    # Draw pro angles as text overlay
    cv2.putText(contact_frame, f"You vs {pro_data.get('player', 'Pro')} at CONTACT",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (204, 255, 0), 2)

    contact_frame_rgb = cv2.cvtColor(contact_frame, cv2.COLOR_BGR2RGB)
    return contact_frame_rgb, "\n".join(comparison_lines)


# ─── MODULE 4: Swing Plane ───────────────────────────────────────────

def analyze_swing_plane(video_path, playing_hand='right'):
    """Track wrist trajectory and fit swing plane."""
    if video_path is None:
        return None, "No video uploaded."

    frames = read_video(video_path)
    if not frames:
        return None, "Could not read video."

    keypoints_seq = pose_engine.extract_keypoints_batch(frames)
    velocities = pose_engine.get_wrist_velocity(keypoints_seq, fps=30)
    accelerations = pose_engine.get_wrist_acceleration(velocities, fps=30)
    phases = swing_classifier.detect_phases(keypoints_seq, velocities, accelerations)

    # Collect wrist positions during swing phases
    swing_phases = ['preparation', 'loading', 'contact', 'follow_through']
    wrist_points = []
    for i, kp in enumerate(keypoints_seq):
        if kp and phases[i] in swing_phases:
            side = playing_hand
            wrist_points.append((kp[f'{side}_wrist'][0], kp[f'{side}_wrist'][1]))

    if len(wrist_points) < 5:
        return None, "Not enough wrist data to fit swing plane."

    wrist_points = np.array(wrist_points)

    # Calculate swing plane angle (angle of best-fit line through wrist path)
    # Using the trajectory in 2D (image plane)
    dx = wrist_points[-1, 0] - wrist_points[0, 0]
    dy = wrist_points[-1, 1] - wrist_points[0, 1]
    # In image coords, y is inverted (down = positive)
    plane_angle = np.degrees(np.arctan2(-dy, dx))  # negative dy because y is inverted

    # Create matplotlib visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                     facecolor='#0D0D0D')

    # Left: wrist trajectory on frame
    ax1.set_facecolor('#0D0D0D')
    ax1.set_title('Wrist Trajectory (Side View)', color='#CCFF00', fontsize=14)
    # Plot trajectory colored by phase
    colors_map = {
        'preparation': '#FFFF00', 'loading': '#00A5FF',
        'contact': '#FF0000', 'follow_through': '#00FF00',
    }
    for i in range(len(wrist_points) - 1):
        phase = phases[i] if i < len(phases) else 'idle'
        c = colors_map.get(phase, '#808080')
        ax1.plot(wrist_points[i:i+2, 0], -wrist_points[i:i+2, 1], color=c, linewidth=2)

    ax1.set_xlabel('X (pixels)', color='white')
    ax1.set_ylabel('Y (pixels, inverted)', color='white')
    ax1.tick_params(colors='white')

    # Right: swing plane angle visualization
    ax2.set_facecolor('#0D0D0D')
    ax2.set_title('Swing Plane Analysis', color='#CCFF00', fontsize=14)

    # Draw reference planes
    angles_ref = {'Flat (5-15°)': 10, 'Topspin (15-35°)': 25, 'Heavy Topspin (>35°)': 40}
    for label, ang in angles_ref.items():
        x = np.linspace(0, 100, 50)
        y = x * np.tan(np.radians(ang))
        ax2.plot(x, y, '--', alpha=0.4, label=label)

    # User's swing plane
    x = np.linspace(0, 100, 50)
    y = x * np.tan(np.radians(abs(plane_angle)))
    ax2.plot(x, y, color='#CCFF00', linewidth=3, label=f'Your plane: {plane_angle:.1f}°')
    ax2.legend(facecolor='#1a1a1a', edgecolor='#CCFF00', labelcolor='white')
    ax2.set_xlabel('Forward (toward net)', color='white')
    ax2.set_ylabel('Upward', color='white')
    ax2.tick_params(colors='white')
    ax2.set_ylim(0, 80)

    plt.tight_layout()
    fig_path = tempfile.mktemp(suffix='.png')
    fig.savefig(fig_path, facecolor='#0D0D0D', dpi=100)
    plt.close(fig)

    # Generate text report
    if plane_angle < 5:
        verdict = "Very flat swing — good for approach shots but won't generate topspin."
    elif plane_angle < 15:
        verdict = "Flat to mild topspin. Good for aggressive flat drives."
    elif plane_angle < 35:
        verdict = "Solid topspin plane. This is the pro range for consistent groundstrokes."
    else:
        verdict = "Extreme low-to-high — heavy topspin like Nadal. Great for clay, may sacrifice pace on hard courts."

    report = f"Swing Plane Angle: {plane_angle:.1f}°\n{verdict}"

    return fig_path, report


# ─── MODULE 5: Ball Flight Prediction ────────────────────────────────

def predict_ball_flight(speed_mph, launch_angle, contact_height, spin_rpm,
                         lateral_angle):
    """Simulate and visualize ball trajectory."""
    speed_ms = speed_mph / 2.237

    result = ball_physics.simulate_trajectory(
        speed_ms, launch_angle, contact_height, spin_rpm, lateral_angle
    )
    mc_result = ball_physics.monte_carlo_shots(
        speed_ms, launch_angle, contact_height, spin_rpm, lateral_angle,
        n_sims=100
    )

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0D0D0D')

    # Left: side view trajectory
    ax1.set_facecolor('#0D0D0D')
    ax1.set_title('Ball Trajectory (Side View)', color='#CCFF00', fontsize=14)
    traj = result['trajectory']
    ax1.plot(traj['x'], traj['y'], color='#CCFF00', linewidth=2)
    ax1.axhline(y=0, color='white', linewidth=0.5)
    # Net
    ax1.plot([ball_physics.NET_DISTANCE, ball_physics.NET_DISTANCE],
             [0, ball_physics.NET_HEIGHT], color='white', linewidth=2, label='Net')
    # Net clearance indicator
    nc = result['net_clearance']
    if nc is not None:
        nc_color = '#00FF00' if nc > 0.5 else '#FFFF00' if nc > 0.1 else '#FF0000'
        ax1.scatter([ball_physics.NET_DISTANCE], [ball_physics.NET_HEIGHT + max(nc, 0)],
                    color=nc_color, s=100, zorder=5, label=f'Net clearance: {nc:.2f}m')
    ax1.set_xlabel('Distance (m)', color='white')
    ax1.set_ylabel('Height (m)', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1a1a1a', edgecolor='#CCFF00', labelcolor='white')
    ax1.set_xlim(-1, 25)
    ax1.set_ylim(-0.5, max(traj['y']) * 1.2 + 1)

    # Right: bird's eye view with Monte Carlo
    ax2.set_facecolor('#0D0D0D')
    ax2.set_title("Landing Zone (Bird's Eye)", color='#CCFF00', fontsize=14)

    # Draw court
    half_w = ball_physics.COURT_WIDTH / 2
    court_rect = plt.Rectangle((0, -half_w), ball_physics.COURT_LENGTH, ball_physics.COURT_WIDTH,
                                fill=False, edgecolor='white', linewidth=1)
    ax2.add_patch(court_rect)
    # Net
    ax2.plot([ball_physics.NET_DISTANCE, ball_physics.NET_DISTANCE],
             [-half_w, half_w], color='white', linewidth=2)
    # Service line
    ax2.plot([ball_physics.NET_DISTANCE + ball_physics.SERVICE_LINE,
              ball_physics.NET_DISTANCE + ball_physics.SERVICE_LINE],
             [-half_w / 2, half_w / 2], color='white', linewidth=0.5, linestyle='--')

    # Monte Carlo landing points
    mc_pts = np.array(mc_result['landing_points'])
    in_court = [(ball_physics.NET_DISTANCE < x < ball_physics.COURT_LENGTH and
                 abs(z) < half_w) for x, z in mc_pts]
    ax2.scatter(mc_pts[:, 0], mc_pts[:, 1], c=['#00FF00' if ic else '#FF0000' for ic in in_court],
                alpha=0.4, s=15)

    # Landing ellipse
    ellipse = Ellipse((mc_result['mean_x'], mc_result['mean_z']),
                       mc_result['std_x'] * 2, mc_result['std_z'] * 2,
                       fill=False, edgecolor='#CCFF00', linewidth=2, linestyle='--')
    ax2.add_patch(ellipse)
    ax2.scatter([result['landing_x']], [result['landing_z']],
                color='#CCFF00', s=100, marker='x', linewidths=3, label='Predicted landing')

    ax2.set_xlabel('Distance from baseline (m)', color='white')
    ax2.set_ylabel('Lateral (m)', color='white')
    ax2.tick_params(colors='white')
    ax2.set_xlim(-2, 26)
    ax2.set_ylim(-8, 8)
    ax2.set_aspect('equal')
    ax2.legend(facecolor='#1a1a1a', edgecolor='#CCFF00', labelcolor='white')

    plt.tight_layout()
    fig_path = tempfile.mktemp(suffix='.png')
    fig.savefig(fig_path, facecolor='#0D0D0D', dpi=100)
    plt.close(fig)

    report = ball_physics.generate_trajectory_report(result, mc_result)
    return fig_path, report


# ─── BUILD GRADIO APP ─────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="SwingForge AI") as app:
        gr.Markdown(
            "# SwingForge AI\n"
            "### Drop a video. Mirror a legend. Fix your swing.\n"
            "---"
        )

        with gr.Tabs():
            # ── Tab 1: Upload & Analyze ──
            with gr.TabItem("Upload & Analyze"):
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_video = gr.Video(label="Upload your swing video")
                        hand_select = gr.Radio(
                            choices=['right', 'left'], value='right',
                            label="Playing Hand"
                        )
                        analyze_btn = gr.Button("Analyze Swing", variant="primary")
                    with gr.Column(scale=1):
                        output_video = gr.Video(label="Annotated Video")
                with gr.Row():
                    coaching_report = gr.Textbox(
                        label="Coaching Report", lines=15, interactive=False
                    )
                    json_output = gr.Textbox(
                        label="JSON Report", lines=15, interactive=False
                    )

                analyze_btn.click(
                    fn=analyze_video,
                    inputs=[upload_video, hand_select],
                    outputs=[output_video, coaching_report, json_output],
                )

            # ── Tab 2: Live Webcam ──
            with gr.TabItem("Live Webcam"):
                gr.Markdown(
                    "**Press Record, stand 6-8 feet back, and swing!** "
                    "The analyzed feed shows your skeleton + joint angles in real time."
                )
                hand_select_live = gr.Radio(
                    choices=['right', 'left'], value='right',
                    label="Playing Hand"
                )
                with gr.Row():
                    webcam_feed = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="Press Record to start live analysis",
                    )
                    webcam_output = gr.Image(
                        label="Analyzed Feed (skeleton + angles)",
                    )

                webcam_feed.stream(
                    fn=process_webcam_frame,
                    inputs=[webcam_feed, hand_select_live],
                    outputs=webcam_output,
                    stream_every=0.1,
                    time_limit=600,
                )

            # ── Tab 3: Pro Compare ──
            with gr.TabItem("Pro Compare"):
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_video = gr.Video(label="Upload your swing video")
                        pro_select = gr.Dropdown(
                            choices=list(PROS.keys()),
                            value='Djokovic (Forehand)',
                            label="Compare against"
                        )
                        hand_compare = gr.Radio(
                            choices=['right', 'left'], value='right',
                            label="Playing Hand"
                        )
                        compare_btn = gr.Button("Compare", variant="primary")
                    with gr.Column(scale=1):
                        compare_image = gr.Image(label="Contact Frame Comparison")
                compare_text = gr.Textbox(
                    label="Comparison Report", lines=20, interactive=False
                )

                compare_btn.click(
                    fn=compare_to_pro,
                    inputs=[compare_video, pro_select, hand_compare],
                    outputs=[compare_image, compare_text],
                )

            # ── Tab 4: Swing Plane ──
            with gr.TabItem("Swing Plane"):
                with gr.Row():
                    with gr.Column(scale=1):
                        plane_video = gr.Video(label="Upload your swing video")
                        hand_plane = gr.Radio(
                            choices=['right', 'left'], value='right',
                            label="Playing Hand"
                        )
                        plane_btn = gr.Button("Analyze Swing Plane", variant="primary")
                    with gr.Column(scale=1):
                        plane_plot = gr.Image(label="Swing Plane Visualization")
                plane_report = gr.Textbox(
                    label="Swing Plane Report", lines=5, interactive=False
                )

                plane_btn.click(
                    fn=analyze_swing_plane,
                    inputs=[plane_video, hand_plane],
                    outputs=[plane_plot, plane_report],
                )

            # ── Tab 5: Ball Flight ──
            with gr.TabItem("Ball Flight"):
                gr.Markdown("Simulate ball trajectory from swing parameters.")
                with gr.Row():
                    with gr.Column(scale=1):
                        speed_input = gr.Slider(
                            30, 140, value=75, step=1,
                            label="Ball Speed (mph)"
                        )
                        angle_input = gr.Slider(
                            -10, 45, value=12, step=0.5,
                            label="Launch Angle (degrees)"
                        )
                        height_input = gr.Slider(
                            0.5, 3.0, value=1.0, step=0.1,
                            label="Contact Height (m)"
                        )
                        spin_input = gr.Slider(
                            0, 4000, value=1800, step=100,
                            label="Topspin (RPM)"
                        )
                        lateral_input = gr.Slider(
                            -30, 30, value=5, step=1,
                            label="Lateral Angle (degrees, + = crosscourt)"
                        )
                        flight_btn = gr.Button("Simulate", variant="primary")
                    with gr.Column(scale=1):
                        flight_plot = gr.Image(label="Trajectory Visualization")
                flight_report = gr.Textbox(
                    label="Ball Flight Report", lines=10, interactive=False
                )

                flight_btn.click(
                    fn=predict_ball_flight,
                    inputs=[speed_input, angle_input, height_input,
                            spin_input, lateral_input],
                    outputs=[flight_plot, flight_report],
                )

    return app


def launch():
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    launch()
