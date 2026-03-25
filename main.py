#!/usr/bin/env python3
"""
SwingForge AI — Tennis Swing Analysis
Drop a video. Mirror a legend. Fix your swing.

Usage:
    python main.py              # Launch Gradio web UI
    python main.py --live       # Launch OpenCV live shadow mode
    python main.py --analyze VIDEO_PATH  # Analyze a video file from CLI
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="SwingForge AI — Tennis Swing Analysis")
    parser.add_argument('--live', action='store_true',
                        help='Launch live shadow swing mode (OpenCV webcam)')
    parser.add_argument('--drill', action='store_true',
                        help='Launch focused drill mode (one metric at a time)')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Analyze a video file from CLI')
    parser.add_argument('--hand', type=str, default='right', choices=['right', 'left'],
                        help='Playing hand (default: right)')
    parser.add_argument('--port', type=int, default=7860,
                        help='Gradio server port (default: 7860)')
    args = parser.parse_args()

    if args.drill:
        from modules.drill_mode import run_drill_mode
        run_drill_mode(playing_hand=args.hand)

    elif args.live:
        from modules.live_shadow import run_shadow_mode
        run_shadow_mode(playing_hand=args.hand)

    elif args.analyze:
        from core.pose_engine import PoseEngine
        from core.swing_classifier import SwingClassifier
        from core.coaching import CoachingEngine
        from utils import read_video

        print(f"Analyzing: {args.analyze}")
        pose = PoseEngine()
        classifier = SwingClassifier()
        coach = CoachingEngine()

        frames = read_video(args.analyze)
        if not frames:
            print("Error: Could not read video.")
            sys.exit(1)

        keypoints_seq = pose.extract_keypoints_batch(frames)
        velocities = pose.get_wrist_velocity(keypoints_seq)
        accelerations = pose.get_wrist_acceleration(velocities)
        phases = classifier.detect_phases(keypoints_seq, velocities, accelerations)

        contact_idx = classifier.get_contact_frame_index(phases)
        if contact_idx is None:
            import numpy as np
            contact_idx = int(np.argmax(velocities))

        angles = pose.get_joint_angles(keypoints_seq[contact_idx], side=args.hand)
        stroke = classifier.classify_stroke(keypoints_seq[contact_idx])
        follow = classifier.detect_follow_through_completion(keypoints_seq, phases)
        scores = coach.score_swing(angles, stroke, follow)
        report = coach.generate_coaching_report(scores, angles, stroke)

        print(f"\nStroke: {stroke}")
        print(f"Contact frame: {contact_idx}")
        print(f"\n{report}")

        pose.release()

    else:
        from ui.app import build_app
        print("Starting SwingForge AI...")
        print(f"Open http://localhost:{args.port} in your browser")
        app = build_app()
        app.launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
