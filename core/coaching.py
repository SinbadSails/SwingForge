"""
CoachingEngine — Scores swing metrics, compares to pro references,
and generates specific, actionable coaching feedback.
"""

import json
import os
import numpy as np

# Ideal ranges for each metric (forehand defaults)
IDEAL_RANGES = {
    'forehand': {
        'elbow_angle': (125, 165),       # arm extension at contact (pros: 130-170°)
        'hip_rotation': (45, 95),        # shoulder-hip separation (2D-tolerant)
        'shoulder_angle': (55, 100),     # peak unit turn during loading
        'knee_angle': (105, 145),        # deepest knee bend during loading
        'racket_lag': (70, 145),         # peak forearm angle during backswing
        'contact_height_ratio': (0.9, 1.3),
    },
    'backhand': {
        'elbow_angle': (135, 158),
        'hip_rotation': (55, 90),
        'shoulder_angle': (65, 100),
        'knee_angle': (120, 150),
        'racket_lag': (65, 105),
        'contact_height_ratio': (0.9, 1.3),
    },
    'serve': {
        'elbow_angle': (158, 178),
        'hip_rotation': (40, 75),
        'shoulder_angle': (155, 178),
        'knee_angle': (108, 140),
        'racket_lag': (110, 165),
        'contact_height_ratio': (1.5, 2.5),
    },
}

# Scoring weights
WEIGHTS = {
    'elbow_angle': 0.15,
    'hip_rotation': 0.25,
    'shoulder_angle': 0.20,
    'knee_angle': 0.15,
    'racket_lag': 0.15,
    'follow_through': 0.10,
}

# Pro reference data directory
PRO_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pros')


class CoachingEngine:
    def __init__(self):
        self.pro_data = {}
        self._load_pro_data()

    def _load_pro_data(self):
        """Load pre-extracted pro player keypoint data from JSON files."""
        if not os.path.exists(PRO_DATA_DIR):
            return
        for fname in os.listdir(PRO_DATA_DIR):
            if fname.endswith('.json'):
                key = fname.replace('.json', '')
                with open(os.path.join(PRO_DATA_DIR, fname)) as f:
                    self.pro_data[key] = json.load(f)

    def score_metric(self, value, ideal_range):
        """Score a single metric 0-100 based on distance from ideal range.
        Penalty: -3.3 points per degree outside the range.
        30° off = 0 score. Inside the range = 100.
        """
        low, high = ideal_range
        if low <= value <= high:
            # Bonus: score higher near the middle of the range
            mid = (low + high) / 2
            range_half = (high - low) / 2
            dist_from_mid = abs(value - mid)
            # 100 at center, 85 at edges of range
            return 85.0 + 15.0 * (1.0 - dist_from_mid / (range_half + 1e-8))

        if value < low:
            dist = low - value
        else:
            dist = value - high

        # -3.3 points per degree outside the range (0 at 30° off)
        penalty = min(dist * 3.3, 100)
        return max(0.0, 100.0 - penalty)

    def score_swing(self, contact_angles, stroke_type='forehand',
                     follow_through_complete=True, loading_angles=None):
        """Score a swing using angles from the CORRECT phase for each metric.

        - elbow_angle, hip_rotation: measured at CONTACT (arm extension, hip drive)
        - shoulder_angle, knee_angle, racket_lag: measured at LOADING (coil, bend, lag)
        - follow_through: measured post-contact

        If loading_angles not provided, all metrics use contact_angles.
        """
        if contact_angles is None:
            return None

        ideals = IDEAL_RANGES.get(stroke_type, IDEAL_RANGES['forehand'])
        scores = {}

        # Metrics scored at CONTACT
        for metric in ['elbow_angle', 'hip_rotation']:
            if metric in contact_angles and metric in ideals:
                scores[metric] = self.score_metric(contact_angles[metric], ideals[metric])
            else:
                scores[metric] = 50.0

        # Metrics scored at LOADING (use loading_angles if available)
        load_src = loading_angles if loading_angles is not None else contact_angles
        for metric in ['shoulder_angle', 'knee_angle', 'racket_lag']:
            if metric in load_src and metric in ideals:
                scores[metric] = self.score_metric(load_src[metric], ideals[metric])
            else:
                scores[metric] = 50.0

        scores['follow_through'] = 100.0 if follow_through_complete else 30.0

        overall = sum(scores[m] * WEIGHTS[m] for m in WEIGHTS if m in scores)
        scores['overall'] = round(overall, 1)

        return scores

    def compare_to_pro(self, user_angles, pro_name, stroke_type='forehand'):
        """Compare user angles to a pro player's reference data.
        Returns dict with per-joint differences and textual comparison.
        """
        pro_key = f"{pro_name}_{stroke_type}"
        if pro_key not in self.pro_data:
            return None

        pro_angles = self.pro_data[pro_key].get('contact_angles', {})
        comparison = {}

        for metric in user_angles:
            if metric in pro_angles:
                diff = user_angles[metric] - pro_angles[metric]
                comparison[metric] = {
                    'user': round(user_angles[metric], 1),
                    'pro': round(pro_angles[metric], 1),
                    'diff': round(diff, 1),
                    'pro_name': pro_name.capitalize(),
                }
        return comparison

    def generate_coaching_report(self, scores, contact_angles, stroke_type='forehand',
                                  pro_comparison=None, loading_angles=None):
        """Generate a human-readable coaching report.
        Uses the correct angle source for each metric's phase.
        """
        if scores is None or contact_angles is None:
            return "Could not analyze swing — no pose detected at contact point."

        # Build a merged angle dict that uses the right phase for each metric
        angles = dict(contact_angles)
        if loading_angles:
            for m in ['shoulder_angle', 'knee_angle', 'racket_lag']:
                if m in loading_angles:
                    angles[m] = loading_angles[m]

        lines = []
        lines.append(f"SwingScore: {scores['overall']}/100")
        lines.append("")

        # Verdict
        if scores['overall'] >= 85:
            lines.append("Excellent swing mechanics! Minor refinements below.")
        elif scores['overall'] >= 70:
            lines.append("Solid foundation — focus on these corrections to level up.")
        elif scores['overall'] >= 50:
            lines.append("Good effort — several key areas need work.")
        else:
            lines.append("Let's rebuild from the ground up. Focus on the top 2 corrections.")

        lines.append("")

        # Rank corrections by impact (lowest score * highest weight)
        ideals = IDEAL_RANGES.get(stroke_type, IDEAL_RANGES['forehand'])
        corrections = []
        for metric in ['elbow_angle', 'hip_rotation', 'shoulder_angle', 'knee_angle', 'racket_lag']:
            if metric in scores and scores[metric] < 90:
                impact = WEIGHTS.get(metric, 0.1) * (100 - scores[metric])
                corrections.append((metric, scores[metric], impact, angles.get(metric, 0)))

        corrections.sort(key=lambda x: -x[2])

        # Top 2 corrections
        for i, (metric, score, impact, value) in enumerate(corrections[:2]):
            ideal = ideals.get(metric, (0, 0))
            nice_name = metric.replace('_', ' ').title()

            line = f"{i+1}. {nice_name}: {value:.1f}° (ideal: {ideal[0]}-{ideal[1]}°, score: {score:.0f}/100)"
            lines.append(line)

            # Add pro comparison if available
            if pro_comparison and metric in pro_comparison:
                pc = pro_comparison[metric]
                lines.append(f"   {pc['pro_name']}'s {nice_name.lower()} is {pc['pro']:.1f}°. "
                             f"Yours is {pc['diff']:+.1f}° off.")

            # Specific drill
            lines.append(f"   Drill: {self._get_drill(metric, value, ideal)}")
            lines.append("")

        # Follow-through note
        if scores.get('follow_through', 100) < 50:
            lines.append("Follow-through: Incomplete — your arm should cross your body after contact.")
            lines.append("   Drill: Shadow 20 swings focusing on finishing with your racket over your opposite shoulder.")

        return "\n".join(lines)

    def _get_drill(self, metric, value, ideal_range):
        """Generate a specific drill for a given metric."""
        low, high = ideal_range
        drills = {
            'elbow_angle': {
                'too_low': "Shadow 20 forehands focusing on extending your elbow through contact. Think 'reach and push'.",
                'too_high': "Your arm is too straight — add slight bend at contact. Practice with a focus on relaxed, whip-like motion.",
            },
            'hip_rotation': {
                'too_low': "Stand sideways, coil your hips back, then drive forward. 20 reps with no racket — just hip rotation.",
                'too_high': "You're over-rotating. Plant your front foot and let the hips stop at 90° to the net.",
            },
            'shoulder_angle': {
                'too_low': "Full unit turn — get your non-dominant shoulder pointing at the ball. 15 slow-motion shadow swings.",
                'too_high': "You're opening up too early. Keep the shoulder coiled until the hip drives forward.",
            },
            'knee_angle': {
                'too_low': "Bend those knees! Drop into a mini-squat before every swing. 20 split-step-to-swing drills.",
                'too_high': "Good knee bend but you're sitting too deep. Stay athletic, not squatting.",
            },
            'racket_lag': {
                'too_low': "Let the racket lag behind your elbow longer. Practice the 'waiter's tray' position at the back of your swing.",
                'too_high': "Your racket is too far behind — you're losing control. Compact the backswing.",
            },
        }
        direction = 'too_low' if value < low else 'too_high'
        return drills.get(metric, {}).get(direction,
                                          "Shadow 20 swings focusing on this movement pattern.")

    def get_injury_warnings(self, angles, stroke_type='serve'):
        """Flag mechanics that could increase injury risk."""
        warnings = []
        if angles is None:
            return warnings

        # Shoulder impingement risk: shoulder angle too closed on serve
        if stroke_type == 'serve' and 'shoulder_angle' in angles:
            if angles['shoulder_angle'] < 140:
                warnings.append("Shoulder impingement risk: your serving shoulder angle is too closed. "
                                "Open up your trophy position to reduce rotator cuff strain.")

        # Lumbar hyperextension on serve
        if stroke_type == 'serve' and 'hip_rotation' in angles:
            if angles['hip_rotation'] > 100:
                warnings.append("Lumbar hyperextension risk: excessive hip tilt during serve. "
                                "Strengthen core to maintain neutral spine through the motion.")

        # Knee torque on landing
        if 'knee_angle' in angles and angles['knee_angle'] < 90:
            warnings.append("Knee stress warning: deep knee bend beyond 90° at contact increases "
                            "patellar tendon load. Maintain a more athletic stance.")

        return warnings
