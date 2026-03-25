"""
BallPhysics — Simulates ball trajectory from swing data using
projectile motion with aerodynamic drag and Magnus force (topspin).
"""

import numpy as np
from scipy.integrate import solve_ivp


# Tennis ball constants
BALL_MASS = 0.057       # kg
BALL_RADIUS = 0.033     # m
BALL_AREA = np.pi * BALL_RADIUS ** 2
AIR_DENSITY = 1.21      # kg/m³
DRAG_COEFF = 0.55
GRAVITY = 9.81          # m/s²

# Court dimensions (meters)
COURT_LENGTH = 23.77
COURT_WIDTH = 10.97
NET_HEIGHT = 0.914      # center
NET_DISTANCE = COURT_LENGTH / 2  # from baseline
SERVICE_LINE = 6.40     # from net


class BallPhysics:
    def __init__(self):
        pass

    def estimate_ball_speed(self, wrist_velocity_px, pixels_per_meter=100):
        """Convert wrist velocity (px/s) to estimated ball speed (m/s).
        Ball speed is typically 1.5-2.5x wrist speed depending on racket mechanics.
        """
        wrist_speed_ms = wrist_velocity_px / pixels_per_meter
        # Approximate racket head speed multiplier
        return wrist_speed_ms * 2.0

    def estimate_spin_rpm(self, swing_plane_angle, follow_through_angle):
        """Estimate topspin RPM from swing plane angle and follow-through.
        swing_plane_angle: degrees from horizontal (higher = more topspin)
        """
        # Empirical approximation: pro topspin ~1500-3000 RPM
        base_rpm = 800
        spin_from_plane = swing_plane_angle * 50  # ~50 RPM per degree of low-to-high
        spin_from_follow = follow_through_angle * 10
        return min(base_rpm + spin_from_plane + spin_from_follow, 4000)

    def simulate_trajectory(self, speed_ms, launch_angle_deg, height_m,
                             spin_rpm=1500, lateral_angle_deg=0, n_steps=500):
        """Simulate ball trajectory with drag and Magnus force.

        Args:
            speed_ms: initial ball speed (m/s)
            launch_angle_deg: vertical launch angle (degrees above horizontal)
            height_m: contact height above ground (m)
            spin_rpm: topspin in RPM (positive = topspin)
            lateral_angle_deg: horizontal angle (0 = straight, + = crosscourt right)
            n_steps: simulation resolution

        Returns:
            dict with trajectory data, landing position, net clearance
        """
        theta = np.radians(launch_angle_deg)
        phi = np.radians(lateral_angle_deg)

        # Initial velocity components
        vx = speed_ms * np.cos(theta) * np.cos(phi)  # forward (toward net)
        vy = speed_ms * np.sin(theta)                 # vertical (up)
        vz = speed_ms * np.cos(theta) * np.sin(phi)   # lateral

        omega = spin_rpm * 2 * np.pi / 60  # rad/s

        def derivatives(t, state):
            x, y, z, vx, vy, vz = state
            v = np.sqrt(vx**2 + vy**2 + vz**2)
            if v < 0.01:
                return [0, 0, 0, 0, 0, 0]

            # Drag force
            F_drag = 0.5 * AIR_DENSITY * DRAG_COEFF * BALL_AREA * v
            ax_drag = -F_drag * vx / (BALL_MASS * v)
            ay_drag = -F_drag * vy / (BALL_MASS * v)
            az_drag = -F_drag * vz / (BALL_MASS * v)

            # Magnus force (topspin: force pushes ball down in flight)
            # Simplified: topspin axis perpendicular to velocity in vertical plane
            F_magnus = (4/3) * np.pi * BALL_RADIUS**3 * AIR_DENSITY * omega * v * 0.5
            # Topspin causes downward force
            ay_magnus = -F_magnus / BALL_MASS

            ax = ax_drag
            ay = ay_drag + ay_magnus - GRAVITY
            az = az_drag

            return [vx, vy, vz, ax, ay, az]

        # Solve ODE
        t_span = (0, 5)  # max 5 seconds
        t_eval = np.linspace(0, 5, n_steps)
        y0 = [0, height_m, 0, vx, vy, vz]

        def hit_ground(t, state):
            return state[1]  # y = 0
        hit_ground.terminal = True

        sol = solve_ivp(derivatives, t_span, y0, t_eval=t_eval, events=hit_ground,
                        max_step=0.01)

        trajectory = {
            'x': sol.y[0].tolist(),
            'y': sol.y[1].tolist(),
            'z': sol.y[2].tolist(),
            't': sol.t.tolist(),
        }

        # Landing position
        landing_x = sol.y[0][-1]
        landing_z = sol.y[2][-1]

        # Net clearance: find y at x = NET_DISTANCE
        net_clearance = None
        for i in range(len(sol.y[0]) - 1):
            if sol.y[0][i] <= NET_DISTANCE <= sol.y[0][i + 1]:
                # Linear interpolation
                frac = (NET_DISTANCE - sol.y[0][i]) / (sol.y[0][i + 1] - sol.y[0][i] + 1e-8)
                y_at_net = sol.y[1][i] + frac * (sol.y[1][i + 1] - sol.y[1][i])
                net_clearance = y_at_net - NET_HEIGHT
                break

        # Check if ball lands in court
        in_court = (NET_DISTANCE < landing_x < COURT_LENGTH and
                    abs(landing_z) < COURT_WIDTH / 2)

        return {
            'trajectory': trajectory,
            'landing_x': float(landing_x),
            'landing_z': float(landing_z),
            'landing_distance_from_baseline': float(COURT_LENGTH - landing_x),
            'net_clearance': float(net_clearance) if net_clearance is not None else None,
            'in_court': in_court,
            'speed_mph': float(speed_ms * 2.237),
            'spin_rpm': spin_rpm,
        }

    def monte_carlo_shots(self, speed_ms, launch_angle_deg, height_m,
                           spin_rpm=1500, lateral_angle_deg=0, n_sims=100,
                           speed_var=0.10, angle_var=2.0):
        """Run Monte Carlo simulation with variance to show shot distribution."""
        landings = []
        for _ in range(n_sims):
            s = speed_ms * (1 + np.random.uniform(-speed_var, speed_var))
            a = launch_angle_deg + np.random.uniform(-angle_var, angle_var)
            lat = lateral_angle_deg + np.random.uniform(-angle_var, angle_var)
            spin = spin_rpm * (1 + np.random.uniform(-0.15, 0.15))
            result = self.simulate_trajectory(s, a, height_m, spin, lat, n_steps=200)
            landings.append((result['landing_x'], result['landing_z']))

        landings = np.array(landings)
        return {
            'landing_points': landings.tolist(),
            'mean_x': float(np.mean(landings[:, 0])),
            'mean_z': float(np.mean(landings[:, 1])),
            'std_x': float(np.std(landings[:, 0])),
            'std_z': float(np.std(landings[:, 1])),
            'in_court_pct': float(np.mean([
                (NET_DISTANCE < lx < COURT_LENGTH and abs(lz) < COURT_WIDTH / 2)
                for lx, lz in landings
            ]) * 100),
        }

    def generate_trajectory_report(self, result, mc_result=None):
        """Generate human-readable trajectory report."""
        lines = []
        lines.append(f"Ball Speed: {result['speed_mph']:.0f} mph")
        lines.append(f"Topspin: ~{result['spin_rpm']:.0f} RPM")

        if result['net_clearance'] is not None:
            nc = result['net_clearance']
            if nc > 0.5:
                status = "Safe"
            elif nc > 0.1:
                status = "Tight"
            else:
                status = "CLIPPED"
            lines.append(f"Net Clearance: {nc:.2f}m ({status})")
        else:
            lines.append("Net Clearance: Ball did not reach the net")

        if result['in_court']:
            dist = result['landing_distance_from_baseline']
            lines.append(f"Landing: {dist:.1f}m from opposite baseline — IN")
        else:
            lines.append(f"Landing: OUT (x={result['landing_x']:.1f}m, z={result['landing_z']:.1f}m)")

        if mc_result:
            lines.append(f"\nMonte Carlo ({len(mc_result['landing_points'])} simulations):")
            lines.append(f"  In-court rate: {mc_result['in_court_pct']:.0f}%")
            lines.append(f"  Landing spread: ±{mc_result['std_x']:.1f}m deep, ±{mc_result['std_z']:.1f}m wide")

        return "\n".join(lines)
