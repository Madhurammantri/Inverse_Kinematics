import numpy as np
from scipy.optimize import minimize
from math import pi, cos, sin

# Updated link lengths (in meters)
H_base = 0.280
H_offset = 0.10392
Base_Height = H_base + H_offset
L1 = 0.530      # Link 1
Lx = 0.105      # New Link x
Ly = 0.355 - 0.105  # New Link y
L3 = 0.180      # Link 3
EE_offset = (19 + 75.54) / 1000

def dh_transform(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# All joints are revolute with updated link structure
def forward_kinematics(q):
    q1, q2, q3, q4, q5 = q
    dh_params = [
        (0,    pi/2, Base_Height, q1),   # Base to Link1
        (0,   -pi/2, 0,           q2),   # Link1 to Link x
        (L1,   0,     0,           q3),   # Link x to Link y
        (Lx,   0,     0,           q4),   # Link y to Link 3
        (Ly,   0,     0,           q5)    # Link 3 to EE
    ]
    T = np.eye(4)
    for a, alpha, d, theta in dh_params:
        T = T @ dh_transform(a, alpha, d, theta)
    T = T @ np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, L3 + EE_offset],
        [0, 0, 0, 1]
    ])
    return T

def rotation_matrix_from_rpy(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ])
    Ry = np.array([
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ])
    Rz = np.array([
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def ik_cost(q, target_pos, target_rot):
    T = forward_kinematics(q)
    pos_error = np.linalg.norm(T[:3, 3] - target_pos)
    rot_error = np.linalg.norm(T[:3, :3] - target_rot)
    return pos_error + rot_error

def solve_ik(x, y, z, roll, pitch, yaw):
    target_pos = np.array([x, y, z])
    target_rot = rotation_matrix_from_rpy(roll, pitch, yaw)
    q_init = [0, 0, 0, 0, 0]  # initial guess

    bounds = [
        (-pi, pi),              # q1 (base)
        (-2.1462, 2.1462),      # q2 (link1 to link x)
        (-3.14, 3.14),          # q3 (link x to link y)
        (-1.2398, 3.14),        # q4 (link y to link 3)
        (-3.14, 3.14)           # q5 (EE rotation)
    ]

    result = minimize(ik_cost, q_init, args=(target_pos, target_rot), bounds=bounds)

    if result.success:
        return result.x
    else:
        return None

# Example usage:
x, y, z = 0.65, 0.102, 0.401
roll, pitch, yaw = 0.6414, 0.8308, 1.2909
solution = solve_ik(x, y, z, roll, pitch, yaw)

if solution is not None:
    print("Joint Angles (in radians):")
    for i, angle in enumerate(solution):
        print(f"q{i+1}: {angle:.4f}")
else:
    print("No solution found. Check reachability of the pose.")
