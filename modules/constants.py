T = 600 # Num timesteps
N = 18 # Num nodes

NODE_INDEX = {
    'neck': 0,
    'left_shoulder': 1, 'right_shoulder': 2,
    'left_hip': 3, 'right_hip': 4,
    'left_elbow': 5, 'right_elbow': 6,
    'left_wrist': 7, 'right_wrist': 8,
    'left_knee': 9, 'right_knee': 10,
    'left_ankle': 11, 'right_ankle': 12,
    'nose': 13,
    'left_eye': 14, 'right_eye': 15,
    'left_ear': 16, 'right_ear': 17,
}

EDGE_LABELS = [
    ('neck', 'nose'),
    ('neck', 'left_shoulder'),
    ('neck', 'right_shoulder'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_elbow', 'left_wrist'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle'),
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'),
    ('left_wrist', 'right_wrist'),
    ('right_wrist', 'left_wrist'),
    ('left_ankle', 'right_ankle'),
    ('right_ankle', 'left_ankle'),
    ('left_ankle', 'left_wrist'),
    ('left_wrist', 'left_ankle'),
    ('right_ankle', 'right_wrist'),
    ('right_wrist', 'right_ankle'),
    ('left_ankle', 'right_wrist'),
    ('right_wrist', 'left_ankle'),
    ('right_ankle', 'left_wrist'),
    ('left_wrist', 'right_ankle'),
]

EDGE_INDEX = [[], []]
for (src, dest) in EDGE_LABELS:
    EDGE_INDEX[0].append(NODE_INDEX[src])
    EDGE_INDEX[1].append(NODE_INDEX[dest])

PHYSICAL_EDGES = [
    ('neck', 'nose'),
    ('neck', 'left_shoulder'),
    ('neck', 'right_shoulder'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_elbow', 'left_wrist'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle'),
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'),
]

PHYSICAL_EDGE_INDEX = [[], []]
for (src, dest) in PHYSICAL_EDGES:
    PHYSICAL_EDGE_INDEX[0].append(NODE_INDEX[src])
    PHYSICAL_EDGE_INDEX[1].append(NODE_INDEX[dest])

FEATURE_LIST = [
    'hands_position_cca_z', 'feet_position_cca_z',
    'hands_angles_cca_vec', 'feet_angles_cca_vec',
    'hands_v_corr_z', 'feet_v_corr_z', 'hands_distance_5th',
    'feet_distance_5th', 'hands_lift_95th', 'feet_lift_95th',
    'hands_close_prob', 'feet_close_prob', 'hands_lift_prob',
    'feet_lift_prob', 'hands_activity', 'hands_mobility',
    'hands_complexity', 'feet_activity', 'feet_mobility',
    'feet_complexity'
]