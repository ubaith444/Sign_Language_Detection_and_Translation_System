"""
Phonological Sub-Unit Feature Extractor based on HamNoSys (Hamburg Sign Language Notation System).
Decomposes sign landmarks into constituent cherological / phonological parameters:
- Handshape (Finger configuration & thumb posture)
- Orientation (Palm normal vector & extended finger direction)
- Location (Place of Articulation relative to anatomical landmarks)
- Movement (Kinetic direction, trajectory curvature & repetition)
- Two-Handed Relation (Dominance, symmetry, contact & active-passive roles)
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class HamNoSysExtractor:
    """
    Decomposes normalized 222-D skeletal landmarks and temporal trajectories into
    formal HamNoSys phonological feature representations.
    """

    # Handshapes
    HS_FLAT_HAND = "FLAT_HAND"           # [hamflathand] All fingers extended together
    HS_OPEN_SPREAD = "OPEN_SPREAD_5"     # [hamfinger5] Open hand with fingers spread
    HS_FIST = "FIST"                     # [hamfist] Closed fist
    HS_INDEX_POINT = "INDEX_POINT"       # [hamfinger2] Index extended, others curled
    HS_V_SHAPE = "V_SHAPE"               # [hamfinger23] Index and middle extended
    HS_THUMB_UP = "THUMB_UP"             # [hamthumbacross] Thumb up, fingers curled
    HS_PINCH_CUP = "PINCH_CUPPED"        # [hamcup] Curved cupped fingers (C-shape / O-shape)
    HS_BENT_HAND = "BENT_HAND"           # [hambenthands] Fingers bent at MCP joints

    # Palm Orientations
    OR_PALM_IN = "PALM_INWARD"           # Facing signer's torso
    OR_PALM_OUT = "PALM_OUTWARD"         # Facing conversational partner
    OR_PALM_UP = "PALM_UPWARD"           # Facing sky / upward
    OR_PALM_DOWN = "PALM_DOWNWARD"       # Facing floor / downward
    OR_PALM_MEDIAL = "PALM_MEDIAL"       # Facing body midline

    # Locations (Places of Articulation)
    LOC_NEUTRAL_SPACE = "NEUTRAL_SPACE"   # In front of torso, unsupported
    LOC_FOREHEAD = "FOREHEAD"             # Upper head / brow region
    LOC_CHIN_MOUTH = "CHIN_MOUTH"         # Lower facial region
    LOC_CHEST = "CHEST"                   # Upper torso / sternum
    LOC_STOMACH_WAIST = "WAIST"           # Lower torso
    LOC_CONTRALATERAL = "CONTRALATERAL"   # Across opposite shoulder / chest

    # Movements
    MOV_STATIONARY = "STATIONARY"         # Maintained static position
    MOV_FORWARD = "FORWARD_THRUST"        # Towards conversant
    MOV_BACKWARD = "RETRACTION"           # Towards signer
    MOV_UPWARD = "UPWARD"                 # Vertical rise
    MOV_DOWNWARD = "DOWNWARD"             # Vertical descent
    MOV_LATERAL = "LATERAL_SWEEP"         # Horizontal translation
    MOV_OSCILLATING = "OSCILLATING"       # Repetitive sign cycle

    # Two-Handed Relations
    REL_UNIMANUAL_RIGHT = "UNIMANUAL_RIGHT"
    REL_UNIMANUAL_LEFT = "UNIMANUAL_LEFT"
    REL_BIMANUAL_SYMMETRIC = "BIMANUAL_SYMMETRIC"
    REL_BIMANUAL_ACTIVE_PASSIVE = "BIMANUAL_ACTIVE_PASSIVE"

    def __init__(self):
        # MediaPipe landmark indices for 21 hand keypoints:
        # 0: Wrist, 4: Thumb Tip, 8: Index Tip, 12: Middle Tip, 16: Ring Tip, 20: Pinky Tip
        # 5: Index MCP, 9: Middle MCP, 13: Ring MCP, 17: Pinky MCP
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_mcps = [2, 5, 9, 13, 17]

    def decompose_frame(
        self,
        landmarks_dict: Dict[str, Any],
        trajectory_buffer: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extracts phonological sub-units from a single frame's landmarks,
        optionally incorporating temporal movement history.
        """
        left_hand = np.array(landmarks_dict.get("left_hand", np.zeros((21, 3))))
        right_hand = np.array(landmarks_dict.get("right_hand", np.zeros((21, 3))))
        pose = np.array(landmarks_dict.get("pose", np.zeros((12, 3))))

        lh_active = bool(np.linalg.norm(left_hand) > 0.05)
        rh_active = bool(np.linalg.norm(right_hand) > 0.05)

        # 1. Two-Handed Relation Analysis
        if rh_active and lh_active:
            # Check symmetry
            dist_diff = np.abs(left_hand[:, 1] - right_hand[:, 1])
            is_symmetric = bool(np.mean(dist_diff) < 0.12)
            relation = self.REL_BIMANUAL_SYMMETRIC if is_symmetric else self.REL_BIMANUAL_ACTIVE_PASSIVE
            primary_hand = right_hand
            primary_label = "RIGHT"
        elif rh_active:
            relation = self.REL_UNIMANUAL_RIGHT
            primary_hand = right_hand
            primary_label = "RIGHT"
        elif lh_active:
            relation = self.REL_UNIMANUAL_LEFT
            primary_hand = left_hand
            primary_label = "LEFT"
        else:
            relation = "RESTING_INACTIVE"
            primary_hand = right_hand
            primary_label = "NONE"

        # 2. Handshape Decomposition
        handshape, finger_extensions = self._classify_handshape(primary_hand)

        # 3. Palm Orientation
        orientation, normal_vector = self._classify_orientation(primary_hand)

        # 4. Location / Place of Articulation
        location = self._classify_location(primary_hand, pose)

        # 5. Movement Extraction from Trajectory Buffer
        movement, mov_metrics = self._classify_movement(trajectory_buffer, primary_label)

        # 6. Generate HamNoSys Phonetic Notation Token String
        hamnosys_token = self._synthesize_hamnosys_token(
            handshape, orientation, location, movement, relation
        )

        return {
            "dominant_hand": primary_label,
            "two_handed_relation": relation,
            "handshape": {
                "category": handshape,
                "finger_extensions": finger_extensions
            },
            "orientation": {
                "category": orientation,
                "palm_normal": [round(float(v), 3) for v in normal_vector]
            },
            "location": {
                "place_of_articulation": location
            },
            "movement": {
                "trajectory": movement,
                "velocity_magnitude": mov_metrics.get("velocity", 0.0),
                "curvature": mov_metrics.get("curvature", 0.0)
            },
            "hamnosys_transcription": hamnosys_token,
            "phonological_summary": f"[{handshape} | {orientation} | {location} | {movement} | {relation}]"
        }

    def _classify_handshape(self, hand: np.ndarray) -> Tuple[str, Dict[str, bool]]:
        """Determines handshape based on finger extension angles and thumb opposition."""
        if np.linalg.norm(hand) < 0.05:
            return "REST", {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}

        wrist = hand[0]
        extensions = {}
        finger_names = ["index", "middle", "ring", "pinky"]
        tip_indices = [8, 12, 16, 20]
        mcp_indices = [5, 9, 13, 17]

        # Check thumb extension (distance from MCP 5 vs tip 4)
        thumb_tip = hand[4]
        thumb_mcp = hand[2]
        thumb_extended = np.linalg.norm(thumb_tip - wrist) > (np.linalg.norm(thumb_mcp - wrist) * 1.25)
        extensions["thumb"] = bool(thumb_extended)

        # Check four fingers: tip further from wrist than MCP
        for name, tip_idx, mcp_idx in zip(finger_names, tip_indices, mcp_indices):
            tip = hand[tip_idx]
            mcp = hand[mcp_idx]
            dist_tip = np.linalg.norm(tip - wrist)
            dist_mcp = np.linalg.norm(mcp - wrist)
            extensions[name] = bool(dist_tip > (dist_mcp * 1.18))

        num_extended = sum(extensions[k] for k in finger_names)

        if num_extended == 4:
            # All 4 fingers extended
            spread_dist = np.linalg.norm(hand[8] - hand[20])
            mcp_dist = np.linalg.norm(hand[5] - hand[17])
            if spread_dist > (mcp_dist * 1.5):
                hs = self.HS_OPEN_SPREAD
            else:
                hs = self.HS_FLAT_HAND
        elif num_extended == 0:
            if extensions["thumb"]:
                hs = self.HS_THUMB_UP
            else:
                hs = self.HS_FIST
        elif num_extended == 1 and extensions["index"]:
            hs = self.HS_INDEX_POINT
        elif num_extended == 2 and extensions["index"] and extensions["middle"]:
            hs = self.HS_V_SHAPE
        elif num_extended >= 3 and not extensions["pinky"]:
            hs = self.HS_PINCH_CUP
        else:
            hs = self.HS_BENT_HAND

        return hs, extensions

    def _classify_orientation(self, hand: np.ndarray) -> Tuple[str, List[float]]:
        """Computes palm normal vector from wrist, index MCP, and pinky MCP."""
        if np.linalg.norm(hand) < 0.05:
            return self.OR_PALM_OUT, [0.0, 0.0, 1.0]

        wrist = hand[0]
        index_mcp = hand[5]
        pinky_mcp = hand[17]

        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        normal = np.cross(v1, v2)
        norm_val = np.linalg.norm(normal)
        if norm_val > 1e-6:
            normal = normal / norm_val
        else:
            normal = np.array([0.0, 0.0, 1.0])

        nx, ny, nz = normal

        # Classify based on dominant normal axis
        if abs(nz) >= abs(nx) and abs(nz) >= abs(ny):
            orientation = self.OR_PALM_OUT if nz > 0 else self.OR_PALM_IN
        elif abs(ny) >= abs(nx) and abs(ny) >= abs(nz):
            orientation = self.OR_PALM_DOWN if ny > 0 else self.OR_PALM_UP
        else:
            orientation = self.OR_PALM_MEDIAL

        return orientation, [float(nx), float(ny), float(nz)]

    def _classify_location(self, hand: np.ndarray, pose: np.ndarray) -> str:
        """Determines place of articulation relative to upper body pose keypoints."""
        if np.linalg.norm(hand) < 0.05:
            return self.LOC_NEUTRAL_SPACE

        wrist = hand[0]
        hand_y = wrist[1]

        # Use pose landmarks if present
        # In normalized coords: 0 is top, 1 is bottom
        if np.linalg.norm(pose) > 0.05:
            nose_y = pose[0, 1] if pose.shape[0] > 0 else 0.2
            shoulder_y = (pose[1, 1] + pose[2, 1]) / 2.0 if pose.shape[0] > 2 else 0.4
            mid_hip_y = (pose[7, 1] + pose[8, 1]) / 2.0 if pose.shape[0] > 8 else 0.8
        else:
            nose_y = 0.22
            shoulder_y = 0.42
            mid_hip_y = 0.78

        if hand_y < nose_y:
            return self.LOC_FOREHEAD
        elif hand_y < shoulder_y - 0.05:
            return self.LOC_CHIN_MOUTH
        elif hand_y < shoulder_y + 0.18:
            return self.LOC_CHEST
        elif hand_y > mid_hip_y:
            return self.LOC_STOMACH_WAIST
        else:
            return self.LOC_NEUTRAL_SPACE

    def _classify_movement(
        self,
        trajectory_buffer: Optional[List[Dict[str, Any]]],
        primary_label: str
    ) -> Tuple[str, Dict[str, float]]:
        """Analyzes temporal trajectory displacement to characterize movement chereme."""
        if not trajectory_buffer or len(trajectory_buffer) < 4:
            return self.MOV_STATIONARY, {"velocity": 0.0, "curvature": 0.0}

        key = "right_hand" if primary_label == "RIGHT" else "left_hand"
        positions = []
        for frame in trajectory_buffer:
            if isinstance(frame, dict):
                h = frame.get(key)
                if h is not None and len(h) > 0:
                    positions.append(np.array(h[0]))  # Wrist position
            elif isinstance(frame, np.ndarray):
                # vector_222 layout: left hand [0:63], right hand [63:126]
                # wrist coordinates are first 3 floats of the hand
                if primary_label == "RIGHT" and len(frame) >= 66:
                    wrist = frame[63:66]
                elif len(frame) >= 3:
                    wrist = frame[0:3]
                else:
                    wrist = np.zeros(3, dtype=np.float32)
                if np.max(np.abs(wrist)) > 1e-4:
                    positions.append(wrist)

        if len(positions) < 4:
            return self.MOV_STATIONARY, {"velocity": 0.0, "curvature": 0.0}

        positions = np.array(positions)
        disp_total = positions[-1] - positions[0]
        dist_total = np.linalg.norm(disp_total)

        # Check total path length vs net displacement for curvature / oscillation
        diffs = np.diff(positions, axis=0)
        path_length = np.sum(np.linalg.norm(diffs, axis=1))

        if path_length < 0.04:
            return self.MOV_STATIONARY, {"velocity": float(path_length), "curvature": 0.0}

        curvature = float(path_length / max(dist_total, 1e-4))
        velocity = float(path_length / len(positions))

        if curvature > 2.2:
            return self.MOV_OSCILLATING, {"velocity": velocity, "curvature": curvature}

        # Check dominant displacement axis
        dx, dy, dz = disp_total
        if abs(dy) >= abs(dx) and abs(dy) >= abs(dz):
            mov = self.MOV_UPWARD if dy < 0 else self.MOV_DOWNWARD
        elif abs(dz) >= abs(dx) and abs(dz) >= abs(dy):
            mov = self.MOV_FORWARD if dz > 0 else self.MOV_BACKWARD
        else:
            mov = self.MOV_LATERAL

        return mov, {"velocity": velocity, "curvature": curvature}

    def _synthesize_hamnosys_token(
        self,
        handshape: str,
        orientation: str,
        location: str,
        movement: str,
        relation: str
    ) -> str:
        """Synthesizes structured HamNoSys symbol string."""
        hs_map = {
            self.HS_FLAT_HAND: "hamflathand",
            self.HS_OPEN_SPREAD: "hamfinger5",
            self.HS_FIST: "hamfist",
            self.HS_INDEX_POINT: "hamfinger2",
            self.HS_V_SHAPE: "hamfinger23",
            self.HS_THUMB_UP: "hamthumbacross",
            self.HS_PINCH_CUP: "hamcup",
            self.HS_BENT_HAND: "hambenthands"
        }
        or_map = {
            self.OR_PALM_IN: "hampalmin",
            self.OR_PALM_OUT: "hampalmout",
            self.OR_PALM_UP: "hampalmu",
            self.OR_PALM_DOWN: "hampalmd",
            self.OR_PALM_MEDIAL: "hampalml"
        }
        loc_map = {
            self.LOC_FOREHEAD: "hamforehead",
            self.LOC_CHIN_MOUTH: "hamchin",
            self.LOC_CHEST: "hamchest",
            self.LOC_NEUTRAL_SPACE: "hamneutralspace",
            self.LOC_STOMACH_WAIST: "hambelly",
            self.LOC_CONTRALATERAL: "hamacross"
        }
        mov_map = {
            self.MOV_STATIONARY: "hamnomovement",
            self.MOV_FORWARD: "hammoveo",
            self.MOV_BACKWARD: "hammovei",
            self.MOV_UPWARD: "hammoveu",
            self.MOV_DOWNWARD: "hammoved",
            self.MOV_LATERAL: "hammovel",
            self.MOV_OSCILLATING: "hamrepeat"
        }

        hs_sym = hs_map.get(handshape, "hamhand")
        or_sym = or_map.get(orientation, "hampalm")
        loc_sym = loc_map.get(location, "hamspace")
        mov_sym = mov_map.get(movement, "hammov")

        rel_prefix = "hamtwohanded " if "BIMANUAL" in relation else ""
        return f"{rel_prefix}{loc_sym} {hs_sym} {or_sym} {mov_sym}"
