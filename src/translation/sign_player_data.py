"""
Animation Keyframe Library for the ISL 2D/3D Skeletal Sign Player.
Contains canonical skeletal keyframe coordinates for ISL lexical signs and fingerspelling letters.
Used by the browser Sign Player canvas to visually play back ISL signs for Deaf users.
"""

from typing import Dict, Any, List


class SignPlayerData:
    """
    Supplies keyframe skeleton joint coordinates for visual playback of ISL signs.
    Coordinates are normalized to [0.0, 1.0] viewport space.
    """

    @classmethod
    def get_sign_animation(cls, sign_name: str) -> Dict[str, Any]:
        """Returns animation metadata and keyframes for a requested sign or letter."""
        sign = sign_name.upper().strip()

        # Check if it's a known lexical sign
        if sign in cls.LEXICAL_ANIMATIONS:
            return cls.LEXICAL_ANIMATIONS[sign]

        # Check if it's a single letter for fingerspelling
        if len(sign) == 1 and sign.isalpha():
            return cls._generate_letter_animation(sign)

        # Fallback generic presentation
        return cls._generate_generic_sign(sign)

    @classmethod
    def _generate_letter_animation(cls, letter: str) -> Dict[str, Any]:
        """Generates static/hold animation frames for fingerspelled alphabet letter."""
        # Right hand centered in signing space with distinct finger extension
        return {
            "sign": letter,
            "type": "FINGERSPELLING",
            "frames": [
                {
                    "head": [0.5, 0.2],
                    "left_shoulder": [0.35, 0.35],
                    "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.35, 0.65],
                    "right_wrist": [0.58, 0.45],
                    "right_hand_shape": letter,
                    "description": f"ISL Fingerspelling letter {letter}"
                },
                {
                    "head": [0.5, 0.2],
                    "left_shoulder": [0.35, 0.35],
                    "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.35, 0.65],
                    "right_wrist": [0.58, 0.44],
                    "right_hand_shape": letter,
                    "description": f"Hold letter {letter}"
                }
            ]
        }

    @classmethod
    def _generate_generic_sign(cls, sign: str) -> Dict[str, Any]:
        return {
            "sign": sign,
            "type": "GENERIC",
            "frames": [
                {
                    "head": [0.5, 0.2],
                    "left_shoulder": [0.35, 0.35],
                    "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.42, 0.50],
                    "right_wrist": [0.58, 0.50],
                    "description": f"Signing {sign}"
                }
            ]
        }

    LEXICAL_ANIMATIONS = {
        "NAMASTE": {
            "sign": "NAMASTE",
            "type": "BIMANUAL_SYMMETRIC",
            "hamnosys": "hamchest hamflathand hampalml hamnomovement",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_elbow": [0.38, 0.52], "right_elbow": [0.62, 0.52],
                    "left_wrist": [0.44, 0.46], "right_wrist": [0.56, 0.46],
                    "caption": "Hands moving together to chest"
                },
                {
                    "head": [0.5, 0.21], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_elbow": [0.40, 0.54], "right_elbow": [0.60, 0.54],
                    "left_wrist": [0.48, 0.42], "right_wrist": [0.52, 0.42],
                    "caption": "Palms pressed in front of chest, slight head bow"
                },
                {
                    "head": [0.5, 0.22], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_elbow": [0.40, 0.54], "right_elbow": [0.60, 0.54],
                    "left_wrist": [0.49, 0.41], "right_wrist": [0.51, 0.41],
                    "caption": "Namaste hold"
                }
            ]
        },
        "HELLO": {
            "sign": "HELLO",
            "type": "UNIMANUAL_RIGHT",
            "hamnosys": "hamforehead hamflathand hampalmout hammovelo",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.58, 0.24],
                    "right_elbow": [0.68, 0.40],
                    "caption": "Hand touches temple/brow"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.68, 0.22],
                    "right_elbow": [0.72, 0.38],
                    "caption": "Salute arc moving outward"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.74, 0.26],
                    "right_elbow": [0.75, 0.42],
                    "caption": "Release outward towards partner"
                }
            ]
        },
        "THANK_YOU": {
            "sign": "THANK_YOU",
            "type": "UNIMANUAL_RIGHT",
            "hamnosys": "hamchin hamflathand hampalmin hammoveo",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.50, 0.28],
                    "caption": "Fingertips touching chin / lower lip"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.54, 0.38],
                    "caption": "Hand moves forward and down"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.58, 0.48],
                    "caption": "Open palm extended towards conversant"
                }
            ]
        },
        "HELP": {
            "sign": "HELP",
            "type": "BIMANUAL_ACTIVE_PASSIVE",
            "hamnosys": "hamchest hamflathand hampalmu hammoveu",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.46, 0.52], "right_wrist": [0.48, 0.48],
                    "caption": "Right fist placed on open flat left palm"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.46, 0.44], "right_wrist": [0.48, 0.40],
                    "caption": "Both hands lifted upward together"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.46, 0.38], "right_wrist": [0.48, 0.34],
                    "caption": "Upward lift completion"
                }
            ]
        },
        "WATER": {
            "sign": "WATER",
            "type": "UNIMANUAL_RIGHT",
            "hamnosys": "hamchin hamfinger23 hampalml hamrepeat",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.52, 0.30],
                    "caption": "W-handshape near corner of mouth"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.50, 0.28],
                    "caption": "Index/middle finger tapping chin"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.52, 0.30],
                    "caption": "Second tap cycle"
                }
            ]
        },
        "FOOD": {
            "sign": "FOOD",
            "type": "UNIMANUAL_RIGHT",
            "hamnosys": "hamchin hamcup hampalmin hamrepeat",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.50, 0.32],
                    "caption": "Cupped hand approaching mouth"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.49, 0.27],
                    "caption": "Fingertips tap lips repeatedly"
                }
            ]
        },
        "BOOK": {
            "sign": "BOOK",
            "type": "BIMANUAL_SYMMETRIC",
            "hamnosys": "hamchest hamflathand hampalmu hamnomovement",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.48, 0.46], "right_wrist": [0.52, 0.46],
                    "caption": "Palms pressed vertically (closed book)"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.42, 0.48], "right_wrist": [0.58, 0.48],
                    "caption": "Hands hinge open with pinkies touching (opened book)"
                }
            ]
        },
        "TEACHER": {
            "sign": "TEACHER",
            "type": "BIMANUAL_SYMMETRIC",
            "hamnosys": "hamforehead hamcup hampalmout hammoveo",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.42, 0.24], "right_wrist": [0.58, 0.24],
                    "caption": "Flattened O-hands at temples (teach)"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.42, 0.32], "right_wrist": [0.58, 0.32],
                    "caption": "Hands move forward and open"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.38, 0.55], "right_wrist": [0.62, 0.55],
                    "caption": "Flat palms slide vertically downward (person marker)"
                }
            ]
        },
        "GIVE": {
            "sign": "GIVE",
            "type": "UNIMANUAL_RIGHT",
            "hamnosys": "hamchest hamflathand hampalmu hammoveo",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.52, 0.48],
                    "caption": "Hand cupped or flat facing upward near chest"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.62, 0.44],
                    "caption": "Hand thrusts forward towards recipient"
                }
            ]
        },
        "WHAT": {
            "sign": "WHAT",
            "type": "BIMANUAL_SYMMETRIC",
            "hamnosys": "hamchest hamflathand hampalmu hamrepeat",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.40, 0.48], "right_wrist": [0.60, 0.48],
                    "caption": "Both open palms up at chest level, furrowed brows"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.38, 0.48], "right_wrist": [0.62, 0.48],
                    "caption": "Hands shake side-to-side in inquiry"
                }
            ]
        },
        "WHERE": {
            "sign": "WHERE",
            "type": "UNIMANUAL_RIGHT",
            "hamnosys": "hamchest hamfinger2 hampalmout hamrepeat",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.54, 0.38],
                    "caption": "Index finger pointing upward, furrowed brows"
                },
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.58, 0.38],
                    "caption": "Index finger oscillates left to right"
                }
            ]
        },
        "ME": {
            "sign": "ME",
            "type": "UNIMANUAL_RIGHT",
            "hamnosys": "hamchest hamfinger2 hampalmin hamnomovement",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.54, 0.46],
                    "caption": "Index finger points to own chest"
                }
            ]
        },
        "YOU": {
            "sign": "YOU",
            "type": "UNIMANUAL_RIGHT",
            "hamnosys": "hamchest hamfinger2 hampalmout hammoveo",
            "frames": [
                {
                    "head": [0.5, 0.2], "left_shoulder": [0.35, 0.35], "right_shoulder": [0.65, 0.35],
                    "left_wrist": [0.30, 0.65], "right_wrist": [0.56, 0.42],
                    "caption": "Index finger points forward to conversational partner"
                }
            ]
        }
    }
