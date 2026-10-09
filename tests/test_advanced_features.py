"""
Comprehensive Unit and Integration Tests for:
1. Phonological Sub-Unit Feature Extractor (HamNoSys Decomposition)
2. Non-Manual Facial Expression Classifier (Question vs. Statement Modality)
3. Two-Way Communication: Speech-to-ISL Translation & Sign Player Data
4. Session History, Dialogue Logging & PDF/TXT/JSON Transcript Exporter
"""

import os
import json
import shutil
import unittest
import numpy as np
from pathlib import Path

from src.linguistics.hamnosys_extractor import HamNoSysExtractor
from src.vision.facial_modality_classifier import FacialModalityClassifier
from src.translation.english_to_isl import EnglishToISLTranslator
from src.translation.sign_player_data import SignPlayerData
from src.session.session_manager import SessionManager
from src.config import DATA_DIR


class TestPhonologyAndHamNoSys(unittest.TestCase):
    def setUp(self):
        self.extractor = HamNoSysExtractor()

    def test_empty_landmarks_rest(self):
        """Verifies graceful handling of empty or inactive landmarks."""
        res = self.extractor.decompose_frame({})
        self.assertEqual(res["dominant_hand"], "NONE")
        self.assertIn("REST", res["handshape"]["category"])
        self.assertIn("hamnomovement", res["hamnosys_transcription"])

    def test_handshape_flat_hand(self):
        """Verifies flat open hand configuration detection."""
        hand = np.zeros((21, 3))
        # Wrist at origin
        hand[0] = [0.5, 0.5, 0.0]
        # MCP joints
        hand[5] = [0.48, 0.45, 0.0]
        hand[9] = [0.50, 0.44, 0.0]
        hand[13] = [0.52, 0.45, 0.0]
        hand[17] = [0.54, 0.46, 0.0]
        # Extended finger tips (significantly farther than MCPs)
        hand[4] = [0.44, 0.40, 0.0]
        hand[8] = [0.48, 0.32, 0.0]
        hand[12] = [0.50, 0.30, 0.0]
        hand[16] = [0.52, 0.32, 0.0]
        hand[20] = [0.54, 0.34, 0.0]

        res = self.extractor.decompose_frame({"right_hand": hand})
        self.assertIn(res["handshape"]["category"], [HamNoSysExtractor.HS_FLAT_HAND, HamNoSysExtractor.HS_OPEN_SPREAD])
        self.assertEqual(res["dominant_hand"], "RIGHT")

    def test_bimanual_symmetry(self):
        """Verifies two-handed symmetric sign detection."""
        hand_l = np.zeros((21, 3))
        hand_r = np.zeros((21, 3))
        hand_l[0] = [0.4, 0.5, 0.0]
        hand_r[0] = [0.6, 0.5, 0.0]

        res = self.extractor.decompose_frame({"left_hand": hand_l, "right_hand": hand_r})
        self.assertEqual(res["two_handed_relation"], HamNoSysExtractor.REL_BIMANUAL_SYMMETRIC)


class TestFacialModalityClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = FacialModalityClassifier()

    def test_declarative_statement(self):
        """Verifies default neutral statement classification."""
        res = self.classifier.process_face_landmarks(None)
        self.assertEqual(res["modality"], FacialModalityClassifier.MODALITY_STATEMENT)
        self.assertEqual(res["punctuation"], ".")

    def test_polar_question_eyebrow_raise(self):
        """Verifies Polar (Yes/No) Question classification from raised eyebrows."""
        face = np.zeros((20, 3))
        # Eyebrows raised significantly higher (smaller Y) relative to nose tip
        face[1] = [0.4, 0.12, 0]   # Left brow mid
        face[4] = [0.6, 0.12, 0]   # Right brow mid
        face[2] = [0.45, 0.13, 0]  # Left brow inner
        face[3] = [0.55, 0.13, 0]  # Right brow inner
        face[14] = [0.5, 0.45, 0]  # Nose tip
        face[16] = [0.35, 0.35, 0] # Left cheek
        face[17] = [0.65, 0.35, 0] # Right cheek

        res = self.classifier.process_face_landmarks(face)
        self.assertEqual(res["modality"], FacialModalityClassifier.MODALITY_POLAR_QUESTION)
        self.assertEqual(res["punctuation"], "?")

    def test_wh_question_furrow(self):
        """Verifies Wh-Question classification from brow furrow."""
        face = np.zeros((20, 3))
        # Narrow distance between inner eyebrows (furrowed)
        face[1] = [0.45, 0.22, 0]
        face[4] = [0.55, 0.22, 0]
        face[2] = [0.49, 0.22, 0]  # Inner brows pinched close together
        face[3] = [0.51, 0.22, 0]
        face[14] = [0.5, 0.40, 0]
        face[16] = [0.30, 0.35, 0]
        face[17] = [0.70, 0.35, 0]

        res = self.classifier.process_face_landmarks(face)
        self.assertEqual(res["modality"], FacialModalityClassifier.MODALITY_WH_QUESTION)
        self.assertEqual(res["punctuation"], "?")


class TestEnglishToISLTranslation(unittest.TestCase):
    def setUp(self):
        self.translator = EnglishToISLTranslator()

    def test_svo_to_sov_restructuring(self):
        """Verifies English Subject-Verb-Object to ISL Subject-Object-Verb transformation."""
        res = self.translator.translate_to_isl("I give you the book")
        self.assertEqual(res["gloss_string"], "ME YOU BOOK GIVE")
        self.assertFalse(res["is_question"])
        self.assertGreater(len(res["sign_cues"]), 0)

    def test_wh_question_positioning(self):
        """Verifies question word moves to sentence-final position."""
        res = self.translator.translate_to_isl("Where are you going?")
        self.assertIn("WHERE ?", res["gloss_string"])
        self.assertTrue(res["is_question"])
        self.assertEqual(res["modality"], "WH_QUESTION")

    def test_fingerspelling_decomposition(self):
        """Verifies out-of-lexicon words are cleanly fingerspelled."""
        res = self.translator.translate_to_isl("Amit")
        self.assertEqual(res["gloss_string"], "A M I T")
        self.assertEqual(len(res["sign_cues"]), 4)


class TestSignPlayerData(unittest.TestCase):
    def test_lexical_sign_retrieval(self):
        """Verifies animated keyframe coordinate sequence for vocabulary signs."""
        signs = ["NAMASTE", "HELLO", "THANK_YOU", "HELP", "WATER", "FOOD", "BOOK", "TEACHER", "GIVE", "WHAT", "WHERE"]
        for s in signs:
            anim = SignPlayerData.get_sign_animation(s)
            self.assertEqual(anim["sign"], s)
            self.assertIn("frames", anim)
            self.assertGreater(len(anim["frames"]), 0)

    def test_fingerspelling_letter_retrieval(self):
        """Verifies single letter animation keyframe generation."""
        anim = SignPlayerData.get_sign_animation("B")
        self.assertEqual(anim["sign"], "B")
        self.assertEqual(anim["type"], "FINGERSPELLING")


class TestSessionManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = DATA_DIR / "test_session_exports"
        os.makedirs(self.test_dir, exist_ok=True)
        self.session = SessionManager("test_session_unit")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_logging_and_exports(self):
        """Verifies dialogue turn logging and PDF, TXT, JSON exports."""
        self.session.log_turn(
            speaker="Signer (ISL)",
            input_content="ME TEACHER MEET",
            translated_content="I am meeting the teacher.",
            modality="Declarative Statement",
            confidence=0.98
        )
        self.session.log_turn(
            speaker="Speaker (Hearing)",
            input_content="Where are you going?",
            translated_content="YOU GO WHERE ?",
            modality="Wh-Question",
            confidence=1.0,
            direction="ENGLISH_TO_ISL"
        )

        summary = self.session.get_session_summary()
        self.assertEqual(summary["total_turns"], 2)

        # JSON Export
        json_path = self.session.export_json(str(self.test_dir / "session.json"))
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            self.assertEqual(d["total_turns"], 2)

        # Plain Text Export
        txt_path = self.session.export_text(str(self.test_dir / "session.txt"))
        self.assertTrue(os.path.exists(txt_path))
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Turn #1", content)
            self.assertIn("Turn #2", content)

        # PDF Export
        pdf_path = self.session.export_pdf(str(self.test_dir / "session.pdf"))
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)


if __name__ == "__main__":
    unittest.main()
