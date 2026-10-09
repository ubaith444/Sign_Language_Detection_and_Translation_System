"""
Session History, Logging and Multi-Format Transcript Exporter.
Tracks bidirectional dialogue turns between ISL Signer and Hearing Speaker,
and exports formatted transcripts in PDF, plain text (.txt), and JSON format.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class SessionManager:
    """
    Manages communication sessions, turn logs, and audit trails
    with export capabilities for PDF, TXT, and JSON.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.created_at = datetime.now().isoformat()
        self.turns: List[Dict[str, Any]] = []

    def log_turn(
        self,
        speaker: str,                # "Signer (ISL)" or "Speaker (Hearing)"
        input_content: str,          # Raw input (e.g., "ME TEACHER MEET" or "What is your name?")
        translated_content: str,     # Translated output (e.g., "I am meeting the teacher." or "YOUR NAME WHAT ?")
        modality: str = "Declarative Statement",
        phonology_summary: str = "",
        confidence: float = 1.0,
        direction: str = "ISL_TO_ENGLISH"  # "ISL_TO_ENGLISH" or "ENGLISH_TO_ISL"
    ) -> Dict[str, Any]:
        """Records a completed communication turn into session history."""
        turn_index = len(self.turns) + 1
        turn_record = {
            "turn_id": turn_index,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "speaker": speaker,
            "direction": direction,
            "input_content": input_content,
            "translated_content": translated_content,
            "modality": modality,
            "phonology_summary": phonology_summary,
            "confidence": round(float(confidence), 2)
        }
        self.turns.append(turn_record)
        return turn_record

    def get_session_summary(self) -> Dict[str, Any]:
        """Returns overall session state and turn count."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "total_turns": len(self.turns),
            "turns": self.turns
        }

    def clear(self):
        """Resets turns for a new session."""
        self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.created_at = datetime.now().isoformat()
        self.turns.clear()

    def export_json(self, output_path: str) -> str:
        """Exports session log to standard JSON format."""
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = self.get_session_summary()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return output_path

    def export_text(self, output_path: str) -> str:
        """Exports formatted plain-text dialogue transcript."""
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        lines = [
            "=" * 72,
            " INDIAN SIGN LANGUAGE (ISL) TWO-WAY COMMUNICATION TRANSCRIPT",
            f" Session ID:  {self.session_id}",
            f" Date/Time:   {self.created_at}",
            f" Total Turns: {len(self.turns)}",
            "=" * 72,
            ""
        ]

        if not self.turns:
            lines.append("No dialogue turns recorded in this session.")
        else:
            for t in self.turns:
                lines.append(f"[{t['timestamp']}] Turn #{t['turn_id']} - {t['speaker']}")
                lines.append(f"  Input:       {t['input_content']}")
                lines.append(f"  Translation: {t['translated_content']}")
                lines.append(f"  Modality:    {t['modality']} (Conf: {int(t['confidence'] * 100)}%)")
                if t['phonology_summary']:
                    lines.append(f"  Phonology:   {t['phonology_summary']}")
                lines.append("-" * 72)

        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path

    def export_pdf(self, output_path: str) -> str:
        """
        Exports a publication-quality vector PDF transcript document
        with headers, metadata, and dialogue table using Matplotlib PdfPages.
        """
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with PdfPages(output_path) as pdf:
            # Figure layout
            fig = plt.figure(figsize=(8.5, 11), dpi=150)
            fig.patch.set_facecolor("#ffffff")

            # Top Header Bar
            plt.figtext(0.08, 0.94, "Indian Sign Language (ISL) Translation System",
                        fontsize=15, weight="bold", color="#0f172a")
            plt.figtext(0.08, 0.915, "Two-Way Communication Session Transcript & Linguistic Audit Log",
                        fontsize=9.5, color="#475569")

            # Metadata Table Box
            meta_text = (
                f"Session ID: {self.session_id}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Total Dialogue Turns: {len(self.turns)}  |  Modality Engine: Non-Manual Facial & HamNoSys"
            )
            plt.figtext(0.08, 0.865, meta_text, fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.0))

            # Render turns
            y_pos = 0.81
            if not self.turns:
                plt.figtext(0.08, 0.75, "No communication turns recorded in this session.",
                            fontsize=10, style="italic", color="#64748b")
            else:
                for idx, t in enumerate(self.turns[:12]):  # Up to 12 turns per page
                    is_signer = "Signer" in t["speaker"]
                    badge_color = "#1d4ed8" if is_signer else "#047857"
                    bg_color = "#f8fafc" if (idx % 2 == 0) else "#ffffff"

                    # Background card
                    box = dict(boxstyle="square,pad=0.4", facecolor=bg_color, edgecolor="#e2e8f0", linewidth=0.8)
                    plt.figtext(0.08, y_pos, f"Turn #{t['turn_id']}  [{t['timestamp']}]",
                                fontsize=8, weight="bold", color="#334155")
                    plt.figtext(0.70, y_pos, t["speaker"],
                                fontsize=8, weight="bold", color=badge_color)

                    y_pos -= 0.024
                    plt.figtext(0.10, y_pos, f"Input:        {t['input_content']}",
                                fontsize=8.5, color="#1e293b")

                    y_pos -= 0.022
                    plt.figtext(0.10, y_pos, f"Translation:  {t['translated_content']}",
                                fontsize=8.5, weight="bold", color="#0f172a")

                    y_pos -= 0.020
                    extra_meta = f"Modality: {t['modality']}  |  Confidence: {int(t['confidence'] * 100)}%"
                    if t.get("phonology_summary"):
                        extra_meta += f"  |  Phonology: {t['phonology_summary']}"
                    plt.figtext(0.10, y_pos, extra_meta, fontsize=7.5, color="#64748b")

                    y_pos -= 0.032
                    if y_pos < 0.10:
                        break

            # Footer
            plt.figtext(0.08, 0.04,
                        "System Architecture: Indian Sign Language Translation & Two-Way Verification Suite",
                        fontsize=7.5, color="#94a3b8")
            plt.figtext(0.85, 0.04, "Page 1 of 1", fontsize=7.5, color="#94a3b8")

            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        return output_path
