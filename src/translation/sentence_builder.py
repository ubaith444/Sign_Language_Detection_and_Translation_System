from typing import List, Dict, Any, Optional
from src.translation.seq2seq_translator import ISLTranslator

class SentenceBuilder:
    """
    Manages real-time accumulation, fingerspelling assembly, and syntactic translation
    of recognized ISL signs into sentences.
    Provides explicit user controls: word spacing, character/word deletion, clear,
    and speech playback.
    """

    MODE_GLOSS = "GLOSS_TRANSLATION"
    MODE_FINGERSPELLING = "FINGERSPELLING"

    def __init__(self, translator: Optional[ISLTranslator] = None):
        self.translator = translator or ISLTranslator()
        self.mode = self.MODE_GLOSS
        self.tokens: List[str] = []
        self.confidences: List[float] = []
        self.raw_text: str = ""
        self.translated_text: str = ""
        self.translation_meta: Dict[str, Any] = {}

    def set_mode(self, mode: str) -> Dict[str, Any]:
        """Switches between Fingerspelling and Gloss Translation mode."""
        if mode in [self.MODE_GLOSS, self.MODE_FINGERSPELLING]:
            self.mode = mode
            self._update_output()
        return self.get_state()

    def add_word(self, token: str, confidence: float = 1.0) -> Dict[str, Any]:
        """Adds a recognized sign or fingerspelled letter."""
        clean_tok = token.strip().upper()
        if not clean_tok:
            return self.get_state()

        self.tokens.append(clean_tok)
        self.confidences.append(float(confidence))
        self._update_output()
        return self.get_state()

    def add_space(self) -> Dict[str, Any]:
        """
        Inserts a word spacing boundary.
        Essential for fingerspelling to demarcate separate words (e.g. 'HELLO' [SPACE] 'WORLD').
        """
        if self.tokens and self.tokens[-1] != " ":
            self.tokens.append(" ")
            self.confidences.append(1.0)
            self._update_output()
        return self.get_state()

    def delete_last(self) -> Dict[str, Any]:
        """Deletes the most recently added token (backspace)."""
        if self.tokens:
            self.tokens.pop()
            self.confidences.pop()
            self._update_output()
        return self.get_state()

    def undo(self) -> Dict[str, Any]:
        """Alias for delete_last."""
        return self.delete_last()

    def clear(self) -> Dict[str, Any]:
        """Clears all accumulated tokens and reset sentence."""
        self.tokens.clear()
        self.confidences.clear()
        self.raw_text = ""
        self.translated_text = ""
        self.translation_meta.clear()
        return self.get_state()

    def _update_output(self):
        """Processes current token stream into raw text and translated English."""
        if not self.tokens:
            self.raw_text = ""
            self.translated_text = ""
            self.translation_meta = {}
            return

        if self.mode == self.MODE_FINGERSPELLING:
            # Reconstruct words from character stream with spaces
            words = []
            curr_word = []
            for tok in self.tokens:
                if tok == " ":
                    if curr_word:
                        words.append("".join(curr_word))
                        curr_word = []
                else:
                    curr_word.append(tok)
            if curr_word:
                words.append("".join(curr_word))

            self.raw_text = " ".join(words)
            # In fingerspelling mode, assembled words form the English text
            self.translated_text = self.raw_text.capitalize() + ("." if self.raw_text and not self.raw_text.endswith(".") else "")
            self.translation_meta = {
                "translation_method": "fingerspelling_assembly",
                "assembled_words": words
            }
        else: # Gloss Translation Mode
            # Filter spaces for gloss translation
            gloss_list = [t for t in self.tokens if t != " "]
            res = self.translator.translate(gloss_list)
            self.raw_text = res.get("raw_concatenation", "")
            trans = res.get("translated_text", "")
            
            # Apply active modality punctuation
            if getattr(self, "current_modality", None) in ["POLAR_QUESTION", "WH_QUESTION"]:
                if trans.endswith("."):
                    trans = trans[:-1] + "?"
                elif trans and not trans.endswith("?"):
                    trans += "?"
            
            self.translated_text = trans
            self.translation_meta = res

    def set_modality(self, modality: str, punctuation: str = ".") -> Dict[str, Any]:
        """Updates active non-manual facial modality and refines translation punctuation."""
        self.current_modality = modality
        self.modality_punctuation = punctuation
        if self.translated_text:
            if modality in ["POLAR_QUESTION", "WH_QUESTION"] or punctuation == "?":
                if self.translated_text.endswith("."):
                    self.translated_text = self.translated_text[:-1] + "?"
                elif not self.translated_text.endswith("?"):
                    self.translated_text += "?"
            elif modality == "DECLARATIVE_STATEMENT":
                if self.translated_text.endswith("?"):
                    self.translated_text = self.translated_text[:-1] + "."
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """Returns active state dictionary."""
        valid_confs = [c for c, t in zip(self.confidences, self.tokens) if t != " "]
        avg_conf = float(sum(valid_confs) / len(valid_confs)) if valid_confs else 0.0

        return {
            "mode": self.mode,
            "tokens": list(self.tokens),
            "token_count": len(self.tokens),
            "raw_text": self.raw_text,
            "translated_text": self.translated_text,
            "average_confidence": round(avg_conf, 3),
            "translation_method": self.translation_meta.get("translation_method", "none"),
            "modality": getattr(self, "current_modality", "DECLARATIVE_STATEMENT")
        }
