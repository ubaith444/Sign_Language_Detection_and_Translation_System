"""
Two-Way Communication Engine: English Speech/Text to ISL Grammar Translation.
Converts English Subject-Verb-Object (SVO) sentences into Indian Sign Language (ISL)
Subject-Object-Verb (SOV) / Topic-Comment gloss sequences with question syntax restructuring
and fingerspelling fallback for out-of-lexicon vocabulary.
"""

import re
from typing import List, Dict, Any, Tuple


class EnglishToISLTranslator:
    """
    Translates spoken or typed English text into grammatical ISL gloss tokens
    and animatable sign sequence schedules.
    """

    # Lexicon vocabulary supported by system models and sign player
    SUPPORTED_SIGNS = {
        "NAMASTE", "HELLO", "THANK_YOU", "PLEASE", "HELP",
        "WATER", "FOOD", "YES", "NO", "STOP",
        "TEACHER", "STUDENT", "BOOK", "PEN", "GIVE", "DRINK",
        "ME", "YOU", "WE", "GOODBYE", "WHAT", "WHERE",
        "HOW_ARE_YOU", "SORRY", "MEET", "GO", "COME"
    }

    # Common English words mapping directly to ISL glosses
    SYNONYM_MAP = {
        "hi": "HELLO",
        "hey": "HELLO",
        "namaste": "NAMASTE",
        "thanks": "THANK_YOU",
        "thank": "THANK_YOU",
        "bye": "GOODBYE",
        "i": "ME",
        "my": "ME",
        "me": "ME",
        "myself": "ME",
        "you": "YOU",
        "your": "YOU",
        "yours": "YOU",
        "we": "WE",
        "us": "WE",
        "our": "WE",
        "teacher": "TEACHER",
        "instructor": "TEACHER",
        "student": "STUDENT",
        "pupil": "STUDENT",
        "book": "BOOK",
        "notebook": "BOOK",
        "pen": "PEN",
        "water": "WATER",
        "food": "FOOD",
        "meal": "FOOD",
        "eat": "FOOD",
        "help": "HELP",
        "assist": "HELP",
        "assistance": "HELP",
        "give": "GIVE",
        "giving": "GIVE",
        "gave": "GIVE",
        "drink": "DRINK",
        "drinking": "DRINK",
        "what": "WHAT",
        "where": "WHERE",
        "how": "HOW_ARE_YOU",
        "sorry": "SORRY",
        "excuse": "SORRY",
        "meet": "MEET",
        "meeting": "MEET",
        "go": "GO",
        "going": "GO",
        "went": "GO",
        "come": "COME",
        "coming": "COME",
        "came": "COME",
        "stop": "STOP",
        "halt": "STOP",
        "yes": "YES",
        "no": "NO",
        "please": "PLEASE"
    }

    # Functional English words omitted in ISL grammar (copulas, articles, auxiliaries)
    OMIT_WORDS = {
        "is", "am", "are", "was", "were", "be", "been", "being",
        "a", "an", "the",
        "do", "does", "did",
        "to", "of", "for", "with", "at", "by", "from"
    }

    # Interrogatives that are placed at sentence boundary / final position in ISL
    QUESTION_WORDS = {"what", "where", "who", "why", "when", "which", "how"}

    def __init__(self):
        pass

    def translate_to_isl(self, english_text: str) -> Dict[str, Any]:
        """
        Parses English text and produces a structured ISL gloss sequence,
        grammar transformation metadata, and animated sign cue schedule.
        """
        raw_text = (english_text or "").strip()
        if not raw_text:
            return {
                "english_input": "",
                "isl_gloss_sequence": [],
                "gloss_string": "",
                "is_question": False,
                "modality": "DECLARATIVE_STATEMENT",
                "cues": []
            }

        is_question = "?" in raw_text or any(raw_text.lower().startswith(q) for q in self.QUESTION_WORDS)

        # Clean text
        clean = re.sub(r"[^\w\s]", "", raw_text.lower())
        tokens = clean.split()

        # Step 1: Detect and isolate question words for sentence-final positioning
        q_words = [w for w in tokens if w in self.QUESTION_WORDS]
        content_words = [w for w in tokens if w not in self.QUESTION_WORDS and w not in self.OMIT_WORDS]

        # Step 2: SOV Restructuring (Subject-Object-Verb)
        reordered_words = self._restructure_sov(content_words)

        # In ISL, question words typically move to the end of the sentence
        if q_words:
            for qw in q_words:
                reordered_words.append(qw)

        # Step 3: Map words to ISL Glosses or Fingerspelling Tokens
        isl_glosses: List[str] = []
        cues: List[Dict[str, Any]] = []
        current_time = 0.0

        for word in reordered_words:
            gloss = self.SYNONYM_MAP.get(word)
            if gloss and gloss in self.SUPPORTED_SIGNS:
                isl_glosses.append(gloss)
                duration = 1.6 if gloss in ["HELLO", "THANK_YOU", "HOW_ARE_YOU"] else 1.2
                cues.append({
                    "sign": gloss,
                    "type": "LEXICAL_SIGN",
                    "source_word": word,
                    "start_time": round(current_time, 2),
                    "duration": duration
                })
                current_time += duration + 0.3
            else:
                # Fingerspell out-of-vocabulary word
                word_upper = word.upper()
                for char in word_upper:
                    if char.isalpha():
                        letter_gloss = f"LETTER_{char}"
                        isl_glosses.append(char)
                        cues.append({
                            "sign": char,
                            "type": "FINGERSPELLING_LETTER",
                            "source_word": word,
                            "start_time": round(current_time, 2),
                            "duration": 0.8
                        })
                        current_time += 0.9
                current_time += 0.3

        gloss_str = " ".join(isl_glosses)
        if is_question and gloss_str:
            gloss_str += " ?"

        modality = "WH_QUESTION" if q_words else ("POLAR_QUESTION" if is_question else "DECLARATIVE_STATEMENT")

        return {
            "english_input": raw_text,
            "isl_gloss_sequence": isl_glosses,
            "gloss_string": gloss_str,
            "is_question": is_question,
            "modality": modality,
            "total_duration_sec": round(current_time, 2),
            "sign_cues": cues
        }

    def _restructure_sov(self, words: List[str]) -> List[str]:
        """
        Converts simple English SVO pattern (e.g. ['i', 'give', 'book'] or ['i', 'meet', 'teacher'])
        to ISL SOV pattern (e.g. ['i', 'book', 'give'] or ['i', 'teacher', 'meet']).
        """
        if len(words) <= 2:
            return words

        # Common transitive verbs
        verbs = {"give", "meet", "drink", "eat", "help", "read", "take", "bring"}
        verb_idx = -1
        for i, w in enumerate(words):
            if w in verbs or self.SYNONYM_MAP.get(w) in ["GIVE", "MEET", "DRINK", "FOOD", "HELP"]:
                verb_idx = i
                break

        # If verb is between subject and object: S V O -> S O V
        if 0 < verb_idx < len(words) - 1:
            subject = words[:verb_idx]
            verb = words[verb_idx]
            obj = words[verb_idx + 1:]
            return subject + obj + [verb]

        return words
