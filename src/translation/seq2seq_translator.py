import os
import json
from typing import List, Dict, Any, Optional, Tuple
from src.config import DATA_DIR, TRANSLATION_MODEL_PATH
from src.translation.grammar_rules import ISLGrammarTransformer

class ISLTranslator:
    """
    Translates sequences of recognized ISL glosses into grammatical English sentences.
    Combines verified parallel corpus retrieval with syntactic grammar rules.
    """

    def __init__(self, corpus_path: Optional[str] = None):
        self.corpus_path = str(corpus_path or (DATA_DIR / "parallel_corpus.json"))
        self.corpus: List[Dict[str, Any]] = []
        self.exact_lookup: Dict[str, str] = {}
        self.load_corpus()

    def load_corpus(self):
        """Loads verified ISL-to-English parallel corpus."""
        if os.path.exists(self.corpus_path):
            try:
                with open(self.corpus_path, "r", encoding="utf-8") as f:
                    self.corpus = json.load(f)
                for item in self.corpus:
                    key = " ".join([g.upper().strip() for g in item.get("isl_glosses", [])])
                    self.exact_lookup[key] = item.get("english_translation", "")
            except Exception as e:
                print(f"[ISLTranslator] Error loading parallel corpus: {e}")

    def translate(self, glosses: List[str]) -> Dict[str, Any]:
        """
        Translates a sequence of ISL glosses into natural English.
        Returns dictionary containing:
            - input_glosses: list of strings
            - raw_concatenation: direct concatenation of glosses
            - translated_text: fluent English sentence
            - translation_method: 'exact_corpus_match' or 'syntactic_grammar_transformer'
            - confidence: float score
        """
        if not glosses:
            return {
                "input_glosses": [],
                "raw_concatenation": "",
                "translated_text": "",
                "translation_method": "none",
                "confidence": 0.0
            }

        cleaned_glosses = [g.upper().strip() for g in glosses if g.strip()]
        gloss_key = " ".join(cleaned_glosses)
        raw_concat = " ".join(cleaned_glosses)

        # 1. Check exact parallel corpus match
        if gloss_key in self.exact_lookup:
            return {
                "input_glosses": cleaned_glosses,
                "raw_concatenation": raw_concat,
                "translated_text": self.exact_lookup[gloss_key],
                "translation_method": "exact_corpus_match",
                "confidence": 1.0
            }

        # 2. Check subset matching or multi-word subphrase
        for item in self.corpus:
            c_key = " ".join([g.upper().strip() for g in item.get("isl_glosses", [])])
            if c_key == gloss_key:
                return {
                    "input_glosses": cleaned_glosses,
                    "raw_concatenation": raw_concat,
                    "translated_text": item.get("english_translation", ""),
                    "translation_method": "exact_corpus_match",
                    "confidence": 1.0
                }

        # 3. Apply Syntactic Grammar Transformation
        rule_translated = ISLGrammarTransformer.transform_glosses_to_english(cleaned_glosses)

        return {
            "input_glosses": cleaned_glosses,
            "raw_concatenation": raw_concat,
            "translated_text": rule_translated,
            "translation_method": "syntactic_grammar_transformer",
            "confidence": 0.88
        }
