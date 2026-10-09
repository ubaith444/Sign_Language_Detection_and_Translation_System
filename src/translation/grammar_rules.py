from typing import List, Dict, Optional, Tuple

class ISLGrammarTransformer:
    """
    Syntactic transformation engine that maps Indian Sign Language (ISL)
    Subject-Object-Verb (SOV) and Topic-Comment gloss sequences into
    grammatically well-formed English Subject-Verb-Object (SVO) sentences.
    """

    PRONOUNS_SUBJECT = {
        "ME": "I",
        "YOU": "You",
        "HE": "He",
        "SHE": "She",
        "WE": "We",
        "THEY": "They"
    }

    PRONOUNS_OBJECT = {
        "ME": "me",
        "YOU": "you",
        "HE": "him",
        "SHE": "her",
        "WE": "us",
        "THEY": "them"
    }

    PRONOUNS_POSSESSIVE = {
        "ME": "my",
        "YOU": "your",
        "HE": "his",
        "SHE": "her",
        "WE": "our",
        "THEY": "their"
    }

    VERBS = {
        "MEET": {"base": "meet", "present_cont": "am meeting", "past": "met"},
        "GO": {"base": "go", "present_cont": "is going", "past": "went"},
        "COME": {"base": "come", "present_cont": "is coming", "past": "came"},
        "WANT": {"base": "want", "present_cont": "wants", "past": "wanted"},
        "NEED": {"base": "need", "present_cont": "needs", "past": "needed"},
        "EAT": {"base": "eat", "present_cont": "is eating", "past": "ate"},
        "DRINK": {"base": "drink", "present_cont": "is drinking", "past": "drank"},
        "HELP": {"base": "help", "present_cont": "is helping", "past": "helped"},
        "CALL": {"base": "call", "present_cont": "is calling", "past": "called"},
        "GIVE": {"base": "give", "present_cont": "is giving", "past": "gave"},
        "AGREE": {"base": "agree", "present_cont": "agrees", "past": "agreed"},
        "REJECT": {"base": "reject", "present_cont": "rejects", "past": "rejected"},
        "WISH": {"base": "wish", "present_cont": "wishes", "past": "wished"}
    }

    NOUNS = {
        "WATER": {"article": "some", "noun": "water"},
        "FOOD": {"article": "some", "noun": "food"},
        "BOOK": {"article": "the", "noun": "book"},
        "HOUSE": {"article": "the", "noun": "house"},
        "TEACHER": {"article": "the", "noun": "teacher"},
        "DOCTOR": {"article": "a", "noun": "doctor"},
        "STUDENT": {"article": "a", "noun": "student"},
        "FRIEND": {"article": "my", "noun": "friend"},
        "WORLD": {"article": "the", "noun": "world"},
        "NAME": {"article": "", "noun": "name"},
        "PROBLEM": {"article": "a", "noun": "problem"}
    }

    QUESTION_WORDS = {
        "WHAT": "What",
        "WHERE": "Where",
        "HOW_ARE_YOU": "How are you",
        "WHEN": "When",
        "WHY": "Why",
        "WHO": "Who"
    }

    @classmethod
    def transform_glosses_to_english(cls, glosses: List[str]) -> str:
        """
        Transforms an ordered list of ISL glosses into a natural English sentence.
        Handles SOV to SVO inversion, wh-question fronting, copula insertion,
        and morphological agreements.
        """
        if not glosses:
            return ""

        # Normalize glosses
        tokens = [g.upper().strip() for g in glosses if g.strip()]
        if not tokens:
            return ""

        # 1. Single sign fast paths
        if len(tokens) == 1:
            return cls._transform_single_sign(tokens[0])

        # 2. Question Handling (WH-word typically sentence-final in ISL)
        wh_word = None
        for q in cls.QUESTION_WORDS:
            if q in tokens:
                wh_word = q
                break

        if wh_word:
            return cls._handle_question_structure(tokens, wh_word)

        # 3. Imperative / Request Handling ("PLEASE HELP ME", "FRIEND COME PLEASE")
        if "PLEASE" in tokens or "HELP" in tokens and "ME" in tokens:
            return cls._handle_imperative_request(tokens)

        # 4. Identity / Copula insertion ("ME STUDENT", "HE FRIEND")
        if len(tokens) == 2 and tokens[0] in cls.PRONOUNS_SUBJECT and tokens[1] in cls.NOUNS:
            subject = cls.PRONOUNS_SUBJECT[tokens[0]]
            noun_info = cls.NOUNS[tokens[1]]
            copula = "am" if subject == "I" else ("is" if subject in ["He", "She"] else "are")
            article = noun_info["article"]
            noun = noun_info["noun"]
            return f"{subject} {copula} {article} {noun}."

        # 5. SOV to SVO Clause Transformation (e.g. ["ME", "WATER", "WANT"] -> "I want some water.")
        transformed = cls._handle_sov_clause(tokens)
        if transformed:
            return transformed

        # 6. Fallback General Transformation: Title-case, join, and punctuate
        words = []
        for i, t in enumerate(tokens):
            if i == 0 and t in cls.PRONOUNS_SUBJECT:
                words.append(cls.PRONOUNS_SUBJECT[t])
            elif t in cls.NOUNS:
                words.append(cls.NOUNS[t]["noun"])
            elif t in cls.VERBS:
                words.append(cls.VERBS[t]["base"])
            else:
                words.append(t.lower())

        sentence = " ".join(words)
        return sentence.capitalize() + "."

    @classmethod
    def _transform_single_sign(cls, token: str) -> str:
        """Transforms solitary ISL signs into appropriate English utterances."""
        mappings = {
            "NAMASTE": "Namaste, greetings and welcome.",
            "HELLO": "Hello!",
            "GOODBYE": "Goodbye!",
            "THANK_YOU": "Thank you very much.",
            "SORRY": "I am sorry.",
            "PLEASE": "Please.",
            "HELP": "Help is requested.",
            "YES": "Yes.",
            "NO": "No.",
            "PEACE": "Peace.",
            "WATER": "Water.",
            "FOOD": "Food.",
            "HOW_ARE_YOU": "How are you doing today?",
            "WHAT": "What is that?",
            "WHERE": "Where is it located?",
            "HOUSE": "House.",
            "BOOK": "Book."
        }
        return mappings.get(token, token.capitalize() + ".")

    @classmethod
    def _handle_question_structure(cls, tokens: List[str], wh_word: str) -> str:
        """
        Reorders ISL sentence-final question syntax into English WH-fronted syntax.
        Examples:
            ["YOU", "NAME", "WHAT"] -> "What is your name?"
            ["YOU", "GO", "WHERE"]  -> "Where are you going?"
        """
        remaining = [t for t in tokens if t != wh_word]

        if wh_word == "WHAT":
            if "NAME" in remaining:
                if "YOU" in remaining:
                    return "What is your name?"
                return "What is the name?"
            return "What is happening?"

        if wh_word == "WHERE":
            if "GO" in remaining:
                if "YOU" in remaining:
                    return "Where are you going?"
                if "HE" in remaining or "SHE" in remaining:
                    return "Where is he going?"
                return "Where are you going?"
            return "Where is that located?"

        if wh_word == "HOW_ARE_YOU":
            return "How are you doing today?"

        return f"{cls.QUESTION_WORDS.get(wh_word, 'What')} is {' '.join(remaining).lower()}?"

    @classmethod
    def _handle_imperative_request(cls, tokens: List[str]) -> str:
        """
        Processes requests and polite imperatives.
        Examples:
            ["PLEASE", "HELP", "ME"] -> "Please help me."
            ["BOOK", "ME", "GIVE"]   -> "Please give me the book."
        """
        has_please = "PLEASE" in tokens
        cleaned = [t for t in tokens if t != "PLEASE"]

        if "HELP" in cleaned and "ME" in cleaned:
            return "Please help me." if has_please else "Help me, please."

        if "DOCTOR" in cleaned and ("CALL" in cleaned or "COME" in cleaned):
            return "Please call a doctor immediately."

        if "COME" in cleaned and "FRIEND" in cleaned:
            return "Please come over, my friend."

        return "Please " + " ".join(cleaned).lower() + "."

    @classmethod
    def _handle_sov_clause(cls, tokens: List[str]) -> Optional[str]:
        """
        Identifies Subject, Object, Verb constituents in tokens
        and synthesizes an SVO English clause.
        Example:
            ["ME", "TEACHER", "MEET"] -> "I am meeting the teacher."
            ["ME", "WATER", "WANT"]   -> "I would like some water."
        """
        subject = None
        obj = None
        verb = None

        for t in tokens:
            if t in cls.PRONOUNS_SUBJECT and subject is None:
                subject = t
            elif t in cls.VERBS and verb is None:
                verb = t
            elif t in cls.NOUNS and obj is None:
                obj = t

        if subject and verb and obj:
            subj_en = cls.PRONOUNS_SUBJECT[subject]
            verb_info = cls.VERBS[verb]
            obj_info = cls.NOUNS[obj]

            # Aspect agreement
            if verb == "WANT":
                verb_str = "would like"
            elif verb == "NEED":
                verb_str = "need"
            elif verb in ["MEET", "GO", "COME"]:
                verb_str = verb_info["present_cont"]
            else:
                verb_str = verb_info["base"]

            article = obj_info["article"]
            noun = obj_info["noun"]
            obj_str = f"{article} {noun}".strip()

            return f"{subj_en} {verb_str} {obj_str}."

        if subject and verb:
            subj_en = cls.PRONOUNS_SUBJECT[subject]
            verb_info = cls.VERBS[verb]
            return f"{subj_en} {verb_info['present_cont']}."

        return None
