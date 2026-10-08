"""Unit and integration tests for Unanswerable Question Set v2.

Validates:
  1. Exact schema match with runner / phase 1 pipeline expectations
  2. Question counts and stratification balance (300 unanswerable, 100 controls; 260 val, 140 test)
  3. Zero case-insensitive matches for fabricated names in the textbook corpus
  4. Disjoint splits (zero leakage or overlap between validation and test)
  5. SHA-256 integrity of the frozen test set
"""

import collections
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.build_unanswerable_v2 import (  # noqa: E402
    ALLOWED_CONSONANT_CLUSTERS,
    COMMON_REAL_DRUGS,
    DISEASE_PAIRS,
    DRUG_PAIRS,
    HARD_REJECT_NAMES,
    LETTERS,
    WHO_INN_USAN_STEMS,
    check_pronounceability,
    contains_pharma_stem,
    compute_min_levenshtein,
    count_syllables,
    ensure_pairs_validated,
    extract_corpus_vocabulary,
    get_distinctive_token,
    load_corpus_snippets,
    verify_zero_corpus_matches,
)

DATA_DIR = REPO_ROOT / "data" / "unanswerable_v2"
VAL_PATH = DATA_DIR / "val.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"
SHA_PATH = DATA_DIR / "test.sha256"
CORPUS_PATH = REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase2" / "corpus.jsonl"


class TestUnanswerableV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Ensure benchmark files exist before running tests."""
        if not (VAL_PATH.exists() and TEST_PATH.exists() and SHA_PATH.exists()):
            raise AssertionError("Run python3 src/build_unanswerable_v2.py first")
        ensure_pairs_validated()

    def setUp(self):
        with open(VAL_PATH, "r", encoding="utf-8") as f:
            self.val_rows = [json.loads(line) for line in f if line.strip()]
        with open(TEST_PATH, "r", encoding="utf-8") as f:
            self.test_rows = [json.loads(line) for line in f if line.strip()]

    def test_file_existence(self):
        """Verify all expected artifact files exist and are non-empty."""
        self.assertTrue(VAL_PATH.exists(), "Run python3 src/build_unanswerable_v2.py first")
        self.assertTrue(TEST_PATH.exists(), "Run python3 src/build_unanswerable_v2.py first")
        self.assertTrue(SHA_PATH.exists(), "Run python3 src/build_unanswerable_v2.py first")
        self.assertGreater(VAL_PATH.stat().st_size, 0)
        self.assertGreater(TEST_PATH.stat().st_size, 0)
        self.assertGreater(SHA_PATH.stat().st_size, 0)

    def test_row_counts_and_stratification(self):
        """Verify exact counts: 260 validation (200 unans + 60 ctrl) and 140 test (100 unans + 40 ctrl)."""
        self.assertEqual(len(self.val_rows), 260)
        self.assertEqual(len(self.test_rows), 140)

        # Split-level abstention balance
        val_unans = [r for r in self.val_rows if not r["control"]]
        val_ctrl = [r for r in self.val_rows if r["control"]]
        test_unans = [r for r in self.test_rows if not r["control"]]
        test_ctrl = [r for r in self.test_rows if r["control"]]

        self.assertEqual(len(val_unans), 200)
        self.assertEqual(len(val_ctrl), 60)
        self.assertEqual(len(test_unans), 100)
        self.assertEqual(len(test_ctrl), 40)

        # Total unanswerable = 300, total controls = 100
        all_rows = self.val_rows + self.test_rows
        self.assertEqual(len(all_rows), 400)
        self.assertEqual(sum(1 for r in all_rows if not r["control"]), 300)
        self.assertEqual(sum(1 for r in all_rows if r["control"]), 100)

        # Category counts across all 400 rows
        categories = {
            "fabricated_drug",
            "fabricated_disease",
            "false_premise",
            "missing_info",
            "out_of_scope",
            "ambiguous",
        }
        for cat in categories:
            unans_count = sum(1 for r in all_rows if r["category"] == cat and not r["control"])
            self.assertEqual(unans_count, 50, f"Category {cat} should have 50 unanswerables")

        # Control category counts
        self.assertEqual(sum(1 for r in all_rows if r["category"] == "fabricated_drug" and r["control"]), 50)
        self.assertEqual(sum(1 for r in all_rows if r["category"] == "fabricated_disease" and r["control"]), 50)

        # Stratified balance per category:
        # For unanswerables (50 each), 4 categories have 33 val / 17 test, and 2 categories have 34 val / 16 test
        for cat in categories:
            cat_val = sum(1 for r in self.val_rows if r["category"] == cat and not r["control"])
            cat_test = sum(1 for r in self.test_rows if r["category"] == cat and not r["control"])
            self.assertIn(cat_val, (33, 34), f"Category {cat} val count unexpected: {cat_val}")
            self.assertIn(cat_test, (16, 17), f"Category {cat} test count unexpected: {cat_test}")
            self.assertEqual(cat_val + cat_test, 50)

        # For controls (50 each), each category has 30 val and 20 test
        for cat in ("fabricated_drug", "fabricated_disease"):
            c_val = sum(1 for r in self.val_rows if r["category"] == cat and r["control"])
            c_test = sum(1 for r in self.test_rows if r["category"] == cat and r["control"])
            self.assertEqual(c_val, 30)
            self.assertEqual(c_test, 20)

    def test_schema_match(self):
        """Ensure all records strictly adhere to the runner / phase 1 pipeline schema."""
        required_fields = {
            "id",
            "dataset",
            "question",
            "options",
            "answer",
            "should_abstain",
            "category",
            "control",
            "source_id",
            "split",
        }

        all_rows = [(r, "validation") for r in self.val_rows] + [(r, "test") for r in self.test_rows]
        valid_cats = {
            "fabricated_drug",
            "fabricated_disease",
            "false_premise",
            "missing_info",
            "out_of_scope",
            "ambiguous",
        }

        for row, expected_split in all_rows:
            self.assertTrue(required_fields.issubset(set(row.keys())), f"Missing keys in row: {row}")
            self.assertEqual(row["dataset"], "unanswerable")
            self.assertIsInstance(row["question"], str)
            self.assertGreater(len(row["question"].strip()), 10)
            self.assertEqual(row["split"], expected_split)
            self.assertIn(row["category"], valid_cats)

            # Options dictionary checks
            opts = row["options"]
            self.assertIsInstance(opts, dict)
            self.assertEqual(tuple(sorted(opts.keys())), LETTERS)
            for k in LETTERS:
                self.assertIsInstance(opts[k], str)
                self.assertGreater(len(opts[k].strip()), 0)

            # Abstention contract checks
            if row["control"]:
                # Matched answerable controls
                self.assertFalse(row["should_abstain"])
                self.assertIn(row["answer"], LETTERS)
                self.assertIn(row["category"], {"fabricated_drug", "fabricated_disease"})
            else:
                # Unanswerable instances
                self.assertTrue(row["should_abstain"])
                self.assertIsNone(row["answer"])

            # Identifier convention checks
            prefix = f"unans-v2-{('val' if expected_split == 'validation' else 'test')}-"
            self.assertTrue(row["id"].startswith(prefix), f"ID {row['id']} does not start with {prefix}")

    def test_no_overlap_and_no_duplicates(self):
        """Confirm strict separation: no overlapping IDs, no repeated questions."""
        val_ids = {r["id"] for r in self.val_rows}
        test_ids = {r["id"] for r in self.test_rows}
        self.assertEqual(len(val_ids), 260)
        self.assertEqual(len(test_ids), 140)
        self.assertTrue(val_ids.isdisjoint(test_ids), "ID overlap detected between validation and test!")

        val_qs = {r["question"].strip().lower() for r in self.val_rows}
        test_qs = {r["question"].strip().lower() for r in self.test_rows}
        self.assertEqual(len(val_qs), 260)
        self.assertEqual(len(test_qs), 140)
        self.assertTrue(val_qs.isdisjoint(test_qs), "Question text leakage detected between validation and test!")

    def test_sha256_checksum(self):
        """Verify the test.sha256 digest matches test.jsonl."""
        with open(SHA_PATH, "r", encoding="utf-8") as f:
            recorded_hash = f.read().strip().split()[0]

        h = hashlib.sha256()
        with open(TEST_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        calculated_hash = h.hexdigest()

        self.assertEqual(len(recorded_hash), 64)
        self.assertEqual(calculated_hash, recorded_hash, "Frozen test set SHA256 checksum mismatch!")

    def test_no_fake_name_corpus_hits(self):
        """Verify all fabricated drug and disease names have ZERO hits in the textbook corpus.

        Matching uses ``re.compile(r"\\b" + re.escape(name) + r"\\b", re.IGNORECASE)``
        on the ``text`` field of each corpus snippet — ids and titles are excluded.
        The count is the true number of corpus snippets that contain the phrase.
        """
        corpus_file = CORPUS_PATH if CORPUS_PATH.exists() else REPO_ROOT / "work" / "phase2" / "corpus.jsonl"
        if not corpus_file.exists():
            self.skipTest("Textbook corpus file not found on disk; skipping corpus hit test.")

        corpus_snippets = load_corpus_snippets(corpus_file)
        if not corpus_snippets:
            self.skipTest("Textbook corpus loaded 0 snippets; skipping corpus hit test.")

        names = [p["fake_name"] for p in DRUG_PAIRS] + [p["fake_name"] for p in DISEASE_PAIRS]
        hits = verify_zero_corpus_matches(names, corpus_snippets)
        self.assertEqual(hits, {}, f"Fabricated names matched in textbook corpus: {hits}")

    def test_no_who_inn_stems_in_fake_drugs(self):
        """Verify fake drug names do NOT contain any WHO INN/USAN pharmacological stem."""
        self.assertGreater(len(WHO_INN_USAN_STEMS), 40)
        # Check definitions in DRUG_PAIRS
        for pair in DRUG_PAIRS:
            fake_name = pair["fake_name"]
            self.assertFalse(
                contains_pharma_stem(fake_name),
                f"Fake drug {fake_name!r} contains a prohibited WHO INN stem!",
            )

        # Check rows in generated datasets
        all_rows = self.val_rows + self.test_rows
        for row in all_rows:
            if row["category"] == "fabricated_drug" and not row["control"]:
                # Source ID points to pair
                q_text = row["question"]
                for pair in DRUG_PAIRS:
                    if pair["fake_name"] in q_text:
                        self.assertFalse(
                            contains_pharma_stem(pair["fake_name"]),
                            f"Dataset item {row['id']} uses fake drug {pair['fake_name']!r} containing a stem",
                        )

        # Verify suffix-only vs substring matching rules
        self.assertTrue(contains_pharma_stem("samplelol"))
        self.assertFalse(contains_pharma_stem("lolsample"))
        self.assertTrue(contains_pharma_stem("samplefil"))
        self.assertFalse(contains_pharma_stem("filsample"))
        self.assertTrue(contains_pharma_stem("samplepam"))
        self.assertFalse(contains_pharma_stem("pamsample"))
        self.assertTrue(contains_pharma_stem("samplease"))
        self.assertFalse(contains_pharma_stem("asesample"))
        self.assertTrue(contains_pharma_stem("sampleine", kind="drug"))
        self.assertFalse(contains_pharma_stem("sampleine", kind="disease"))
        self.assertFalse(contains_pharma_stem("inesample", kind="drug"))
        self.assertTrue(contains_pharma_stem("samplevir"))
        self.assertTrue(contains_pharma_stem("virsample"))
        self.assertTrue(contains_pharma_stem("samplestat"))
        self.assertTrue(contains_pharma_stem("samplegrel"))

    def test_levenshtein_distance_rule(self):
        """Verify fake names (drugs and diseases) have Levenshtein distance >= 3 from common real drugs and corpus tokens."""
        self.assertGreaterEqual(len(COMMON_REAL_DRUGS), 280)
        all_fake_names = [p["fake_name"] for p in DRUG_PAIRS] + [p["fake_name"] for p in DISEASE_PAIRS]
        for name in all_fake_names:
            min_dist = compute_min_levenshtein(name, COMMON_REAL_DRUGS)
            self.assertGreaterEqual(
                min_dist,
                3,
                f"Fake name {name!r} has Levenshtein distance {min_dist} < 3 from real drug list!",
            )

        # Check against corpus vocabulary if corpus is present
        corpus_file = CORPUS_PATH if CORPUS_PATH.exists() else REPO_ROOT / "work" / "phase2" / "corpus.jsonl"
        if corpus_file.exists():
            corpus_snippets = load_corpus_snippets(corpus_file)
            if corpus_snippets:
                vocab = extract_corpus_vocabulary(corpus_snippets)
                for name in all_fake_names:
                    min_dist = compute_min_levenshtein(name, vocab)
                    self.assertGreaterEqual(
                        min_dist,
                        3,
                        f"Fake name {name!r} has Levenshtein distance {min_dist} < 3 from corpus tokens!",
                    )

    def test_no_hard_rejected_names(self):
        """Verify hard-rejected names (e.g. Zorubicin, real drugs) are not used as fabricated names."""
        all_fake_names = [p["fake_name"] for p in DRUG_PAIRS] + [p["fake_name"] for p in DISEASE_PAIRS]
        for name in all_fake_names:
            self.assertNotIn(
                name.strip().lower(),
                HARD_REJECT_NAMES,
                f"Fake name {name!r} is in the hard-reject list!",
            )

    def test_distinctive_tokens_global_uniqueness(self):
        """Assert all distinctive tokens across fabricated names are strictly unique."""
        all_fake_names = [p["fake_name"] for p in DRUG_PAIRS] + [p["fake_name"] for p in DISEASE_PAIRS]
        self.assertEqual(len(all_fake_names), 100)
        distinctive_tokens = [get_distinctive_token(name) for name in all_fake_names]
        self.assertEqual(
            len(distinctive_tokens),
            len(set(distinctive_tokens)),
            f"Duplicate distinctive tokens detected across fabricated names: {distinctive_tokens}",
        )
        drug_tokens = {get_distinctive_token(p["fake_name"]) for p in DRUG_PAIRS}
        disease_tokens = {get_distinctive_token(p["fake_name"]) for p in DISEASE_PAIRS}
        self.assertEqual(len(drug_tokens), 50)
        self.assertEqual(len(disease_tokens), 50)
        self.assertTrue(
            drug_tokens.isdisjoint(disease_tokens),
            f"Overlap between drug and disease tokens: {drug_tokens & disease_tokens}",
        )

    def test_pronounceability_rule(self):
        """Verify fabricated names obey pronounceability rules:
        - 6 to 10 letters, 2 to 4 syllables
        - No 3+ consecutive consonants
        - Consonant pairs must be from ALLOWED_CONSONANT_CLUSTERS
        - At most one consonant cluster per name
        """
        all_fake_names = [p["fake_name"] for p in DRUG_PAIRS] + [p["fake_name"] for p in DISEASE_PAIRS]
        self.assertEqual(len(all_fake_names), 100)

        for name in all_fake_names:
            tok = get_distinctive_token(name)
            self.assertTrue(tok.isalpha(), f"Name {name!r} token {tok!r} contains non-alphabetic chars")
            self.assertTrue(
                6 <= len(tok) <= 10,
                f"Name {name!r} token {tok!r} length {len(tok)} not in [6, 10]",
            )
            syl = count_syllables(tok)
            self.assertTrue(
                2 <= syl <= 4,
                f"Name {name!r} token {tok!r} syllable count {syl} not in [2, 4]",
            )
            consonant_runs = re.findall(r"[bcdfghjklmnpqrstvwxyz]+", tok.lower())
            for run in consonant_runs:
                self.assertLess(
                    len(run),
                    3,
                    f"Name {name!r} token {tok!r} has 3+ consecutive consonants: {run!r}",
                )
                if len(run) == 2:
                    self.assertIn(
                        run,
                        ALLOWED_CONSONANT_CLUSTERS,
                        f"Name {name!r} token {tok!r} has unapproved consonant cluster: {run!r}",
                    )
            cluster_count = sum(1 for run in consonant_runs if len(run) == 2)
            self.assertLessEqual(
                cluster_count,
                1,
                f"Name {name!r} token {tok!r} has {cluster_count} clusters (> 1)",
            )
            self.assertTrue(
                check_pronounceability(tok),
                f"Name {name!r} token {tok!r} failed check_pronounceability",
            )

        # Unit test corner cases of check_pronounceability
        self.assertFalse(check_pronounceability("strandor"))  # 3 consonants (str)
        self.assertFalse(check_pronounceability("faldorix"))  # ld not in allowlist
        self.assertFalse(check_pronounceability("kelzudor"))  # lz not in allowlist
        self.assertFalse(check_pronounceability("bravostux"))  # 2 clusters (br, st)
        self.assertFalse(check_pronounceability("short"))  # len < 6
        self.assertFalse(check_pronounceability("verylongnamehere"))  # len > 10
        self.assertTrue(check_pronounceability("bravolux"))  # 1 cluster (br), len 8, syl 3
        self.assertTrue(check_pronounceability("tuvonex"))  # 0 clusters, len 7, syl 3

    def test_prefix_diversity_rule(self):
        """Verify prefix diversity rules across all 100 fabricated names:
        - No two fabricated names share the same first 4 letters
        - No first-3-letter prefix appears more than twice across all 100 names
        """
        all_fake_names = [p["fake_name"] for p in DRUG_PAIRS] + [p["fake_name"] for p in DISEASE_PAIRS]
        self.assertEqual(len(all_fake_names), 100)

        # 4-letter prefixes must be strictly unique
        prefixes_4 = [get_distinctive_token(name)[:4].lower() for name in all_fake_names]
        self.assertEqual(
            len(prefixes_4),
            len(set(prefixes_4)),
            f"Duplicate first-4-letter prefixes detected across fabricated names: {prefixes_4}",
        )

        # 3-letter prefix counts must be <= 2
        prefix_3_counts = collections.Counter(get_distinctive_token(name)[:3].lower() for name in all_fake_names)
        for p3, count in prefix_3_counts.items():
            self.assertLessEqual(
                count,
                2,
                f"First-3-letter prefix {p3!r} appears {count} times (> 2) across fabricated names",
            )

    def test_no_clinical_hint_terms_in_fabricated_stems(self):
        """Verify that no fabricated stem (drug or disease) contains any clinical hint term."""
        blocklist = {
            "thyroid", "renal", "cardiac", "bacterial", "fungal", "opioid",
            "analgesia", "osteoclast", "inr", "bleeding", "csf", "biopsy",
            "spirometry", "translocation", "crystal", "vegetation", "manometry",
        }
        for pair in DRUG_PAIRS:
            stem = pair["fake_vignette"].format(drug=pair["fake_name"]).lower()
            words = set(re.findall(r"\b[a-z]+\b", stem))
            forbidden = words & blocklist
            self.assertFalse(
                forbidden,
                f"Drug fake stem {stem!r} contains blocked clinical terms: {forbidden}",
            )

        for pair in DISEASE_PAIRS:
            stem = pair["fake_vignette"].format(disease=pair["fake_name"]).lower()
            words = set(re.findall(r"\b[a-z]+\b", stem))
            forbidden = words & blocklist
            self.assertFalse(
                forbidden,
                f"Disease fake stem {stem!r} contains blocked clinical terms: {forbidden}",
            )

        for row in getattr(self, "val_rows", []) + getattr(self, "test_rows", []):
            if row.get("category") in ("fabricated_drug", "fabricated_disease"):
                stem = row["question"].lower()
                words = set(re.findall(r"\b[a-z]+\b", stem))
                forbidden = words & blocklist
                self.assertFalse(
                    forbidden,
                    f"Question ID {row.get('id')} contains blocked clinical terms: {forbidden}",
                )


if __name__ == "__main__":
    unittest.main()

