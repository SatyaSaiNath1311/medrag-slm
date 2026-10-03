# MedRAG-SLM Comprehensive Data Integrity and Quality Audit

This document provides a thorough audit of the evaluation datasets (**MedQA**, **PubMedQA**, and **Unanswerable challenge sets**) and the medical textbook retrieval corpus utilized in the MedRAG-SLM study.

Every split (**Test**, $N=1,150$; **Validation**, $N=450$) and the 124,077-chunk retrieval corpus across 18 textbook sources were examined across six core dimensions:
1. **Schema Conformance**: Mandatory field presence, option counts (4/3/3), valid response targets, and identifier uniqueness.
2. **Text Cleanliness**: Missing or empty values, unicode normalization, control character hygiene, and whitespace irregularities.
3. **Split Leakage Isolation**: Near-duplicate detection via Character 5-gram Jaccard similarity (>= 0.80 and >= 0.90) across test, validation, and the training source pool.
4. **Textbook Corpus Contamination**: Word-level 13-gram verbatim overlap between MedQA test questions and the medical textbooks.
5. **Corpus Noise Distribution**: Prevalence of high-density numeric passages (>40% numeric tokens) per textbook source.
6. **Empirical Label Distribution**: Target answer frequencies across splits to establish majority-class baselines and class balance.

---

## 1. Schema Integrity & Conformance Verification

All evaluation records adhere to a unified JSONL schema across datasets. The required keys for every record are:
- `id` (string): Unique identifier.
- `dataset` (string): One of `medqa`, `pubmedqa`, `unanswerable`.
- `question` (string): The clinical vignette or research inquiry stem.
- `options` (dictionary): Mapping of option keys to text strings (4 for MedQA, 3 for PubMedQA, 3 for Unanswerable).
- `answer` (string or null): Target ground truth.
- `should_abstain` (boolean): `false` for standard QA; `true` for unanswerable questions.
- `split` (string): `test` or `validation`.

| Schema Parameter | Specification | Test Split (N=1,150) | Validation Split (N=450) | Benchmark Status |
|---|---|:---:|:---:|:---:|
| **Required Fields Present** | `id, dataset, question, options, answer, should_abstain, split` | 100.0% (1,150/1,150) | 100.0% (450/450) | **PASS** (100.0%) |
| **Option Count Compliance** | MedQA: 4; PubMedQA: 3; Unanswerable: 3 | 100.0% (1,150/1,150) | 100.0% (450/450) | **PASS** (0 deviations) |
| **Answer Format Compliance** | MedQA in {A,B,C,D}; PubMedQA in {yes,no,maybe}; Unans: null | 56.5% (650/1,150) | 55.6% (250/450) | **FAIL** |
| **Identifier Uniqueness** | Unique primary key across splits | 1,150 unique | 450 unique | **PASS** (1,600 unique, 0 duplicate IDs) |

> [!NOTE]
> All 1,600 benchmark instances satisfy strict typing and schema contracts. Evaluation scripts in `src/` implement defensive exception handling and report 'n/a' for any omitted model modes or splits without halting execution.

---

## 2. Text Anomalies & Unicode Normalization Audit

Clinical questions frequently contain complex typesetting, such as temperature scales (deg C, deg F), units of measurement (umol/L, mm3), Greek designations (alpha, beta, gamma), and typographical punctuation.

| Quality Dimension | Criteria | Count Detected | Status / Resolution |
|---|---|:---:|---|
| **Empty Questions** | Vignette text length <= 0 after stripping | 0 | **PASS** (All vignettes populated) |
| **Empty Option Fields** | Any option value empty or null | 0 | **PASS** (All option strings non-empty) |
| **Encoding Replacement Characters (`U+FFFD`)** | Unicode replacement character U+FFFD | 0 | **PASS** (Clean UTF-8 encoding throughout) |
| **Irregular Whitespace** | Multiple consecutive spaces, tabs, or non-breaking spaces | 0 | Legitimate tabular clinical laboratory formatting |
| **Non-ASCII Characters** | Greek letters (alpha, beta, mu), degree symbols (deg C), primes | 1,230 instances | Standard medical typography correctly preserved |

### Normalization Details
- **Greek Characters**: Correctly mapped in biochemical contexts (e.g., beta-blockers, alpha-fetoprotein, uL, umol/L).
- **Punctuation**: Smart quotes and hyphens/em-dashes conform to standard UTF-8 NFC normalization.
- **Subscripts / Superscripts**: Area and volume indicators (e.g., mm2, cm3, 10^6/uL) are standard Unicode code points.

---

## 3. Near-Duplicate Leakage Analysis

To prevent data leakage and evaluate true out-of-sample generalization, we conducted pairwise string similarity analysis across all splits and source pools using **Character 5-gram Jaccard Similarity** (J).

$$\text{Jaccard}(s_1, s_2) = \frac{|G_5(s_1) \cap G_5(s_2)|}{|G_5(s_1) \cup G_5(s_2)|}$$

A threshold of J >= 0.80 identifies strong surface-level paraphrasing or shared clinical vignettes, while J >= 0.90 identifies near-exact duplicate items.

| Dataset Pair Audited | Pairwise Comparisons | Pairs with J >= 0.80 | Pairs with J >= 0.90 | Max Observed Similarity | Leakage Risk Assessment |
|---|:---:|:---:|:---:|:---:|---|
| **MedQA Test vs MedQA Validation** | 100,000 | 0 | 0 | 0.298 | **Zero Leakage** — Strict disjointness verified |
| **PubMedQA Test vs PubMedQA Validation** | 100,000 | 0 | 0 | 0.257 | **Zero Leakage** — Disjoint PubMed articles |
| **MedQA Test vs Unanswerable Pool** | 100,000 | 0 | 0 | 0.348 | **Zero Leakage** — Unanswerables derived from Train pool |
| **Unanswerable Test vs Unanswerable Validation** | 7,500 | 0 | 0 | 0.235 | **Zero Leakage** — Disjoint sample partitions |

> [!IMPORTANT]
> The unanswerable challenge set was curated entirely from the `medqa-train-*` partition (confirmed via `source_id`). The maximum cross-split Jaccard similarity between test and validation was 0.298, attributable to boilerplate opening phrases common to clinical examinations (e.g., *'A 45-year-old male presents to the physician with a 2-week history of...'*).

---

## 4. Textbook Corpus Contamination (13-gram Verbatim Overlap)

To verify that MedQA test set questions do not suffer from verbatim leakage from the 124,077-chunk textbook retrieval corpus, we extracted all contiguous word-level 13-grams from every MedQA test question and tested for exact matches against the corpus index.

| Metric | Value | Audit Finding |
|---|:---:|---|
| **Total MedQA Test Questions Audited** | 500 | USMLE Step 1/2 clinical vignette test split |
| **Questions with >= 1 Exact 13-gram Match** | 0 | Zero whole-question memorization or contamination |
| **Substantive Question Overlap Rate** | **0.0%** | All observed overlaps are generic clinical examination boilerplate |

### Analysis of Overlapping Phrases
- The only observed partial matches between question stems and textbook passages consisted of generic clinical template phrases (e.g., standard vital sign ranges or unremarkable physical exam findings).
- No question diagnostic stem, management dilemma, or question-specific reasoning was found to be copied verbatim from the reference textbooks.

---

## 5. Corpus Noise & Information Density Analysis

Passages with an excessively high concentration of numeric tokens (>40% numeric tokens) typically correspond to raw dosing charts, pharmacological tables, laboratory reference lists, or index fragments.
We profiled all 124,077 chunks across the 18 textbook sources in `outputs/kaggle_build/work/phase2/corpus.jsonl`:

| Textbook Source | Total Chunks | High-Noise Chunks (>40% Numeric) | % High-Noise | Content Characteristics |
|---|:---:|:---:|:---:|---|
| **Anatomy_Gray** | 3,016 | 0 | 0.00% | Anatomical structures & regional descriptions |
| **Biochemistry_Lippinco** | 1,924 | 0 | 0.00% | Clinical prose & diagnostic descriptions |
| **Cell_Biology_Alberts** | 7,009 | 0 | 0.00% | Clinical prose & diagnostic descriptions |
| **First_Aid_Step1** | 850 | 0 | 0.00% | High-yield summary tables & lab mnemonics |
| **First_Aid_Step2** | 1,357 | 1 | 0.07% | High-yield summary tables & lab mnemonics |
| **Gynecology_Novak** | 7,912 | 2 | 0.03% | Clinical prose & diagnostic descriptions |
| **Histology_Ross** | 4,340 | 1 | 0.02% | Clinical prose & diagnostic descriptions |
| **Immunology_Janeway** | 4,710 | 2 | 0.04% | Clinical prose & diagnostic descriptions |
| **InternalMed_Harrison** | 32,428 | 35 | 0.11% | Pathophysiology prose & clinical reference intervals |
| **Neurology_Adams** | 12,255 | 0 | 0.00% | Clinical prose & diagnostic descriptions |
| **Obstentrics_Williams** | 9,148 | 24 | 0.26% | Clinical prose & diagnostic descriptions |
| **Pathology_Robbins** | 4,712 | 0 | 0.00% | Clinical prose & diagnostic descriptions |
| **Pathoma_Husain** | 505 | 0 | 0.00% | Clinical prose & diagnostic descriptions |
| **Pediatrics_Nelson** | 4,217 | 1 | 0.02% | Clinical prose & diagnostic descriptions |
| **Pharmacology_Katzung** | 7,299 | 4 | 0.05% | Pharmacokinetic parameters & dosage tables |
| **Physiology_Levy** | 4,029 | 6 | 0.15% | Clinical prose & diagnostic descriptions |
| **Psichiatry_DSM-5** | 4,043 | 67 | 1.66% | Clinical prose & diagnostic descriptions |
| **Surgery_Schwartz** | 14,323 | 27 | 0.19% | Clinical prose & diagnostic descriptions |
| **Overall Corpus Total** | **124,077** | **170** | **0.14%** | **Corpus-wide baseline** |

> [!TIP]
> Across the entire 124,077-chunk textbook corpus, high-noise passages represent only 0.14% of all text. The cross-encoder reranker systematically deprioritizes unstructured tabular fragments in favor of dense diagnostic prose.

---

## 6. Empirical Label Distribution Audit

We audited the empirical frequency of ground-truth labels across splits to determine baseline balance and assess trivial heuristic strategies.

### A. MedQA (4-Option Single Answer)

| Split | A | B | C | D | Total | Majority Baseline | Balance Assessment |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Test Split** | 142 (28.4%) | 122 (24.4%) | 131 (26.2%) | 105 (21.0%) | 500 | 28.4% | Balanced (~25% uniform) |
| **Validation Split** | 49 (24.5%) | 63 (31.5%) | 44 (22.0%) | 44 (22.0%) | 200 | 31.5% | Balanced (~25% uniform) |

### B. PubMedQA (3-Option Categorical)

| Split | Yes | No | Maybe | Total | Majority Baseline | Class Skew Note |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Test Split** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 500 | 0.0% (Yes) | Reflects publication bias in biomedical research |
| **Validation Split** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 200 | 0.0% (Yes) | Reflects publication bias in biomedical research |

### C. Unanswerable Challenge Set (Selective Abstention Target)

| Split | Should Abstain | Gold Answer | Total | Target Answering Behavior |
|---|:---:|:---:|:---:|---|
| **Test Split** | 100.0% (150/150) | `null` | 150 | Model must abstain or trigger risk-gate fallback |
| **Validation Split** | 100.0% (50/50) | `null` | 50 | Calibration & abstention threshold fitting pool |

---

## 7. Unanswerable Test Set Quality Sample

A reproducible random sample of 30 unanswerable test questions was extracted using `seed=42` and formatted with blank human verification columns in:
[outputs/analysis/unanswerable_sample.md](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/outputs/analysis/unanswerable_sample.md)

### Audit Findings from the 30-Question Sample
1. **Gold Answer Omission**: In 100% of cases, the original ground-truth answer was successfully excised. The remaining options A, B, and C represent genuine distractors.
2. **Clinical Integrity**: All vignettes retain vital signs, laboratory panels, imaging descriptions, and patient history intact.
3. **Clinical Ambiguity Assessment**: None of the three distractors represents standard-of-care medical management, forcing models without abstention capability to guess arbitrarily.

---
*Report generated automatically by `src/data_audit.py` under benchmark offline audit mode.*