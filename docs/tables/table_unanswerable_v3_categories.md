| Matched Pair / Arm | Model       | Mode          | Category           | N Unans | CWA (conf>=0.90) % | Recall @ 5% FAR % |
| ------------------ | ----------- | ------------- | ------------------ | ------- | ------------------ | ----------------- |
| Pair A (Generic)   | qwen3-1.7b  | Baseline (C4) | ambiguous          | 17      | 94.1% (16/17)      | 64.7% (11/17)     |
| Pair A (Generic)   | qwen3-1.7b  | Baseline (C4) | fabricated_disease | 17      | 100.0% (17/17)     | 76.5% (13/17)     |
| Pair A (Generic)   | qwen3-1.7b  | Baseline (C4) | fabricated_drug    | 17      | 94.1% (16/17)      | 58.8% (10/17)     |
| Pair A (Generic)   | qwen3-1.7b  | Baseline (C4) | false_premise      | 17      | 88.2% (15/17)      | 64.7% (11/17)     |
| Pair A (Generic)   | qwen3-1.7b  | Baseline (C4) | missing_info       | 16      | 81.2% (13/16)      | 75.0% (12/16)     |
| Pair A (Generic)   | qwen3-1.7b  | Baseline (C4) | out_of_scope       | 16      | 87.5% (14/16)      | 68.8% (11/16)     |
| Pair A (Generic)   | qwen3-1.7b  | RAG (C5)      | ambiguous          | 17      | 76.5% (13/17)      | 64.7% (11/17)     |
| Pair A (Generic)   | qwen3-1.7b  | RAG (C5)      | fabricated_disease | 17      | 88.2% (15/17)      | 52.9% (9/17)      |
| Pair A (Generic)   | qwen3-1.7b  | RAG (C5)      | fabricated_drug    | 17      | 94.1% (16/17)      | 35.3% (6/17)      |
| Pair A (Generic)   | qwen3-1.7b  | RAG (C5)      | false_premise      | 17      | 76.5% (13/17)      | 94.1% (16/17)     |
| Pair A (Generic)   | qwen3-1.7b  | RAG (C5)      | missing_info       | 16      | 87.5% (14/16)      | 62.5% (10/16)     |
| Pair A (Generic)   | qwen3-1.7b  | RAG (C5)      | out_of_scope       | 16      | 87.5% (14/16)      | 56.2% (9/16)      |
| Pair A (Medical)   | medpsy-1.7b | Baseline (C4) | ambiguous          | 17      | 94.1% (16/17)      | -                 |
| Pair A (Medical)   | medpsy-1.7b | Baseline (C4) | fabricated_disease | 17      | 100.0% (17/17)     | -                 |
| Pair A (Medical)   | medpsy-1.7b | Baseline (C4) | fabricated_drug    | 17      | 100.0% (17/17)     | -                 |
| Pair A (Medical)   | medpsy-1.7b | Baseline (C4) | false_premise      | 17      | 76.5% (13/17)      | -                 |
| Pair A (Medical)   | medpsy-1.7b | Baseline (C4) | missing_info       | 16      | 100.0% (16/16)     | -                 |
| Pair A (Medical)   | medpsy-1.7b | Baseline (C4) | out_of_scope       | 16      | 93.8% (15/16)      | -                 |
| Pair A (Medical)   | medpsy-1.7b | RAG (C5)      | ambiguous          | 17      | 64.7% (11/17)      | -                 |
| Pair A (Medical)   | medpsy-1.7b | RAG (C5)      | fabricated_disease | 17      | 88.2% (15/17)      | -                 |
| Pair A (Medical)   | medpsy-1.7b | RAG (C5)      | fabricated_drug    | 17      | 94.1% (16/17)      | -                 |
| Pair A (Medical)   | medpsy-1.7b | RAG (C5)      | false_premise      | 17      | 70.6% (12/17)      | -                 |
| Pair A (Medical)   | medpsy-1.7b | RAG (C5)      | missing_info       | 16      | 81.2% (13/16)      | -                 |
| Pair A (Medical)   | medpsy-1.7b | RAG (C5)      | out_of_scope       | 16      | 100.0% (16/16)     | -                 |
| Pair B (Generic)   | qwen3-4b    | Baseline (C4) | ambiguous          | 17      | 70.6% (12/17)      | 70.6% (12/17)     |
| Pair B (Generic)   | qwen3-4b    | Baseline (C4) | fabricated_disease | 17      | 88.2% (15/17)      | 76.5% (13/17)     |
| Pair B (Generic)   | qwen3-4b    | Baseline (C4) | fabricated_drug    | 17      | 82.3% (14/17)      | 100.0% (17/17)    |
| Pair B (Generic)   | qwen3-4b    | Baseline (C4) | false_premise      | 17      | 82.3% (14/17)      | 82.3% (14/17)     |
| Pair B (Generic)   | qwen3-4b    | Baseline (C4) | missing_info       | 16      | 68.8% (11/16)      | 68.8% (11/16)     |
| Pair B (Generic)   | qwen3-4b    | Baseline (C4) | out_of_scope       | 16      | 93.8% (15/16)      | 75.0% (12/16)     |
| Pair B (Generic)   | qwen3-4b    | RAG (C5)      | ambiguous          | 17      | 76.5% (13/17)      | 58.8% (10/17)     |
| Pair B (Generic)   | qwen3-4b    | RAG (C5)      | fabricated_disease | 17      | 94.1% (16/17)      | 52.9% (9/17)      |
| Pair B (Generic)   | qwen3-4b    | RAG (C5)      | fabricated_drug    | 17      | 76.5% (13/17)      | 58.8% (10/17)     |
| Pair B (Generic)   | qwen3-4b    | RAG (C5)      | false_premise      | 17      | 76.5% (13/17)      | 82.3% (14/17)     |
| Pair B (Generic)   | qwen3-4b    | RAG (C5)      | missing_info       | 16      | 93.8% (15/16)      | 56.2% (9/16)      |
| Pair B (Generic)   | qwen3-4b    | RAG (C5)      | out_of_scope       | 16      | 68.8% (11/16)      | 56.2% (9/16)      |
| Pair B (Medical)   | medpsy-4b   | Baseline (C4) | ambiguous          | 17      | 94.1% (16/17)      | 5.9% (1/17)       |
| Pair B (Medical)   | medpsy-4b   | Baseline (C4) | fabricated_disease | 17      | 76.5% (13/17)      | 52.9% (9/17)      |
| Pair B (Medical)   | medpsy-4b   | Baseline (C4) | fabricated_drug    | 17      | 100.0% (17/17)     | 35.3% (6/17)      |
| Pair B (Medical)   | medpsy-4b   | Baseline (C4) | false_premise      | 17      | 94.1% (16/17)      | 23.5% (4/17)      |
| Pair B (Medical)   | medpsy-4b   | Baseline (C4) | missing_info       | 16      | 93.8% (15/16)      | 31.2% (5/16)      |
| Pair B (Medical)   | medpsy-4b   | Baseline (C4) | out_of_scope       | 16      | 100.0% (16/16)     | 25.0% (4/16)      |
