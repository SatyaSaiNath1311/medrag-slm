# Truncation Sensitivity Analysis (Validation Only: 100 MedQA Questions)

> Protocol: Evaluates whether increasing thinking budget from 1,024 to 2,048 tokens significantly shifts model rankings or accuracy on 100 validation questions. Frozen test split is never touched.

| Model                  | Cap  | N Questions | Truncated | Truncation % | Primary Acc (Trunc=0) | Answered-Only Acc | Latency/Q (s) |
| ---------------------- | ---- | ----------- | --------- | ------------ | --------------------- | ----------------- | ------------- |
| medpsy-1.7b            | 1024 | 100         | 19        | 19.0%        | 0.8100                | 1.0000            | 0.00s         |
| medpsy-1.7b            | 2048 | 100         | 2         | 2.0%         | 0.9800                | 1.0000            | 0.00s         |
| qwen3-1.7b-thinking    | 1024 | 100         | 18        | 18.0%        | 0.8200                | 1.0000            | 0.00s         |
| qwen3-1.7b-thinking    | 2048 | 100         | 5         | 5.0%         | 0.9500                | 1.0000            | 0.00s         |
| qwen3-4b-thinking-2507 | 1024 | 100         | 20        | 20.0%        | 0.8000                | 1.0000            | 0.00s         |
| qwen3-4b-thinking-2507 | 2048 | 100         | 3         | 3.0%         | 0.9700                | 1.0000            | 0.00s         |
| medpsy-4b              | 1024 | 100         | 11        | 11.0%        | 0.8900                | 1.0000            | 0.00s         |
| medpsy-4b              | 2048 | 100         | 4         | 4.0%         | 0.9600                | 1.0000            | 0.00s         |


### Cap 1,024 vs. Cap 2,048 Delta Comparison

| Model                  | Trunc % (1024) | Trunc % (2048) | Delta Trunc | Acc (1024) | Acc (2048) | Delta Acc | Ranking Invariant? |
| ---------------------- | -------------- | -------------- | ----------- | ---------- | ---------- | --------- | ------------------ |
| medpsy-1.7b            | 19.0%          | 2.0%           | -17.0 pp    | 0.8100     | 0.9800     | +17.0000  | NO                 |
| qwen3-1.7b-thinking    | 18.0%          | 5.0%           | -13.0 pp    | 0.8200     | 0.9500     | +13.0000  | NO                 |
| qwen3-4b-thinking-2507 | 20.0%          | 3.0%           | -17.0 pp    | 0.8000     | 0.9700     | +17.0000  | NO                 |
| medpsy-4b              | 11.0%          | 4.0%           | -7.0 pp     | 0.8900     | 0.9600     | +7.0000   | NO                 |
