# Chatbot token-cap decision rule (pre-registered before any code change)

**Date:** 2026-11-01

Source data: outputs/analysis/budget_pair_b.md, section 3, medpsy-4b rows only (MedQA validation, first 100 questions, C1, greedy). No test data is used for this decision.

Rule: the chatbot generation cap for medpsy-4b is the smallest cap in [1024, 1536, 2048, 3072, 4096] whose validation strict accuracy is within 1.0 percentage point of the best strict accuracy across all caps.

Values: 1024 = 88%, 1536 = 92%, 2048 = 93%, 3072 = 94%, 4096 = 94%. Best = 94%. Smallest cap within 1 point = 2048.

Decision: chatbot cap = 2048.

Behaviour kept: if generation reaches the cap, the chatbot abstains and never shows a partial answer. Only the displayed reason text is reworded.

Limitations: tuned on 100 MedQA multiple-choice validation questions; free-text health-worker questions were not evaluated; worst-case wait time at the cap is to be measured in the live test.
