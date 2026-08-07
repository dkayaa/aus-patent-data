# Task 2: Abstract Drafting (Summarization)
---

## Objective
Generate diverse instructions to map "Gold Standard" claims to their official abstracts. This evaluates a model's capacity for constraint-preserving compression, ensuring it can synthesize dense technical boundaries without omitting critical parameters or hallucinating.

## Data Schema
*   **Input:** Full Claims Text (or primary independent claims)
*   **Target:** Official Abstract

## Generation Workflow
1.  **Extract:** Pull `claimsText` and `abstractText` from the IP Australia JSON payload.
2.  **Prompt the Generator LLM (Evol-Instruct Method):**
    *   **User Prompt:** "I am building an instruction-tuning dataset. Generate 5 diverse, professional instructions asking a patent attorney to summarize a set of claims into an abstract. Vary the length and tone (e.g., 'Draft an abstract...', 'Distill the following claims...', 'Provide a technical summary...'). Output only a JSON list of strings."
3.  **Assemble:** Randomly select one of the generated instructions to serve as the prompt. Pair it with the `claimsText` (as the `input`) and the official `abstractText` (as the `output`).