# Task 3: Patent Drafting (Generation)
---

## Objective
Generate diverse instructions to map official abstracts to their corresponding primary independent claims. This tests the model's ability to draft legally robust, structured claims based on high-level summaries.

## Data Schema
*   **Input:** Official Abstract
*   **Target:** Claim 1 (Primary independent claim)

## Generation Workflow
1.  **Extract:** Pull `abstractText` and the specific text of `Claim 1` from the IP Australia JSON payload.
2.  **Prompt the Generator LLM (Evol-Instruct Method):**
    *   **User Prompt:** "Generate 5 diverse instructions asking a patent drafting assistant to write a first independent method/apparatus claim based on an abstract. (e.g., 'Draft claim 1...', 'Based on this abstract, write a legally robust independent claim...'). Output only a JSON list of strings."
3.  **Assemble:** Randomly select one of the generated instructions to serve as the prompt. Pair it with the `abstractText` (as the `input`) and the official `Claim 1` text (as the `output`).