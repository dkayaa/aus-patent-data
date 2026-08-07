# Task 4: Machine Reading Comprehension (Extractive QA)
---

## Objective
Generate highly specific extractive questions and answers based directly on the claims text. This tests the model's ability to accurately retrieve rigid facts from dense legal documents.

## Data Schema
*   **Input:** Claims Text
*   **Target:** Synthetic Factual Answer

## Generation Workflow
1.  **Extract:** Pull the `claimsText` from the IP Australia JSON payload.
2.  **Prompt the Generator LLM:**
    *   **System Prompt:** "You are a patent attorney analyzing claims for infringement."
    *   **User Prompt:** "Read the following claims. Generate one highly specific, technical question regarding a numerical limit, chemical composition, or structural dependency found *explicitly* in the text. Then, provide the exact, concise answer. Format your response as a JSON object with 'question' and 'answer' keys."
3.  **Assemble:** Use the generated 'question' as the `instruction`. Use the original `claimsText` as the `input`. Use the generated 'answer' as the `output`.