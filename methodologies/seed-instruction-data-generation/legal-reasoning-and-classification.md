# Task 1: Legal Reasoning & Classification
---

## Objective
Generate a legally grounded justification for the true IPC code assigned to a patent. This tests a model's ability to categorize technical text according to rigid legal definitions and perform zero-shot generalization on high-cardinality taxonomies.

## Data Schema
*   **Input:** Abstract + Claims Text
*   **Target:** IPC Code + Synthetic WIPO-grounded Justification

## Generation Workflow
1.  **Extract:** Pull `abstractText`, `claimsText`, and the `primary_ipc` code from the IP Australia JSON payload.
2.  **Lookup:** Programmatically query the loaded WIPO dictionary for the official definition of the extracted `primary_ipc`.
3.  **Prompt the Generator LLM:** 
    *   **System Prompt:** "You are an expert Australian Patent Examiner."
    *   **User Prompt:** "Here is an Abstract, the Claims, and the assigned IPC Code with its WIPO definition. Write a 2-sentence technical justification explaining why this code is correct by mapping the claims to the definition."
4.  **Assemble:** Save the LLM response as the `output` (formatted as "Classification: [Code]\nJustification: [LLM Response]"). Save the abstract and claims as the `input`.