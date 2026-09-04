# Task 3: Machine Reading Comprehension (Extractive QA)

> **Retired** from the active eval / SFT pipeline. Do not generate new MRC for paper splits; code and legacy holdings may remain.

---

## Objective
Generate highly specific extractive questions and answers based directly on the claims text. This tests the model's ability to accurately retrieve rigid facts from dense legal documents.

## Data Schema
*   **Instruction:** Sampled task directive (answer using claims only; no outside knowledge)
*   **Input:** `Question: …` + `Claims: …` (instance-specific question + claims text)
*   **Target:** Synthetic factual answer grounded in the claims

## Instruction pool
Before pairing patents, build a **cached pool of diverse instruction phrasings** for this task (same idea as abstract drafting / IPC reasoning):

1. Send a meta-prompt to the generator LLM asking for N professional variants that all ask the model to answer a question from claims alone (no specific question or patent facts).
2. Parse the JSON list of strings and append until the pool reaches the configured size (default 40).
3. Persist under `data/derived/instruction_generation/_pools/mrc.json`.
4. For each training example, **randomly sample one** pool string as the Alpaca `instruction` field.

Instance-specific diversity still comes from generating a **new question per patent**; that question is stored inside `input`, not as `instruction`.

## Generation Workflow
1.  **Extract:** Pull the `claimsText` from the IP Australia JSON payload.
2.  **Sample instruction:** Draw one string from the instruction pool (build/load pool first if missing).
3.  **Prompt the Generator LLM (question + answer only):**
    *   **System Prompt:** "You are a patent attorney analyzing claims for infringement."
    *   **User Prompt:** "Read the following claims. Generate one highly specific, technical question regarding a numerical limit, chemical composition, or structural dependency found *explicitly* in the text. Then, provide the exact, concise answer. Format your response as a JSON object with 'question' and 'answer' keys."
4.  **Assemble:**
    *   `instruction` = sampled pool string
    *   `input` = `Question: [LLM question]\n\nClaims:\n[claimsText]`
    *   `output` = LLM answer
