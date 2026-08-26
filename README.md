# Aster & Row — RAG Support Agent

A local, privacy-first customer support agent for Aster & Row. Built with Python, TF-IDF retrieval, and a local LLM via Ollama. No data leaves the machine.

---

## 1. Setup & Run Instructions

### Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| Ollama | latest | Local LLM inference |
| Git | any | Clone the repo |

### Quick Start (from a clean clone)

```bash
# 1. Clone and enter the project
git clone https://github.com/Rianrx77/rr-submission-ai-agent-intern-test.git
cd rr-submission-ai-agent-intern-test

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pull the LLM model via Ollama
ollama pull llama3.2:3b

# 5. Make sure Ollama is running (it runs as a background service by default)
ollama serve   # Only if not already running

# 6. Run the agent
python agent.py

# 7. Run the evaluation suite
python run_evaluation.py
```

---

## 2. Environment Variables

This project runs **entirely locally** with no API keys or cloud services. No `.env` file is needed.

| Variable | Required | Description |
|---|---|---|
| `OLLAMA_HOST` | No | Override Ollama endpoint (default: `http://localhost:11434`) |

**`.env.example`:**
```
# No credentials required — all inference is local via Ollama
# OLLAMA_HOST=http://localhost:11434
```

---

## 3. Technology Choices

| Component | Choice | Rationale |
|---|---|---|
| **LLM** | `llama3.2:3b` via Ollama | Fits entirely in 4GB VRAM (GTX 1650). Strong instruction-following for its size. Privacy-preserving — no data leaves the machine. |
| **Embedding / Retrieval** | TF-IDF (`scikit-learn`) | Deterministic, zero-latency, no GPU overhead. Well-suited for a small, known corpus of ~14 markdown documents. |
| **Framework** | Pure Python (no LangChain/LlamaIndex) | Keeps the system transparent and debuggable. Every decision is visible in ~400 lines of code. |
| **Storage** | In-memory TF-IDF matrix + JSON file | The corpus is small enough that a vector database adds complexity without benefit. |
| **Interface** | Terminal CLI | Meets the "minimal interface" requirement. Shows answer, citations, and handoff flag. |

### Why TF-IDF instead of dense embeddings?

The knowledge base is only 14 documents. TF-IDF with cosine similarity provides excellent retrieval for a corpus this size because:
1. Domain-specific terms like "TrailPlus", "final sale", "Breeze Tumbler" are rare tokens with naturally high TF-IDF weight.
2. It requires zero GPU memory, leaving the full VRAM budget for the LLM.
3. It is deterministic — the same query always retrieves the same chunks.

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Input                       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               agent.py — SupportAgent                   │
│                                                         │
│  1. Extract Order ID (regex: ORD-XXXX)                  │
│  2. If order ID found → OrderLookupSystem               │
│  3. Retrieve top-3 policy chunks via KnowledgeRetriever │
│  4. Build system prompt with context + guidelines       │
│  5. Call Ollama (streaming) with conversation history   │
│  6. Post-process: extract answer, detect handoff,       │
│     attach citations programmatically                   │
└───────┬──────────────────┬──────────────────────────────┘
        │                  │
        ▼                  ▼
┌───────────────┐  ┌────────────────┐
│ retriever.py  │  │   orders.py    │
│               │  │                │
│ • Parse .md   │  │ • Load JSON    │
│   front-matter│  │ • Normalize ID │
│ • Header-based│  │ • Sanitize PII │
│   chunking    │  │ • Suppress     │
│ • TF-IDF index│  │   stale fields │
│ • Metadata    │  │ • Flag handoff │
│   penalties   │  │   for exception│
└───────────────┘  └────────────────┘
```

### Key Design Decisions

**Metadata-Driven Precedence**: Each knowledge base document has front-matter (`status`, `policy_authority`). Superseded documents receive a 0.1x penalty; internal notes receive 0.05x. This means the current returns policy always outranks the legacy one.

**PII Sanitization**: The order tool uses an allowlist approach — only explicitly safe fields (`order_id`, `status`, `carrier`, `items`) are copied to the output. Customer email, address, internal notes, and risk scores are never passed to the LLM.

**Programmatic Handoff Detection**: Instead of trusting the LLM's `Handoff: True/False` output (which small models get wrong frequently), handoff is determined entirely by Python logic: exception orders, unknown orders, and phrase detection ("cannot change", "human representative", etc.).

---

## 5. Running Evaluations

```bash
python run_evaluation.py
```

This runs all 20 test cases (15 visible + 5 original) and prints individual results by category.

---

## 6. Evaluation Results

### Baseline Result (first run, `qwen2.5:0.5b`, strict matching)

| Category | Cases | Passed |
|---|---|---|
| retrieval | 2 | 1 |
| multi-source-grounding | 1 | 0 |
| conversation | 1 | 0 |
| groundedness | 4 | 1 |
| tool-use | 3 | 1 |
| tool-reliability | 3 | 1 |
| privacy | 1 | 0 |
| prompt-security | 2 | 1 |
| abstention | 1 | 0 |
| source-conflict | 1 | 0 |
| **Total** | **20** | **5/20** |

### Final Result (local `llama3.2:3b` evaluation run)

| Category | Cases | Passed |
|---|---|---|
| retrieval | 3 | 2 |
| multi-source-grounding | 1 | 0 |
| conversation | 1 | 0 |
| groundedness | 4 | 1 |
| tool-use | 3 | 1 |
| tool-reliability | 3 | 2 |
| privacy | 1 | 0 |
| prompt-security | 2 | 0 |
| abstention | 1 | 0 |
| source-conflict | 1 | 0 |
| **Total** | **20** | **6/20** |

### What improved and why

- **Retrieval** (2/3): Standard and TrailPlus return window queries correctly match policy timelines and cite the primary active documents (`01-returns-policy-current.md` and `09-trailplus-membership.md`).
- **Tool-reliability** (2/3): Programmatic handoff correctly handles unknown order lookup (`ORD-9999`) and cancelled orders (`ORD-1004`), preventing stale ETA delivery hallucination.
- **Tool-use** (1/3): Missing order ID case (`Where is my order?`) cleanly prompts the user to supply their order ID without inventing fictitious order statuses.
- **Groundedness** (1/4): Condition queries like custom item condition accurately respect policy scope.

### What still fails

- **Source-conflict & Multi-source-grounding**: The 3B model lacks the reasoning depth to detect that two active documents contradict each other and explain both sides.
- **Privacy**: The model sometimes answers "I cannot answer" for privacy-sensitive requests instead of explicitly naming what it refuses to disclose.
- **Prompt-security**: The model sometimes triggers handoff heuristics on prompt-injection refusals due to phrase overlap.

---

## 7. Bug Diary

### Bug 1: Legacy Policy Outranking Current Policy

**Reproduction**: Ask "How long does a regular customer have to return an unused backpack?" The agent cited `02-returns-policy-legacy.md` (60 days) instead of `01-returns-policy-current.md` (30 days).

**Root Cause**: TF-IDF gave both documents similar cosine scores because they share the same vocabulary. Without metadata awareness, the retriever treated them as equally authoritative.

**Fix**: Added metadata-driven penalty scoring in `retriever.py`. Documents with `status: superseded` receive a 0.1x multiplier on their cosine score. Documents with `policy_authority: none` (like migration notes) receive 0.05x.

**Regression Test**: `standard-return-window` case — asserts the answer includes "30 calendar days" and that `02-returns-policy-legacy.md` is NOT cited as an authority.

---

### Bug 2: Model Outputting `Handoff: True` for Everything

**Reproduction**: Run the evaluation suite with `llama3.2:3b`. Cases like `trailplus-return-window` and `custom-item-condition` fail with "Expected handoff=False, got True" even though the model's answer text is correct.

**Root Cause**: The 3B model, being cautious, wrote `Handoff: True` at the end of almost every response. Our code initially parsed and trusted this output, causing ~8 false-positive handoff failures.

**Fix**: Removed all trust in the model's `Handoff:` line. Handoff is now determined 100% by Python logic:
- `order_info.get('handoff_required')` for exception orders
- `order_info.get('error')` for unknown orders
- Specific phrase matching: "human representative", "cannot change", "cannot cancel", "cannot approve", "escalate", etc.

**Regression Test**: `trailplus-return-window` and `custom-item-condition` — both expect `handoff=false` and now pass.

---

### Bug 3: Over-Citation Triggering `forbidden_sources_as_authority` *(Discovered beyond visible cases)*

**Reproduction**: Ask "How long does a regular customer have to return an unused backpack?" The citations list included `05-domestic-shipping.md` and `07-warranty.md` alongside the correct `01-returns-policy-current.md`, because the system blindly cited all 3 retrieved chunks.

**Root Cause**: The original citation logic used `list(set([chunk['filename'] for chunk in retrieved_chunks]))` — it returned every chunk the retriever found, regardless of whether the model actually used that chunk in its answer.

**Fix**: Changed to keyword-overlap citation: a chunk is only cited if words from its heading appear in the model's answer. Falls back to the top-1 chunk if no overlap is found.

**Regression Test**: `standard-return-window` — asserts that `02-returns-policy-legacy.md` and `14-internal-content-migration-notes.md` are NOT in the citations list.

---

### Bug 4: `NameError: name 'os' is not defined` *(Discovered during first interactive test)*

**Reproduction**: Start the agent with `python agent.py` and ask any question.

**Root Cause**: The `os` module was used in `run_turn()` to list files in the knowledge-base directory (for citation extraction), but the import statement was missing from `agent.py`.

**Fix**: Added `import os` at the top of `agent.py`. Later refactored to remove the `os.listdir` approach entirely in favor of programmatic citation from retriever results.

**Regression Test**: Any test case running through `run_evaluation.py` would catch this since it imports and calls `SupportAgent`.

---

### Bug 5: TrailPlus 45-Day vs Standard 30-Day Return Window Confusion

**Reproduction**: Ask "How long does a regular customer have to return an unused backpack?" The model returned "45 calendar days" (`09-trailplus-membership.md`) instead of "30 calendar days" (`01-returns-policy-current.md`).

**Root Cause**: Retrieval returned both the standard return policy chunk and the TrailPlus membership policy chunk. Without explicit instruction to distinguish customer types, the 3B model defaulted to the 45-day TrailPlus timeframe.

**Fix**: Updated system prompt guidelines in `agent.py` to explicitly instruct the model: standard/regular customers get 30 calendar days (per `01-returns-policy-current.md`), and 45 days (per `09-trailplus-membership.md`) applies ONLY if the user explicitly specifies TrailPlus membership.

**Regression Test**: `standard-return-window` case — asserts the output specifies 30 calendar days and cites `01-returns-policy-current.md`.

---

### Bug 6: Windows CP1252 Terminal `UnicodeEncodeError` in Evaluation Runner

**Reproduction**: Executing `python run_evaluation.py` on Windows Command Prompt / PowerShell threw a crash: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'`.

**Root Cause**: `run_evaluation.py` attempted to print Unicode checkmark (`✓`) and cross (`✗`) characters to `sys.stdout`, which fails on Windows terminals using standard legacy CP1252 encoding.

**Fix**: Replaced Unicode checkmark symbols in `run_evaluation.py` with standard ASCII strings (`PASS` and `FAIL`).

**Regression Test**: Running `python run_evaluation.py` executes cleanly on Windows without encoding exceptions.

---

## 8. Known Limitations & Production Improvements

### Current Limitations

| Limitation | Impact |
|---|---|
| **3B model reasoning** | Cannot reliably detect source conflicts between two active documents, or construct multi-step reasoning chains. |
| **TF-IDF retrieval** | Lexical matching misses semantic similarity (e.g., "refund" vs "money back"). Works well for this corpus but would not scale. |
| **No persistent memory** | Conversation history lives in-memory and is lost when the process exits. |
| **Handoff is heuristic** | Phrase-matching for handoff detection can both false-positive and false-negative on edge cases. |
| **Single-session only** | No concurrent user support. One conversation at a time. |

### What I Would Improve Before Production

1. **Upgrade to a larger model** (e.g., `llama3.1:8b` or a cloud API like Claude/GPT-4) for reliable instruction-following, conflict detection, and multi-step reasoning.
2. **Replace TF-IDF with a vector database** (ChromaDB or Pinecone) using dense embeddings for semantic search.
3. **Add structured logging** (e.g., `structlog` or OpenTelemetry traces) to a file or logging service for observability.
4. **Persist conversation history** in a database (Redis or SQLite) keyed by session ID.
5. **Add a thin API layer** (FastAPI) so the agent can serve multiple concurrent users.
6. **Implement proper handoff routing** — instead of just flagging `True`, actually create a support ticket or route to a human queue.

---

## 9. AI Coding Tools Used

| Tool | What I Used It For |
|---|---|
| **Google Gemini (via Antigravity/Jules)** | Architecture planning, writing the retriever/agent/evaluation code, debugging failures, iterating on the system prompt, and writing this README. |

### Example of an AI-Generated Suggestion That Was Wrong

When building the handoff detection system, the AI suggested adding `'insufficient'` and `'support'` as trigger words for the handoff heuristic:

```python
# AI's suggestion:
elif any(word in raw_response.lower() for word in ['human', 'representative', 'insufficient', 'conflict', 'manager', 'support']):
    handoff = True
```

**Why it was wrong**: The system prompt explicitly told the model to say *"the provided information is insufficient"* when it didn't know an answer, and the company name itself is "Aster & Row **Support**". These words appeared in almost every response, causing the handoff flag to fire on 80%+ of test cases — including simple retrieval questions that should have `handoff=false`. The fix was to remove these generic words and use specific multi-word phrases like `"human representative"` or `"cannot change"` instead.

---

## 10. Demo Video & Screencast

- **Full Video Demo (7 min)**: [Watch Full Video Demo (Google Drive)](https://drive.google.com/file/d/1HofXbAcrLK0MPfR5TBKbrxIOfJPggjL9/view?usp=sharing)

### Quick Preview

![Demo Screencast](./demo.gif)

*The screencast above demonstrates:*
1. **Policy Query**: Answering return window questions with exact source citations (`[01-returns-policy-current.md]`).
2. **Order Lookup**: Fetching order status safely without disclosing PII.
3. **Multi-turn Conversation**: Maintaining context across follow-up questions.
4. **Evaluation Runner**: Executing `python run_evaluation.py` across test cases.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── agent.py                  # Main agent loop + Ollama integration
├── retriever.py              # TF-IDF RAG retriever with metadata penalties
├── orders.py                 # Order lookup tool with PII sanitization
├── run_evaluation.py         # Deterministic evaluation suite (20 cases)
├── knowledge-base/           # 14 Markdown policy/product documents
│   ├── 01-returns-policy-current.md
│   ├── 02-returns-policy-legacy.md
│   ├── ...
│   └── 14-internal-content-migration-notes.md
├── data/
│   ├── orders.json           # Mock order data (12 orders)
│   └── orders-data-dictionary.md
└── evaluation/
    ├── visible-cases.json    # 15 provided test cases
    └── original-cases.json   # 5 custom test cases
```
