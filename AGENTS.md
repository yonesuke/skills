# Skill Engineering Standard: The `skills/` Directory

> [!IMPORTANT]
> **Skills are not just documentation.** They are executable extensions of the agent's cognitive architecture. They must be treated with the same engineering rigor as production code.

This document outlines the **Skill Engineering Standard** for this workspace. It defines how to build high-quality, reliable, and context-efficient skills that empower the agent to solve complex problems autonomously.

---

## 1. Core Philosophy

### 1.1 Cognitive Efficiency
The agent operates within a finite context window. Skills are **compression mechanisms**.
- **Bad**: Dumping 50 pages of API docs into the context.
- **Good**: A 50-line `SKILL.md` that directs the agent to deeper resources *only when needed*.
- **Goal**: Approx. 50-100 tokens for the entry point (`SKILL.md`).

### 1.2 Reliability First
An agent guessing parameters is a failure mode. An agent running a verified script is a success mode.
- **Principle**: **Runnable Truth > Training Data**.
- **Action**: Always prefer providing an executable script over a textual description of how to do something.

### 1.3 Progressive Disclosure
Do not overwhelm the agent with details it doesn't need yet.
- **Level 1 (Discovery)**: "I know this skill exists and what it does." (`SKILL.md`)
- **Level 2 (Execution)**: "I see examples of how to use it." (`examples.md`)
- **Level 3 (Deep Dive)**: "I need to understand the underlying theory." (`reference.md`)

---

## 2. Standard Directory Structure

Every skill lives in its own directory under `skills/`.

### Directory Layout
```text
skills/<skill-name>/
├── SKILL.md         # [REQUIRED] Entry point, API surface, routing logic.
├── examples.md      # [REQUIRED] Few-shot prompts and code patterns.
├── scripts/         # [REQUIRED] Executable ground-truth Python scripts.
│   ├── demo_a.py
│   └── demo_b.py
├── reference.md     # [OPTIONAL] Deep theory, extensive API tables.
└── resources/       # [OPTIONAL] Static assets (templates, images).
```

### Naming Conventions
- **Skill Name**: Use **Gerund Form** (verb + -ing).
    - `processing-pdfs` (YES) vs `pdf-tool` (NO).
    - `analyzing-data` (YES) vs `pandas-helper` (NO).
- **Files**: Stick to the standard names (`SKILL.md`, `examples.md`, `reference.md`).

---

## 3. File Responsibilities

### 3.1 `SKILL.md`: The Orchestrator
This is the **API Surface** of the skill. It must be short, punchy, and directive.

**Requirements:**
- **YAML Frontmatter**: Essential for the system to index the skill.
    ```yaml
    ---
    name: processing-pdfs
    description: Extracts text and tables from PDF documents. Use for data ingestion tasks.
    ---
    ```
- **Description**: Third-person summary of **WHAT** and **WHEN**.
- **Routing**: Links to `examples.md` and `reference.md` with brief explanations of why to click them.
- **Decision Trees**: If the skill handles multiple scenarios, provide a simple checklist or flowchart.

**Antipatterns:**
- pasting huge code blocks (Move to `examples.md`).
- Explaining basic concepts like "What is a PDF?" (Agent knows this).

### 3.2 `examples.md`: The Context Provider
This file provides the **Few-Shot** context the agent needs to write code immediately.

**Requirements:**
- **Self-Contained**: The agent should be able to copy-paste an example and run it.
- **Mirroring**: Every code block MUST correspond to a file in `scripts/`.
- **Format**:
    ```markdown
    ## 1. Basic Usage
    Description of what this does.
    
    **Source:** [scripts/basic_usage.py](scripts/basic_usage.py)
    ```python
    # ... content of scripts/basic_usage.py ...
    ```
    ```

### 3.3 `scripts/`: The Source of Truth
These are **executable artifacts**. They prove that the code in `examples.md` actually works.

**Requirements:**
- **PEP 723 Metadata**: All scripts MUST be self-executing via `uv run`.
    ```python
    # /// script
    # requires-python = ">=3.12"
    # dependencies = ["pandas", "matplotlib"]
    # ///
    ```
- **Standalone**: `if __name__ == "__main__":` block is mandatory.
- **Deterministic**: Where possible, set seeds for reproducibility.
- **Robust**: Handle errors gracefully; don't just crash.

### 3.4 `reference.md`: The Deep Dive
Use this for high-volume information that is rarely needed but critical when it *is* needed.
- Lists of 100+ API endpoints.
- Mathematical theory.
- Tables of constants.

---

## 4. The Engineering Lifecycle

Creating a skill is an iterative engineering process.

1.  **Discovery (Manual Mode)**
    - Solve the user's problem manually.
    - Identify repetitive patterns or fragile steps.
    - *Outcome*: A set of scattered scripts or successful shell commands.

2.  **Implementation (Code First)**
    - Create the `scripts/` directory.
    - Refine your manual scripts into robust, standalone Python files with PEP 723 headers.
    - **Verify**: Run them! `uv run scripts/my_script.py`.

3.  **Integration (Documentation)**
    - Create `examples.md` by wrapping your scripts in markdown.
    - Create `SKILL.md` to index the skill.
    - Create `reference.md` if you have heavy theory to offload.

4.  **Verification (The "Fresh Agent" Test)**
    - Open a new conversation.
    - Ask the agent to perform the task using *only* the new skill.
    - *Pass Criteria*: The agent writes correct code on the **first try** without hallucinations.

---

## 5. Best Practices & Quality Control

### 5.1 Writing & Tone
- **Conciseness**: Every token costs money and context. Be ruthless.
- **Third Person**: "This skill processes..." (Not "I can help you...")
- **Directives**: Use imperative verbs. "Run this," "Check that."

### 5.2 Code Quality
- **Python Version**: Target `>=3.12`.
- **No Magic Numbers**: Define constants.
- **Visuals**: If the output is data, generate a plot (saved to disk) rather than printing 1000 numbers. The agent can use `view_file` to inspect the plot.

### 5.3 Common Anti-Patterns
- **Windows Paths**: Never use backslashes `\`. Always use forward slashes `/`.
- **Assumption of Environment**: Never assume a library is installed. Use PEP 723 to declare it.
- **Deep Nesting**: Do not create generic "utils" folders. Flatten the structure inside `scripts/`.
- **Dead Links**: Verify that every `[link](path)` actually resolves.

---

## 6. Maintenance
- **Refactoring**: If a skill becomes too large (`SKILL.md` > 100 lines), split it or move content to `reference.md`.
- **Deprecation**: If a library changes (e.g., `pandas` update), you MUST update the `scripts/` first, verify them, then update `examples.md`.
