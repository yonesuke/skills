---
description: How to create a new skill following the Skill Engineering Standard
---

# Workflow: Create New Skill

Follow this workflow to create a new skill in `skills/` that complies with [AGENTS.md](../AGENTS.md).

## 1. Preparation
- [ ] Read [skills/AGENTS.md](../../AGENTS.md) to understand the philosophy (Conciseness, Reliability).
- [ ] Choose a name in **Gerund Form** (e.g., `processing-pdfs`, not `pdf-tool`).

## 2. Discovery & Setup
- [ ] Create the directory: `mkdir -p skills/<skill-name>/scripts`.
- [ ] Identify the core logic or script you want to turn into a skill.

## 3. Implementation (Code First)
- [ ] Create a script in `skills/<skill-name>/scripts/<script_name>.py`.
- [ ] Ensure it starts with PEP 723 metadata:
    ```python
    # /// script
    # requires-python = ">=3.12"
    # dependencies = ["pkg_name"]
    # ///
    ```
- [ ] Ensure it has a `if __name__ == "__main__":` block.
- [ ] **Verify Execution**: Run `uv run skills/<skill-name>/scripts/<script_name>.py` to ensure it works.

## 4. Documentation (Integration)
- [ ] Create `skills/<skill-name>/examples.md`:
    - Add a description.
    - Embed the script content in a python code block.
    - Add a link: `**Source:** [scripts/<script_name>.py](scripts/<script_name>.py)`.
- [ ] Create `skills/<skill-name>/SKILL.md`:
    - Add YAML frontmatter (name, description).
    - Add a "Contents" section linking to `examples.md`.
    - Add usage directives (Third-person).

## 5. Verification
- [ ] Check total token count of `SKILL.md` (Target: <100 tokens).
- [ ] Confirm all links in `SKILL.md` and `examples.md` are valid.
