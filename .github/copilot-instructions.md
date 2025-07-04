# Repository Copilot Instructions

## Purpose
This repository hosts proof-of-concept (PoC) agents built with LangChain / AutoGen on Windows + VS Code. All code should prioritize **clarity over cleverness**.

## Coding Guidelines
1. **Simplicity first** – choose the most straightforward, idiomatic solution; avoid premature abstractions.
2. **Human-readable** – use descriptive names, type hints, and docstrings; include inline comments where they clarify intent.
3. **Logging** – employ the built-in `logging` library; default level = INFO, add DEBUG statements only when deeper diagnostics help.
4. **Unit testing** – write pytest tests for every non-trivial module; mock external APIs; aim for ≥ 80 % coverage.
5. **PoC mind-set** – optimise for rapid experimentation:
   * Prefer small scripts over deep package hierarchies.
   * Keep experimental code in `examples/` or `notebooks/`.
6. **Dependencies** – maintain a minimal `requirements.txt`; pin major versions only when necessary.
7. **Platform** – assume Windows PowerShell; provide commands accordingly.
8. **Prompt engineering** – keep LLM prompts explicit and concise; comment any complex reasoning steps.
9. **Copilot usage** – for new files, Copilot should scaffold docstrings, logging setup, and a basic pytest skeleton automatically.

# IMPORTANT
When you changed the code, you MUST run all the available tests using the insturction like below:
```powershell
cd c:\src\pre-workday-poc && pytest . -v --tb=short
```


## When in doubt…
Ask for clarification in Copilot Chat or reference these instructions.
