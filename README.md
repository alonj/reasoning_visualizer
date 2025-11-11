# Deep Research Visualizer

This Streamlit app (`visualizer.py`) lets you inspect step-by-step traces produced by OpenAI reasoning models and deep research systems. Load one or more JSON traces and explore how the agent reasoned, which tools it called, and how each call went.

## Features
- Upload multiple trace JSON files and keep them in session for quick switching.
- Filter by step type, action type, status, or free-text search to focus on the parts you care about.
- Visualize trace structure with an interactive Altair scatter plot keyed by sequence number and step type.
- Inspect richly formatted step cards with summaries, tool metadata, code blocks, and parsed outputs.
- Compare any two traces side-by-side, and download the currently filtered view as a CSV.
- Load the bundled sample trace to experiment with the UI instantly.

## Requirements
- Python 3.10+
- Packages: `streamlit`, `pandas`, `altair`
- Optional (for nicer outputs): `pyarrow` for faster CSV downloads and `watchdog` for Streamlit auto-reload

Install the dependencies into your environment:

```bash
pip install streamlit pandas altair
```

## Run the App
From the project root (where `visualizer.py` lives), start Streamlit:

```bash
streamlit run visualizer.py
```

Streamlit opens a local browser tab (default: http://localhost:8501). When you edit the script, Streamlit hot-reloads automatically.

## Loading Traces
- Click **Upload trace JSONs** in the sidebar to select one or more `.json` files.
- Each file should contain a top-level `response` key whose `output` is an ordered list of step dictionaries.
- Expected per-step fields include `sequence`, `type`, `status`, `action_type`, `action_query`, `summary`, `content`, `code`, and `outputs`. Missing fields are handled gracefully and will display as empty.
- Use the **Load sample trace** button to load `monaco_0005.json` if it resides alongside `visualizer.py`.

## Working With the UI
- Use the **Filters** expander to narrow by step type (reasoning, tool calls, messages, etc.), action type, status, or keyword.
- Review aggregate metrics (total steps, reasoning steps, tool calls) and examine the scatter plot for a quick overview of the trace shape.
- Toggle **Show all steps** to switch between only the latest matching step and the entire filtered trace.
- For comparisons, enable **side-by-side comparison** and pick two traces—perfect for before/after model runs.

## Tip: Trace Format Validation
If your JSON structure differs, adapt the loader to normalize your fields: the parsing logic lives in `load_trace_from_bytes` / `load_trace_from_path` and `prepare_dataframe`. Adding columns or renaming them there keeps the rest of the app working.
