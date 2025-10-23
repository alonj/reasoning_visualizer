"""Streamlit app to explore deep research tool traces from uploaded JSON files."""

from __future__ import annotations

import ast
import hashlib
import io
from pathlib import Path
import re
from typing import Any
import json

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html


DEFAULT_TRACE_PATH = Path(__file__).with_name("monaco_0005.json")

def switch_tab(tab_index):
    js_code = f"""
    <script>
    var tabGroup = window.parent.document.getElementsByClassName("stTabs")[0];
    var tabButtons = tabGroup.getElementsByTagName("button");
    if (tabButtons.length > {tab_index}) {{
        tabButtons[{tab_index}].click();
    }}
    </script>
    """
    html(js_code, height=0, width=0)
	
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""Normalize column names and add helper columns."""
	df = df.copy()
	df.columns = [c.replace(" ", "_").lower() for c in df.columns]
	if "sequence" not in df.columns:
		df.insert(0, "sequence", df.index + 1)
	for col in ("summary", "content"):
		if col in df.columns:
			df[col] = df[col].fillna("")
		else:
			df[col] = ""
	return df


@st.cache_data(show_spinner=False)
def load_trace_from_bytes(raw_bytes: bytes) -> pd.DataFrame:
	"""Load and prepare a trace from in-memory JSON bytes."""
	buffer = io.BytesIO(raw_bytes)
	data = json.load(buffer)
	df = pd.json_normalize(data['response']['output'], meta_prefix='_', record_prefix='_', sep='_')
	return prepare_dataframe(df), data.get('prompt', 'Prompt not found.')

@st.cache_data(show_spinner=False)
def load_trace_from_path(path_str: str) -> pd.DataFrame:
	"""Load and prepare a trace from a file path."""
	with open(path_str, "rb") as f:
		data = json.load(f)
	df = pd.json_normalize(data['response']['output'], meta_prefix='_', record_prefix='_', sep='_')
	return prepare_dataframe(df), data.get('prompt', 'Prompt not found.')

def parse_literal(value: str | float | int | None):
	if value is None or (isinstance(value, float) and pd.isna(value)):
		return None
	if isinstance(value, (int, float)):
		return value
	if isinstance(value, str) and not value.strip():
		return None
	try:
		return ast.literal_eval(value)
	except (SyntaxError, ValueError):
		return value


def render_step(row) -> None:
	record = row.fillna("")
	status = str(record.get("status", "")).strip() or "unknown"
	role = str(record.get("role", "")).strip() or ""
	summary = record.get("summary", "")
	raw_type = str(record.get("type", "unknown")).strip().lower()
	safe_type = re.sub(r'[^a-z0-9_-]+', '-', raw_type)
	action = record.get("action_type", "")
	query = record.get("action_query", "")
	target = record.get("action_url", "") or query
	target = "<a href='" + target + "' target='_blank'>" + target + "</a>" if target.startswith("http") else target
	google_search_link = f"https://www.google.com/search?q={query.replace(' ', '+')}" if query else ""
	bing_search_link = f"https://www.bing.com/search?form=&q={query.replace(' ', '+')}" if query else ""
	search_link = f"🔍 <a href='{google_search_link}' target='_blank'>Google</a> | <a href='{bing_search_link}' target='_blank'>Bing</a>" if query else ""
	pattern = record.get("action_pattern", "")
	if isinstance(summary, str):
		summary = [{"text": summary}]
	summary = "\n".join(line['text'].strip() for line in summary)
	st.markdown(
		f"""
		<div class="step-card">
			<div class="step-header">
				<span class="step-index">#{int(record.get('sequence', 0))}</span>
				<span class="step-type {safe_type}" data-type="{raw_type}">{record.get('type', 'unknown')}</span>
				<div>{'<strong>Action: </strong>' + action if action else ''} {('<strong> ⟶ </strong> ' + (f'"{pattern}" | ' if pattern else '') + target) if target else ''}
				{"(No summary available)" if summary == "" else ""} {search_link}</div>
				<span class="step-status {status}">{status}</span>
			</div>
			{f'<div class="step-meta"><div><strong>Role:</strong>{role}</div></div>' if role else '<div class="step-meta"></div>'}
			<div class="step-summary">{summary}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)
	code_snippet = record.get("code")
	if code_snippet:
		st.code(code_snippet, language="python")
	outputs = parse_literal(record.get("outputs"))
	if outputs:
		st.markdown("**Outputs**")
		st.json(outputs)
	content = record.get("content")
	if content:
		st.markdown("**Message Content**")
		if not isinstance(content, list):
			content = [content]
		for c in content:
			if 'text' in c:
				st.markdown(c['text'])
			c_no_text = {k: v for k, v in c.items() if k != 'text'}
			st.write(c_no_text)


def render_trace(df: pd.DataFrame, dataset_key: str, label: str, prompt: str) -> None:
	close_col, _ = st.columns([1, 5])
	with close_col:
		if st.button("Close tab", key=f"{dataset_key}-close"):
			st.session_state.get("trace_store", {}).pop(dataset_key, None)
			st.rerun()

	with st.expander("Filters", expanded=True):
		type_options = sorted([opt for opt in df.get("type", pd.Series(dtype=str)).dropna().unique() if opt])
		selected_types = st.multiselect(
			"Step type",
			type_options,
			default=type_options,
			key=f"{dataset_key}-type",
		)

		action_options = sorted([opt for opt in df.get("action_type", pd.Series(dtype=str)).dropna().unique() if opt])
		selected_actions = st.multiselect(
			"Action type",
			action_options,
			default=action_options,
			key=f"{dataset_key}-action",
		)

		status_options = sorted(df.get("status", pd.Series(dtype=str)).fillna("unknown").unique())
		selected_status = st.multiselect(
			"Status",
			status_options,
			default=status_options,
			key=f"{dataset_key}-status",
		)

		search_term = st.text_input("Search text", key=f"{dataset_key}-search")

	filtered = df.copy()
	if selected_types:
		filtered = filtered[filtered["type"].isin(selected_types)]
	if selected_actions:
		filtered = filtered[
			(filtered["action_type"].isin(selected_actions))
			| filtered["action_type"].isna()
		]
	if selected_status:
		filtered = filtered[filtered["status"].fillna("unknown").isin(selected_status)]
	if search_term:
		lowered = search_term.lower()
		mask = filtered.apply(
			lambda row: lowered in " ".join(row.astype(str)).lower(),
			axis=1,
		)
		filtered = filtered[mask]

	st.download_button(
		"Download filtered CSV",
		filtered.to_csv(index=False),
		file_name=f"{label.replace(' ', '_').lower()}_filtered.csv",
		key=f"{dataset_key}-download",
	)

	col_a, col_b, col_c = st.columns(3)
	col_a.metric("Total steps", len(df))
	col_b.metric("Reasoning steps", int((df["type"] == "reasoning").sum()))
	col_c.metric("Tool calls", int((df["type"] != "reasoning").sum()))

	if filtered.empty:
		st.info("No steps match the current filters.")
		return

	chart = (
		alt.Chart(filtered.drop(columns=["summary"]))
		.mark_circle(size=200)
		.encode(
			x=alt.X("sequence:Q", title="Sequence"),
			y=alt.Y("type:N", title="Step type"),
			color=alt.Color("type:N", legend=None),
			tooltip=[
				"sequence",
				"type",
				"status",
				"action_type",
				"action_query",
				"action_url",
			],
		)
		.properties(height=320)
		.interactive()
	)
	st.altair_chart(chart, use_container_width=True)

	st.markdown("### Prompt")
	st.markdown(f"#### {prompt}")

	st.markdown("### Trace Details")
	for _, row in filtered.iterrows():
		render_step(row)


def main() -> None:
	st.set_page_config(page_title="Deep Research Visualizer", layout="wide")
	st.markdown(
		"""
		<style>
			.step-card {
				background: linear-gradient(135deg, #273143, #3a4963);
				border-radius: 16px;
				padding: 1.2rem;
				box-shadow: 0 12px 30px rgba(9, 12, 20, 0.35);
				margin-bottom: 1rem;
				border: 1px solid rgba(255, 255, 255, 0.06);
				color: #f3f6ff;
			}
			.step-header {
				display: flex;
				gap: 0.75rem;
				align-items: center;
				margin-bottom: 0.5rem;
			}
			.step-index {
				font-size: 1.25rem;
				font-weight: 700;
				color: #8ab4ff;
			}
			.step-type {
				padding: 0.2rem 0.6rem;
				border-radius: 999px;
				background: rgba(138, 180, 255, 0.18);
				font-size: 0.85rem;
				text-transform: uppercase;
				letter-spacing: 0.04em;
			}
			.step-type{ /* default style */ }
			.step-type[data-type="reasoning"] { background: rgba(92, 225, 166, 0.18); }
            .step-type[data-type="message"] { background: rgba(241, 139, 139, 0.18); }
			.step-type[data-type="web_search_call"] { background: rgba(138, 180, 255, 0.18); }
			.step-type[data-type="code_interpreter_call"] { background: rgba(255, 203, 79, 0.18); }

			.step-status {
				margin-left: auto;
				font-size: 0.8rem;
				text-transform: uppercase;
				letter-spacing: 0.08em;
			}
			.step-status.completed { color: #5ce1a6; }
			.step-status.failed { color: #f18b8b; }
			.step-status.unknown { color: #c2c8d5; }
			.step-meta {
				display: flex;
				gap: 1.5rem;
				flex-wrap: wrap;
				font-size: 0.9rem;
				opacity: 0.85;
				margin-bottom: 0.75rem;
			}
			.step-summary {
				font-size: 1rem;
				line-height: 1.55;
			}
			.stTabs [data-baseweb="tab-list"] {
				gap: 6px;
			}
			.stTabs [data-baseweb="tab"] {
				height: 50px;
				white-space: pre-wrap;
				background-color: #FFFFFF;
				border-radius: 4px 4px 0px 0px;
				gap: 1px;
				padding-top: 10px;
				padding-bottom: 10px;
				font-weight: 600;
			}
			.stTabs [aria-selected="true"] {
				background-color: #F0F2F6;
			}
			a:link {
				color: #b1bdd1;
			}

		</style>
		""",
		unsafe_allow_html=True,
	)
	st.session_state["latest_upload_tab_idx"] = st.session_state.get("latest_upload_tab_idx", 0)
	st.title("Deep Research Reasoning Trace")
	st.caption("Upload one or more deep research traces to inspect them side-by-side.")

	if "trace_store" not in st.session_state:
		st.session_state["trace_store"] = {}

	trace_store: dict[str, dict[str, Any]] = st.session_state["trace_store"]
	with st.sidebar:
		uploaded_files = st.file_uploader(
			"Upload trace JSONs",
			type=["json"],
			accept_multiple_files=True,
		)

		for uploaded in uploaded_files or []:
			content = uploaded.getvalue()
			trace_hash = hashlib.md5(content).hexdigest()
			trace_id = f"upload-{trace_hash}"
			if trace_id not in trace_store:
				df, prompt = load_trace_from_bytes(content)
				trace_store[trace_id] = {
					"name": uploaded.name,
					"df": df,
					"prompt": prompt,
				}
		latest_upload_tab_idx = len(trace_store) - 1 - len(uploaded_files) if uploaded_files else st.session_state.get("latest_upload_tab_idx", 0)
		st.session_state["latest_upload_tab_idx"] = latest_upload_tab_idx
		if DEFAULT_TRACE_PATH.exists():
			if st.button("Load sample trace", key="load-sample"):
				trace_id = f"sample-{DEFAULT_TRACE_PATH.name}"
				df, prompt = load_trace_from_path(str(DEFAULT_TRACE_PATH))
				if trace_id not in trace_store:
					trace_store[trace_id] = {
						"name": f"Sample · {DEFAULT_TRACE_PATH.name}",
						"df": df,
						"prompt": prompt,
					}
				st.rerun()

	if not trace_store:
		st.info("Upload a JSON trace to get started.")
		return

	tab_labels = [entry["name"] for entry in trace_store.values()]
	tabs = st.tabs(tab_labels)

	for (trace_id, entry), tab in zip(trace_store.items(), tabs):
		with tab:
			render_trace(entry["df"], trace_id, entry["name"], entry["prompt"])
	# when new files are uploaded, set active tab to latest upload
	if uploaded_files:
		st.session_state.active_tab = st.session_state["latest_upload_tab_idx"]
		switch_tab(st.session_state.active_tab)


if __name__ == "__main__":
	main()
