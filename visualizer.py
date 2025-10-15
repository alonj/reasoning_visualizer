"""Streamlit app to explore deep research tool traces from uploaded CSV files."""

from __future__ import annotations

import ast
import hashlib
import io
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st


DEFAULT_TRACE_PATH = Path(__file__).with_name("oai_response.csv")


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
	"""Load and prepare a trace from in-memory CSV bytes."""
	buffer = io.BytesIO(raw_bytes)
	df = pd.read_csv(buffer)
	return prepare_dataframe(df)


@st.cache_data(show_spinner=False)
def load_trace_from_path(path_str: str) -> pd.DataFrame:
	"""Load and prepare a trace from a file path."""
	df = pd.read_csv(path_str)
	return prepare_dataframe(df)


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
	role = str(record.get("role", "")).strip() or "n/a"
	st.markdown(
		f"""
		<div class="step-card">
			<div class="step-header">
				<span class="step-index">#{int(record.get('sequence', 0))}</span>
				<span class="step-type">{record.get('type', 'unknown')}</span>
				<span class="step-status {status}">{status}</span>
			</div>
			<div class="step-meta">
				<div><strong>Role:</strong> {role}</div>
				<div><strong>Action:</strong> {record.get('action_type', '') or '—'}</div>
				<div><strong>Target:</strong> {record.get('action_url', '') or record.get('action_query', '') or '—'}</div>
			</div>
			<div class="step-summary">{record.get('summary', '') or 'No summary available.'}</div>
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
		st.write(content)


def render_trace(df: pd.DataFrame, dataset_key: str, label: str) -> None:
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
		alt.Chart(filtered)
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

	st.markdown("### Trace Details")
	for _, row in filtered.iterrows():
		render_step(row)


def main() -> None:
	st.set_page_config(page_title="Deep Research Visualizer", layout="wide")
	st.markdown(
		"""
		<style>
			.step-card {
				background: linear-gradient(135deg, #10131a, #1e2533);
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
		</style>
		""",
		unsafe_allow_html=True,
	)

	st.title("Deep Research Reasoning Trace")
	st.caption("Upload one or more deep research traces to inspect them side-by-side.")

	if "trace_store" not in st.session_state:
		st.session_state["trace_store"] = {}

	trace_store: dict[str, dict[str, Any]] = st.session_state["trace_store"]

	with st.sidebar:
		uploaded_files = st.file_uploader(
			"Upload trace CSVs",
			type=["csv"],
			accept_multiple_files=True,
		)

		for uploaded in uploaded_files or []:
			content = uploaded.getvalue()
			trace_hash = hashlib.md5(content).hexdigest()
			trace_id = f"upload-{trace_hash}"
			if trace_id not in trace_store:
				trace_store[trace_id] = {
					"name": uploaded.name,
					"df": load_trace_from_bytes(content),
				}

		if DEFAULT_TRACE_PATH.exists():
			if st.button("Load sample trace", key="load-sample"):
				trace_id = f"sample-{DEFAULT_TRACE_PATH.name}"
				if trace_id not in trace_store:
					trace_store[trace_id] = {
						"name": f"Sample · {DEFAULT_TRACE_PATH.name}",
						"df": load_trace_from_path(str(DEFAULT_TRACE_PATH)),
					}
				st.rerun()

	if not trace_store:
		st.info("Upload a CSV trace to get started.")
		return

	tab_labels = [entry["name"] for entry in trace_store.values()]
	tabs = st.tabs(tab_labels)

	for (trace_id, entry), tab in zip(trace_store.items(), tabs):
		with tab:
			render_trace(entry["df"], trace_id, entry["name"])


if __name__ == "__main__":
	main()
