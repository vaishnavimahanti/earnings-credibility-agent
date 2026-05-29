from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from src.schemas import AnalystBrief, DodgeCategory
from src.agents.orchestrator import analyze_call
from src.utils import data_loader


st.set_page_config(
    page_title="Earnings credibility agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Earnings call credibility agent")
st.caption(
    "Multi-agent system that flags where management hedged, contradicted prior "
    "quarters, or dodged analyst questions. Every claim is grounded with a "
    "transcript citation."
)


@st.cache_data(show_spinner=False)
def _list_available_tickers() -> list[str]:
    """For the demo, we pre-curate a small list of tickers."""
    return ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "JPM", "BAC"]


with st.sidebar:
    st.header("Input")
    source = st.radio("Source", ["Sample call", "HuggingFace dataset"])

    if source == "HuggingFace dataset":
        ticker = st.selectbox("Ticker", _list_available_tickers())
        year = st.selectbox("Year", [2024, 2023, 2022], index=0)
    else:
        ticker = None
        year = None

    st.divider()
    st.caption(
        "⚠️ This tool operates only on publicly released earnings calls. "
        "It does not analyze MNPI or private information."
    )

    run_btn = st.button("Analyze call", type="primary", use_container_width=True)


def _category_color(cat: DodgeCategory) -> str:
    return {
        DodgeCategory.DIRECT: "#27500A",
        DodgeCategory.PARTIAL: "#633806",
        DodgeCategory.REFRAMED: "#791F1F",
        DodgeCategory.DEFERRED: "#0C447C",
        DodgeCategory.NON_ANSWER: "#501313",
    }[cat]


def _latest_eval() -> dict | None:
    path = Path("eval/results/dodge_eval_latest.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _eval_provenance(total_qa_pairs: int) -> str:
    latest = _latest_eval()
    if not latest:
        return f"Based on {total_qa_pairs} Q&A pairs; eval results not generated yet."
    binary = latest.get("binary_dodge", {})
    track = latest.get("eval_track", "sample_regression")
    n = latest.get("n_examples", "?")
    f1 = binary.get("f1", "?")
    return (
        f"Based on {total_qa_pairs} Q&A pairs · latest eval: binary F1 {f1} "
        f"on {n} examples ({track}; not a production benchmark unless real_heldout)"
    )


def _write_override(brief: AnalystBrief, pair_id: str, original: DodgeCategory, override: str, note: str) -> None:
    out_path = Path("data/overrides.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": brief.ticker,
        "quarter": brief.quarter,
        "pair_id": pair_id,
        "original_label": original.value,
        "override_label": override,
        "note": note,
    }
    with out_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _render_label_chip(label: DodgeCategory) -> str:
    color = _category_color(label)
    return (
        f"<span style='background:{color}; color:white; border-radius:6px; "
        f"padding:3px 8px; font-size:12px; font-weight:600;'>"
        f"{label.value}</span>"
    )


def _render_evidence_tab(brief: AnalystBrief, total_qa_pairs: int):
    st.subheader("Verdict")
    st.markdown(f"**{brief.headline}**")
    cols = st.columns(4)
    cols[0].metric("Score", f"{brief.credibility.overall_score}/100")
    cols[1].metric("Dodge rate", f"{brief.credibility.dodge_rate:.0%}")
    cols[2].metric("Hedging", f"{brief.credibility.avg_hedging_density:.1f}")
    if brief.credibility.contradiction_count:
        contradiction_label = str(brief.credibility.contradiction_count)
    else:
        contradiction_label = "0 (none found; requires prior indexed calls)"
    cols[3].metric("Contradictions", contradiction_label)
    st.caption(_eval_provenance(total_qa_pairs))

    st.divider()
    st.subheader("Q&A evidence")
    st.caption("Expand a row to see the exact quotes and reasoning behind the label.")
    qa_rows = brief.qa_labels or brief.qa_dodges
    if not qa_rows:
        st.success("No Q&A labels available in the brief output.")
        return

    category_filter = st.multiselect(
        "Filter labels",
        options=[c.value for c in DodgeCategory],
        default=[c.value for c in DodgeCategory],
    )
    for pair_id, label in qa_rows:
        if label.category.value not in category_filter:
            continue
        chip = _render_label_chip(label.category)
        with st.expander(
            f"{pair_id} · {label.category.value} · conf {label.confidence:.0%}",
            expanded=False,
        ):
            st.markdown(chip, unsafe_allow_html=True)
            st.markdown(f"**Question quote:** {label.evidence_from_question}")
            st.markdown(f"**Answer quote:** {label.evidence_from_answer}")
            st.markdown(f"**Why:** {label.reasoning}")
            with st.form(f"override-{pair_id}", clear_on_submit=True):
                new_label = st.selectbox(
                    "Override label",
                    options=[c.value for c in DodgeCategory],
                    index=[c.value for c in DodgeCategory].index(label.category.value),
                )
                note = st.text_input("Override note")
                submitted = st.form_submit_button("Log override")
                if submitted:
                    _write_override(brief, pair_id, label.category, new_label, note)
                    st.success("Override logged to data/overrides.jsonl")


def _render_brief_tab(brief: AnalystBrief):
    st.subheader(f"Analyst brief — {brief.ticker} {brief.quarter}")
    st.markdown(f"**{brief.headline}**")

    if brief.contradictions:
        st.markdown("### Contradictions with prior quarters")
        for c in brief.contradictions:
            st.warning(
                f"**{c.prior_quarter} ({c.severity}):** {c.reasoning}\n\n"
                f"_Now:_ {c.current_claim[:300]}…\n\n"
                f"_Then:_ {c.prior_claim[:300]}…"
            )
    else:
        st.info("No contradictions surfaced. This requires prior-quarter calls to be indexed; zero can mean no prior context was available.")

    if brief.key_concerns:
        st.markdown("### Key concerns")
        for s in brief.key_concerns:
            with st.expander(s.heading, expanded=True):
                st.write(s.content)
                if s.citations:
                    st.caption("Sources:")
                    for c in s.citations:
                        st.caption(f"• {c.speaker_name}: \"{c.quote[:150]}…\"")

    if brief.positive_signals:
        st.markdown("### Positive signals")
        for s in brief.positive_signals:
            with st.expander(s.heading, expanded=False):
                st.write(s.content)


def _render_eval_tab():
    latest = _latest_eval()
    if not latest:
        st.warning("No eval JSON found. Run `python -m eval.run_eval` first.")
        return

    st.subheader("Latest classifier eval")
    binary = latest.get("binary_dodge", {})
    cols = st.columns(4)
    cols[0].metric("Binary F1", binary.get("f1", "—"))
    cols[1].metric("Precision", binary.get("precision", "—"))
    cols[2].metric("Recall", binary.get("recall", "—"))
    cols[3].metric("5-way accuracy", latest.get("accuracy", "—"))
    st.caption(latest.get("eval_caveat", "Prototype eval; inspect the eval set before quoting."))

    st.markdown("### Confusion matrix")
    matrix = latest.get("confusion_matrix", {})
    if matrix:
        st.table(matrix)

    st.markdown("### Per-class metrics")
    per_class = latest.get("per_class", {})
    if per_class:
        st.table([
            {"category": k, **v}
            for k, v in per_class.items()
        ])


def _render_brief(brief: AnalystBrief, total_qa_pairs: int):
    tab_evidence, tab_brief, tab_eval = st.tabs(["Evidence", "Brief", "Eval"])
    with tab_evidence:
        _render_evidence_tab(brief, total_qa_pairs)
    with tab_brief:
        _render_brief_tab(brief)
    with tab_eval:
        _render_eval_tab()


if run_btn:
    with st.spinner("Loading call…"):
        if source == "Sample call":
            try:
                call = data_loader.load_sample()
            except FileNotFoundError:
                st.error(
                    "No sample call found. Run `python scripts/build_sample.py` first "
                    "or switch to HuggingFace source."
                )
                st.stop()
        else:
            calls = list(
                data_loader.load_from_huggingface(
                    tickers=[ticker], years=[year], limit=1
                )
            )
            if not calls:
                st.error(f"No call found for {ticker} {year}")
                st.stop()
            call = calls[0]

    st.info(f"Loaded {call.ticker} {call.quarter} {call.year} — {len(call.qa_pairs)} Q&A pairs")

    if not call.qa_pairs:
        st.error(
            "This transcript parsed to **0 Q&A pairs**, so there is nothing to "
            "score. This is a parsing failure, not a credibility signal — the "
            "analyst Q&A section wasn't detected. Inspect the raw transcript's "
            "speaker labels before trusting any output. The tool deliberately "
            "refuses to generate a brief from an empty call."
        )
        st.stop()

    with st.spinner("Running multi-agent analysis (this takes ~60–90s)…"):
        brief = analyze_call(call)

    st.success("Analysis complete.")
    _render_brief(brief, len(call.qa_pairs))

    with st.expander("Download brief as JSON"):
        st.download_button(
            "Download",
            data=brief.model_dump_json(indent=2),
            file_name=f"{brief.ticker}_{brief.quarter.replace(' ', '_')}_brief.json",
            mime="application/json",
        )
else:
    st.info("Pick a call and click **Analyze call** to start.")
