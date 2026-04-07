from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from matcher import InternshipMatcher, greedy_allocate_with_capacity
from parser import parse_resume
from utils import (
    AllocationConfig,
    build_internship_text,
    build_resume_text,
    df_to_csv_bytes,
    guess_title_columns,
    highlight_terms,
    make_label,
)


DEFAULT_INTERNSHIPS_CSV = os.path.join("data", "internship.csv")


def _load_internships_dataset(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin1")


def _prepare_internships(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, List[str]]:
    dfx, text_col = build_internship_text(df)
    label_cols = guess_title_columns(dfx)
    return dfx, text_col, label_cols


def _get_capacity_series(df: pd.DataFrame) -> Optional[pd.Series]:
    cols_lower = {c.lower(): c for c in df.columns}
    for k in ["capacity", "openings", "slots"]:
        if k in cols_lower:
            c = cols_lower[k]
            s = pd.to_numeric(df[c], errors="coerce").fillna(1).astype(int)
            return s.clip(lower=0)
    return None


def _compute_score_matrix(matcher: InternshipMatcher, resume_texts: List[str], internship_texts: List[str]) -> np.ndarray:
    resume_mat = matcher.vectorizer.transform(resume_texts)
    internship_mat = matcher._internship_matrix  # already computed in fit()
    return cosine_similarity(resume_mat, internship_mat)


def main() -> None:
    st.set_page_config(page_title="Resume Parser + NLP Internship Allocator", page_icon="Resume", layout="wide")

    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
          mark { background-color: #fff2ac; padding: 0.05em 0.18em; border-radius: 0.25em; }
          .small-note { color: rgba(49, 51, 63, 0.7); font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Resume Parser + NLP Internship Allocator")
    st.caption(
        "Upload PDF resumes. Internships are loaded from `data/internship.csv` (your provided dataset). "
        "Extract skills, compute NLP match scores, and allocate best internships."
    )

    with st.sidebar:
        st.header("Inputs")
        if not os.path.exists(DEFAULT_INTERNSHIPS_CSV):
            st.error(f"Dataset not found at `{DEFAULT_INTERNSHIPS_CSV}`. Please ensure the file exists.")
            st.stop()
        st.text_input("Internships dataset path", value=DEFAULT_INTERNSHIPS_CSV, disabled=True)
        resumes = st.file_uploader("Upload resumes (PDF)", type=["pdf"], accept_multiple_files=True)

        st.divider()
        st.subheader("Matching settings")
        top_k = st.slider("Show top matches per resume", min_value=1, max_value=5, value=3, step=1)
        enforce_unique = st.toggle("Allocate unique internships (ignore capacity)", value=False)
        run_btn = st.button("Run matching", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Upload inputs in the sidebar, then click **Run matching**.")
        st.stop()

    if not resumes:
        st.error("Please upload at least one resume PDF.")
        st.stop()

    # Load internships dataset (provided)
    internships_df = _load_internships_dataset(DEFAULT_INTERNSHIPS_CSV)
    if internships_df.empty:
        st.error("Internships CSV is empty or unreadable. Please verify the file.")
        st.stop()

    internships_df, text_col, label_cols = _prepare_internships(internships_df)
    if text_col not in internships_df.columns:
        st.error("Could not build a text column from the internship CSV. Ensure it has at least one text-like column.")
        st.stop()

    if internships_df[text_col].fillna("").astype(str).str.len().sum() == 0:
        st.error("Internship descriptions appear empty after preprocessing. Please verify your CSV contents.")
        st.stop()

    # Parse resumes
    parsed = []
    for f in resumes:
        filename, cleaned_text, skills = parse_resume(f)
        parsed.append({"resume_file": filename, "resume_text": cleaned_text, "resume_skills": skills})

    resumes_df = pd.DataFrame(parsed)
    if resumes_df.empty:
        st.error("No resumes could be parsed.")
        st.stop()

    # Build matcher inputs
    resume_texts = [
        build_resume_text(row["resume_text"], row["resume_skills"]) for _, row in resumes_df.iterrows()
    ]
    internship_texts = internships_df[text_col].fillna("").astype(str).tolist()

    matcher = InternshipMatcher()
    matcher.fit(internship_texts)

    # Compute scores matrix for allocation + viz
    score_matrix = _compute_score_matrix(matcher, resume_texts, internship_texts)

    # Allocation (capacity-aware if available)
    capacity_s = _get_capacity_series(internships_df)
    config = AllocationConfig(enforce_unique_internship=enforce_unique, capacity_column=capacity_s.name if capacity_s is not None else None)

    if enforce_unique:
        cap = np.ones(len(internships_df), dtype=int)
    else:
        cap = capacity_s.to_numpy(dtype=int) if capacity_s is not None else np.ones(len(internships_df), dtype=int)
    allocated_idx = greedy_allocate_with_capacity(score_matrix, capacity=cap)

    # Build results
    results_rows: List[Dict[str, Any]] = []
    for r_i, row in resumes_df.iterrows():
        top_matches = matcher.match_top_k(resume_texts[r_i], top_k=top_k)
        alloc = allocated_idx[r_i]

        if alloc is None:
            best = None
        else:
            best = next((m for m in top_matches if m.index == alloc), None)
            if best is None:
                # If greedy allocation picked an index outside top_k, create minimal result
                best = top_matches[0] if top_matches else None

        for rank, m in enumerate(top_matches, start=1):
            intern_row = internships_df.iloc[m.index]
            results_rows.append(
                {
                    "resume_file": row["resume_file"],
                    "resume_skills": ", ".join(row["resume_skills"]) if row["resume_skills"] else "",
                    "match_rank": rank,
                    "internship_index": int(m.index),
                    "internship_label": make_label(intern_row, label_cols),
                    "match_score": round(m.score, 4),
                    "explanation_keywords": ", ".join(m.keywords),
                    "allocated": bool(alloc is not None and m.index == alloc),
                }
            )

    results_df = pd.DataFrame(results_rows)

    # UI: Overview
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.subheader("Allocations")
        alloc_view = results_df[results_df["allocated"] == True].copy()
        if alloc_view.empty:
            st.warning("No allocations were made (capacity may be 0 or descriptions too sparse).")
        else:
            st.dataframe(
                alloc_view[
                    [
                        "resume_file",
                        "internship_label",
                        "match_score",
                        "explanation_keywords",
                        "resume_skills",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download allocations CSV",
                data=df_to_csv_bytes(
                    alloc_view[
                        [
                            "resume_file",
                            "internship_index",
                            "internship_label",
                            "match_score",
                            "explanation_keywords",
                            "resume_skills",
                        ]
                    ]
                ),
                file_name="resume_internship_allocations.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown('<div class="small-note">Tip: set “Show top matches” to 3–5 to get more alternatives.</div>', unsafe_allow_html=True)

    with right:
        st.subheader("Quick stats")
        st.metric("Resumes", len(resumes_df))
        st.metric("Internship rows", len(internships_df))
        if capacity_s is not None and not enforce_unique:
            st.metric("Total capacity", int(capacity_s.sum()))
        st.caption("Match score is cosine similarity on TF‑IDF vectors (higher is better).")

    st.divider()

    # UI: Per-resume details + chart
    st.subheader("Per-resume matches (top choices + explanations)")
    for r_i, row in resumes_df.iterrows():
        resume_name = row["resume_file"]
        resume_skills = row["resume_skills"]
        top_matches = results_df[results_df["resume_file"] == resume_name].sort_values("match_rank")

        allocated_row = top_matches[top_matches["allocated"] == True].head(1)
        allocated_label = allocated_row["internship_label"].iloc[0] if not allocated_row.empty else "—"

        with st.expander(f"{resume_name}  •  Allocated: {allocated_label}", expanded=(r_i == 0)):
            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                st.markdown("**Extracted skills**")
                if resume_skills:
                    st.write(", ".join(resume_skills))
                else:
                    st.warning("No skills found from the predefined list. Matching will rely more on general resume text.")

                # Bar chart of top_k scores
                chart_df = top_matches[["internship_label", "match_score", "allocated"]].copy()
                if not chart_df.empty:
                    st.markdown("**Top match scores**")
                    st.bar_chart(chart_df.set_index("internship_label")["match_score"])

            with c2:
                st.markdown("**Top matches + why**")
                if top_matches.empty:
                    st.info("No matches found.")
                else:
                    for _, mrow in top_matches.iterrows():
                        idx = int(mrow["internship_index"])
                        intern_row = internships_df.iloc[idx]
                        keywords = [k.strip() for k in str(mrow["explanation_keywords"]).split(",") if k.strip()]
                        label = str(mrow["internship_label"])

                        badge = " (allocated)" if bool(mrow["allocated"]) else ""
                        st.markdown(f"**#{int(mrow['match_rank'])}** — {label}{badge}")
                        st.write(f"Score: **{mrow['match_score']}**")
                        if keywords:
                            st.caption(f"Why: matched keywords — {', '.join(keywords[:12])}")

                        # Highlight matched keywords inside internship text (first ~800 chars)
                        preview = str(intern_row.get(text_col, ""))[:800]
                        preview_h = highlight_terms(preview, keywords)
                        if preview_h.strip():
                            st.markdown(preview_h, unsafe_allow_html=True)
                        st.divider()

    st.subheader("All matches (top-k for each resume)")
    st.dataframe(results_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------
# Page setup
# ---------------------------------
st.set_page_config(page_title="🤖 Smart Internship Allocator", layout="wide")
st.title("🤖 Smart Internship Allocator using NLP")
st.markdown(
    """
    This app automatically allocates **students to internships** based on their skills and
    the internship descriptions using **TF-IDF + Cosine Similarity**.
    """
)

# ---------------------------------
# Upload CSVs
# ---------------------------------
st.sidebar.header("📂 Upload Files")
students_file = st.sidebar.file_uploader("Upload Students CSV", type=["csv"])
internships_file = st.sidebar.file_uploader("Upload Internships CSV", type=["csv"])

# ---------------------------------
# When both files are uploaded
# ---------------------------------
if students_file and internships_file:
    students = pd.read_csv(students_file)
    internships = pd.read_csv(internships_file)

    # Validate
    if not {"Name", "Skills"}.issubset(students.columns):
        st.error("❌ Students CSV must have columns: Name, Skills")
    elif not {"Title", "Organization", "Details"}.issubset(internships.columns):
        st.error("❌ Internships CSV must have columns: Title, Organization, Details")
    else:
        st.success("✅ Files uploaded successfully!")

        if st.button("🚀 Run Smart Allocation"):
            # ---------------------------------
            # TF-IDF Matching
            # ---------------------------------
            tfidf = TfidfVectorizer(stop_words="english", max_features=300)
            internship_vectors = tfidf.fit_transform(internships["Details"].fillna(""))

            allocations = []

            for _, student in students.iterrows():
                student_text = student["Skills"]
                student_vector = tfidf.transform([student_text])
                scores = cosine_similarity(student_vector, internship_vectors).flatten()
                internships["Match_Score"] = scores
                best_match = internships.sort_values(by="Match_Score", ascending=False).iloc[0]

                allocations.append({
                    "Student": student["Name"],
                    "Allocated Internship": best_match["Title"],
                    "Organization": best_match["Organization"],
                    "Match Score": round(best_match["Match_Score"], 3)
                })

            alloc_df = pd.DataFrame(allocations)

            # ---------------------------------
            # Display + Download
            # ---------------------------------
            st.success("✅ Allocation Completed!")
            st.dataframe(alloc_df, use_container_width=True)

            csv = alloc_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Allocation Results",
                data=csv,
                file_name="allocations.csv",
                mime="text/csv",
            )

            # Optional insight
            st.markdown("### 📊 Match Score Distribution")
            st.bar_chart(alloc_df.set_index("Student")["Match Score"])
else:
    st.info("👈 Upload both Students and Internships CSVs from the sidebar to start.")
