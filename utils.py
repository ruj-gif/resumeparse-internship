from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


_WHITESPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^a-z0-9\+\#\.\s]")


def clean_text(text: str) -> str:
    """
    Normalize raw text for matching.
    Keeps letters/numbers and a few skill-relevant symbols (+ # .).
    """
    if not text:
        return ""
    t = text.lower()
    t = t.replace("\x00", " ")
    t = _NON_TEXT_RE.sub(" ", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t


def safe_read_csv(uploaded_file) -> pd.DataFrame:
    """
    Read a Streamlit UploadedFile as a pandas DataFrame.
    """
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1")


def dataframe_text_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object"]


def build_internship_text(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Produce a single text column used for NLP matching.

    - If a likely description column exists, use it.
    - Else, concatenate all text-like columns and any 'Req_*' requirement columns.
    """
    if df.empty:
        return df.copy(), "combined_text"

    dfx = df.copy()
    cols_lower = {c.lower(): c for c in dfx.columns}
    preferred = [
        # Common AICTE / job-board style columns
        "skills",
        "skill",
        "skill(s)",
        "key_skills",
        "required_skills",
        "qualification",
        "eligibility",
        "description",
        "job_description",
        "role_description",
        "details",
        "summary",
        "about",
        "internship_description",
        "requirements",
        "responsibilities",
        "role",
        "title",
    ]
    for key in preferred:
        if key in cols_lower:
            col = cols_lower[key]
            dfx["combined_text"] = dfx[col].fillna("").astype(str).map(clean_text)
            return dfx, "combined_text"

    # Fallback: concatenate all object columns + any Req_* columns
    text_cols = dataframe_text_columns(dfx)
    req_cols = [c for c in dfx.columns if str(c).lower().startswith("req_")]
    use_cols = list(dict.fromkeys(text_cols + req_cols))  # dedupe preserving order
    if not use_cols:
        dfx["combined_text"] = ""
        return dfx, "combined_text"

    def _row_text(row) -> str:
        parts: List[str] = []
        for c in use_cols:
            v = row.get(c, "")
            if pd.isna(v):
                continue
            parts.append(str(v))
        return clean_text(" ".join(parts))

    dfx["combined_text"] = dfx.apply(_row_text, axis=1)
    return dfx, "combined_text"


def build_resume_text(full_text: str, skills: Sequence[str]) -> str:
    """
    Represent a resume for matching. Skills are repeated to upweight them vs. noisy text.
    """
    t = clean_text(full_text)
    skills_text = clean_text(" ".join(skills))
    if skills_text:
        return f"{skills_text} {skills_text} {t}".strip()
    return t


def guess_title_columns(df: pd.DataFrame) -> List[str]:
    """
    Columns to show as a human-readable internship label.
    """
    if df.empty:
        return []
    candidates = [
        # Common AICTE columns
        "internship title",
        "company name",
        "company",
        "organization",
        "location",
        "city",
        "state",
        "title",
        "role",
        "position",
        "job_title",
        "internship",
        "project",
        "company",
        "organization",
        "sector",
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    out = []
    for k in candidates:
        if k in cols_lower:
            out.append(cols_lower[k])
    return list(dict.fromkeys(out))


def make_label(row: pd.Series, label_cols: Sequence[str]) -> str:
    if not label_cols:
        return f"Internship #{int(row.name) + 1}"
    parts = []
    for c in label_cols:
        v = row.get(c, "")
        if pd.isna(v) or str(v).strip() == "":
            continue
        parts.append(str(v).strip())
    return " | ".join(parts) if parts else f"Internship #{int(row.name) + 1}"


def highlight_terms(text: str, terms: Sequence[str]) -> str:
    """
    Highlight matched terms inside text using <mark>. Safe to use with st.markdown(..., unsafe_allow_html=True).
    """
    if not text:
        return ""
    if not terms:
        return text
    # Sort longest-first to reduce partial overlaps (e.g. "sql" inside "postgresql")
    unique = sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True)
    if not unique:
        return text

    out = text
    for term in unique[:30]:
        pat = re.compile(rf"(?i)\b({re.escape(term)})\b")
        out = pat.sub(r"<mark>\1</mark>", out)
    return out


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@dataclass(frozen=True)
class AllocationConfig:
    enforce_unique_internship: bool = False
    capacity_column: Optional[str] = None

