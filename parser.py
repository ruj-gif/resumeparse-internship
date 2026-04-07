from __future__ import annotations

import io
import re
from typing import List, Sequence, Tuple

from utils import clean_text


# A pragmatic baseline list. You can extend this easily.
SKILLS_LIST: List[str] = [
    # Programming / DS
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "r",
    "matlab",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "sqlite",
    # Data / ML
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "computer vision",
    "data science",
    "data analysis",
    "statistics",
    "time series",
    "pandas",
    "numpy",
    "scikit-learn",
    "sklearn",
    "tensorflow",
    "pytorch",
    "keras",
    "xgboost",
    "lightgbm",
    # Viz / BI
    "power bi",
    "tableau",
    "excel",
    "matplotlib",
    "seaborn",
    "plotly",
    # Cloud / Dev
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "linux",
    "rest",
    "api",
    "fastapi",
    "flask",
    "django",
    "streamlit",
    # LLMs / GenAI
    "llm",
    "openai",
    "prompt engineering",
    "rag",
]


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract raw text from PDF bytes using pdfplumber.
    """
    if not pdf_bytes:
        return ""
    try:
        import pdfplumber  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency: pdfplumber. Install it in the SAME Python environment "
            "used to run Streamlit (e.g. `python -m pip install pdfplumber`)."
        ) from e
    text_parts: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                text_parts.append(t)
    return "\n".join(text_parts).strip()


def _compile_skill_patterns(skills: Sequence[str]) -> List[Tuple[str, re.Pattern]]:
    patterns: List[Tuple[str, re.Pattern]] = []
    for s in skills:
        s_norm = clean_text(s)
        if not s_norm:
            continue
        # word-boundary for most, but allow "c++" / "c#" matching
        if s_norm in {"c++", "c#"}:
            pat = re.compile(rf"(?i)(?:^|[^a-z0-9])({re.escape(s_norm)})(?:$|[^a-z0-9])")
        else:
            pat = re.compile(rf"(?i)\b{re.escape(s_norm)}\b")
        patterns.append((s_norm, pat))
    return patterns


_SKILL_PATTERNS = _compile_skill_patterns(SKILLS_LIST)


def extract_skills(text: str, skills_list: Sequence[str] | None = None) -> List[str]:
    """
    Extract skills by exact/phrase matching against a predefined list.
    """
    if not text:
        return []
    t = clean_text(text)
    patterns = _SKILL_PATTERNS if skills_list is None else _compile_skill_patterns(skills_list)
    found: List[str] = []
    for skill, pat in patterns:
        if pat.search(t):
            found.append(skill)
    return sorted(set(found))


def parse_resume(uploaded_file) -> Tuple[str, str, List[str]]:
    """
    Returns: (filename, cleaned_text, skills)
    """
    if uploaded_file is None:
        return ("", "", [])
    filename = getattr(uploaded_file, "name", "resume.pdf")
    try:
        pdf_bytes = uploaded_file.read()
    except Exception:
        return (filename, "", [])
    raw = extract_text_from_pdf_bytes(pdf_bytes)
    cleaned = clean_text(raw)
    skills = extract_skills(cleaned)
    return (filename, cleaned, skills)

