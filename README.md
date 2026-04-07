# Resume Parser + NLP Internship Allocator (Streamlit)

A complete end-to-end AI/NLP mini-project that:

- Uploads **one or multiple resume PDFs**
- Extracts text using **pdfplumber**
- Extracts **skills** from resumes using a predefined skills list
- Loads an **internships dataset** from `data/internship.csv`
- Matches resumes to internships using **TF‑IDF + cosine similarity**
- Allocates the best internship per resume (capacity-aware if available)
- Shows **top matches**, **scores**, and a short **“why matched”** explanation
- Lets you **download results as CSV**

---

## Project structure

- `app.py` — Streamlit UI + end-to-end flow
- `parser.py` — PDF parsing + skill extraction
- `matcher.py` — TF‑IDF matcher + greedy allocation
- `utils.py` — cleaning, dataset helpers, highlighting, CSV download helpers
- `data/internship.csv` — internships dataset (required)
- `requirements.txt` — Python dependencies

---

## Dataset format

The app expects `data/internship.csv` to exist.

It works best if the dataset includes a **description / requirements / skills** field.  
If your dataset only has columns like `internship_title`, `company_name`, `location`, matching will still run, but results will be weaker because there’s little text to compare.

Optional (improves allocation):
- A column named `capacity` (or `openings` / `slots`) to allow multiple resumes to be allocated to the same internship row.

---

## Run locally (Windows)

From the project folder (`d:\ai\ai`):

### 1) Create & activate a virtual environment (recommended)

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3) Start the app

```bash
python -m streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## Common errors & fixes

### `ModuleNotFoundError: No module named 'pdfplumber'`
You’re running Streamlit with a different Python than where you installed packages.

Fix:
- Activate your venv (`.\.venv\Scripts\activate`)
- Run: `python -m pip install -r requirements.txt`
- Start with: `python -m streamlit run app.py`

---

## How matching works (high level)

- **Resume → skills**: PDF text is extracted then matched against a predefined skills list.
- **Internship text**: The app builds a single searchable text field (`combined_text`) from the most relevant dataset columns.
- **TF‑IDF + cosine similarity**: Each resume is compared against each internship row.
- **Explanation**: Shows overlapping TF‑IDF keywords that contributed most to the score.
- **Allocation**: A greedy allocator assigns internships to resumes, respecting capacity if available.

---

## Deploy

### Recommended (easy): Streamlit Community Cloud
1. Push this repo to GitHub
2. Go to Streamlit Community Cloud
3. Select your repo and set the main file to `app.py`

### Render (simple Docker-less deploy)
- Build: `pip install -r requirements.txt`
- Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Vercel (not recommended for Streamlit)
Vercel is designed for frontend/serverless (Next.js). Streamlit usually requires a container/web service runtime.
If you must use Vercel, you’ll likely need a container-based deployment approach.

