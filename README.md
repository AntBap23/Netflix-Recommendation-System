# Netflix Recommender Studio

Netflix Recommender Studio is a creative Streamlit app that helps users discover movies and TV shows from the Netflix catalog with a stronger content-based recommendation engine.

It was rebuilt from an earlier Gradio prototype into a cleaner, more polished project with:
- a top-level project structure
- a Streamlit frontend
- a richer recommendation model
- a more playful, cinematic interface

## Live Demo

Try the deployed app here:

[Netflix Recommender Studio](https://netflix-movie-picker.streamlit.app/)

## Features

- Streamlit interface designed to feel fun and interactive
- Smarter recommendations using:
  - title metadata
  - genres
  - descriptions
  - cast
  - directors
  - country
  - release decade
  - rating bucket
  - duration / season structure
- Fuzzy title suggestions for misspelled searches
- “Surprise me” title picker
- Match explanations for each recommendation

## Recommendation Model

This project uses a content-based recommendation approach.

Each title is transformed into a weighted text representation that combines:
- genres
- content type
- cast
- directors
- countries
- release era
- audience rating category
- duration / season information
- description text

The app then uses TF-IDF vectorization and similarity scoring to find titles that are most related to the selected one. Ranking is further improved with small boosts for similar release era and matching content type.

## Project Structure

```text
Netflix-Recommendation-System/
├── app.py
├── recommender.py
├── requirements.txt
├── data/
│   └── netflix_titles.csv
├── .gitignore
└── README.md
```

## Run Locally

Clone the repo and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Deployment

This app can be deployed easily on Streamlit Community Cloud.

### Deploy on Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to https://share.streamlit.io/
3. Connect your GitHub account
4. Select this repository
5. Set the entry file to `app.py`
6. Deploy

## GitHub Notes

The local virtual environment should not be committed.

This repo ignores:
- `.venv/`
- `__pycache__/`
- `.streamlit/`

That means GitHub will only store your source code and project files, not your local environment.

## Dataset

The app uses the Netflix titles dataset stored at:

```text
data/netflix_titles.csv
```

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn

## License

This project is licensed under the MIT License.
