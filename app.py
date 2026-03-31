from __future__ import annotations

import random

import streamlit as st

from recommender import NetflixRecommender


st.set_page_config(
    page_title="Netflix Recommender Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_recommender() -> NetflixRecommender:
    return NetflixRecommender()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400;600;700;800&family=Space+Grotesk:wght@400;500;700&display=swap');

        :root {
            --bg-main: #12080a;
            --bg-panel: rgba(26, 12, 14, 0.78);
            --text-main: #fff9f2;
            --text-soft: #f6d8c1;
            --accent: #ff5f45;
            --accent-2: #ffd166;
            --accent-3: #53f3c3;
            --stroke: rgba(255,255,255,0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 95, 69, 0.24), transparent 28%),
                radial-gradient(circle at 85% 15%, rgba(83, 243, 195, 0.18), transparent 22%),
                radial-gradient(circle at 50% 120%, rgba(255, 209, 102, 0.16), transparent 26%),
                linear-gradient(160deg, #090506 0%, #16090d 45%, #241116 100%);
            color: var(--text-main);
        }

        html, body, [class*="css"]  {
            font-family: "Space Grotesk", sans-serif;
        }

        h1, h2, h3 {
            font-family: "Bricolage Grotesque", sans-serif !important;
            letter-spacing: -0.03em;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2rem 2.2rem;
            border: 1px solid var(--stroke);
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255,95,69,0.18), rgba(255,209,102,0.09) 42%, rgba(14,10,12,0.82) 88%);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.32);
            overflow: hidden;
            position: relative;
        }

        .hero:before {
            content: "";
            position: absolute;
            inset: auto -90px -90px auto;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(83,243,195,0.38), transparent 62%);
            filter: blur(10px);
        }

        .eyebrow {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            color: var(--accent-2);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero h1 {
            font-size: clamp(2.5rem, 4vw, 4.7rem);
            line-height: 0.95;
            margin: 0.8rem 0 1rem;
            color: var(--text-main);
        }

        .hero p {
            max-width: 700px;
            color: var(--text-soft);
            font-size: 1.05rem;
        }

        .stat-chip {
            display: inline-flex;
            gap: 0.45rem;
            align-items: center;
            padding: 0.6rem 0.9rem;
            margin: 0.4rem 0.55rem 0 0;
            border-radius: 999px;
            border: 1px solid var(--stroke);
            background: rgba(255,255,255,0.06);
            color: var(--text-main);
            font-weight: 600;
        }

        .panel {
            padding: 1.2rem 1.2rem 0.8rem;
            border-radius: 24px;
            border: 1px solid var(--stroke);
            background: var(--bg-panel);
            backdrop-filter: blur(12px);
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
            color: var(--accent-2);
        }

        .rec-card {
            padding: 1.1rem;
            border-radius: 22px;
            border: 1px solid var(--stroke);
            background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.025));
            min-height: 255px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }

        .rec-kicker {
            color: var(--accent-3);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.76rem;
            font-weight: 700;
        }

        .rec-title {
            font-family: "Bricolage Grotesque", sans-serif;
            font-size: 1.45rem;
            line-height: 1;
            margin: 0.4rem 0 0.7rem;
            color: var(--text-main);
        }

        .meta-line {
            color: var(--text-soft);
            font-size: 0.92rem;
            margin-bottom: 0.65rem;
        }

        .reason-pill {
            display: inline-block;
            margin-top: 0.8rem;
            padding: 0.45rem 0.7rem;
            border-radius: 14px;
            background: rgba(255, 95, 69, 0.14);
            color: #ffd9ce;
            font-size: 0.85rem;
        }

        .stTextInput input, .stSelectbox [data-baseweb="select"] > div {
            border-radius: 16px !important;
            background: rgba(255,255,255,0.06) !important;
            color: var(--text-main) !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }

        .stButton > button {
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.1);
            background: linear-gradient(135deg, #ff5f45, #ff8c42);
            color: white;
            font-weight: 800;
            padding: 0.7rem 1rem;
        }

        .stSlider [data-baseweb="slider"] {
            padding-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card(rec) -> None:
    genres = " • ".join(rec.genres[:3]) if rec.genres else "Genre surprise"
    st.markdown(
        f"""
        <div class="rec-card">
            <div class="rec-kicker">Match score {rec.score:.0%}</div>
            <div class="rec-title">{rec.title}</div>
            <div class="meta-line">{rec.content_type} • {rec.release_year} • {rec.rating} • {rec.duration}</div>
            <div class="meta-line">{genres}</div>
            <div class="meta-line">{rec.description[:165]}{'...' if len(rec.description) > 165 else ''}</div>
            <div class="reason-pill">{rec.reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()
engine = load_recommender()

if "selected_title" not in st.session_state:
    st.session_state.selected_title = "Stranger Things"

if "surprise_seed" not in st.session_state:
    st.session_state.surprise_seed = random.randint(0, 100000)

catalog_size = len(engine.df)
movie_count = int((engine.df["type"] == "Movie").sum())
show_count = int((engine.df["type"] == "TV Show").sum())

st.markdown(
    f"""
    <section class="hero">
        <span class="eyebrow">Movie Night Control Room</span>
        <h1>Find your next Netflix obsession.</h1>
        <p>
            This rebuilt studio blends descriptions, genres, cast, creators, release era, and content type
            into a stronger recommendation engine, then wraps it in a brighter, more playful Streamlit experience.
        </p>
        <div>
            <span class="stat-chip">🎞 {catalog_size:,} titles</span>
            <span class="stat-chip">🍿 {movie_count:,} movies</span>
            <span class="stat-chip">📺 {show_count:,} shows</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tune the vibe</div>', unsafe_allow_html=True)

    search_text = st.text_input(
        "Search titles",
        value=st.session_state.selected_title,
        placeholder="Try Wednesday, Narcos, The Crown...",
    )
    suggestions = engine.suggest_titles(search_text, limit=8)
    if suggestions:
        selected_title = st.selectbox("Pick a title", suggestions, index=0)
    else:
        selected_title = search_text

    type_filter = st.selectbox("Keep recommendations to", ["Any", "Movie", "TV Show"], index=0)
    result_count = st.slider("How many matches", min_value=6, max_value=18, value=9, step=3)
    release_window = st.slider("How tightly to match release era", min_value=10, max_value=40, value=24, step=2)

    control_1, control_2 = st.columns(2)
    with control_1:
        if st.button("Recommend for me", use_container_width=True):
            st.session_state.selected_title = selected_title
    with control_2:
        if st.button("Surprise me", use_container_width=True):
            st.session_state.selected_title = engine.random_title()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    current_title = st.session_state.selected_title
    selected_row = engine.df[engine.df["title"] == current_title].head(1)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tonight’s anchor title</div>', unsafe_allow_html=True)
    if not selected_row.empty:
        row = selected_row.iloc[0]
        st.markdown(
            f"""
            <div class="rec-card">
                <div class="rec-kicker">Because you picked</div>
                <div class="rec-title">{row['title']}</div>
                <div class="meta-line">{row['type']} • {row['release_year']} • {row['rating']} • {row['duration']}</div>
                <div class="meta-line">{' • '.join(row['genres'][:4])}</div>
                <div class="meta-line">{row['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")
st.markdown('<div class="section-title">Your queue, reimagined</div>', unsafe_allow_html=True)

try:
    recommendations = engine.recommend(
        st.session_state.selected_title,
        top_n=result_count,
        content_type=type_filter,
        release_window=release_window,
    )
except ValueError as exc:
    st.error(str(exc))
else:
    columns = st.columns(3, gap="large")
    for index, recommendation in enumerate(recommendations):
        with columns[index % 3]:
            render_card(recommendation)
