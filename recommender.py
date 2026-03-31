from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from difflib import get_close_matches

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = Path(__file__).resolve().parent / "data" / "netflix_titles.csv"


def _safe_split(value: object, limit: int | None = None) -> list[str]:
    if pd.isna(value):
        return []
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items[:limit] if limit else items


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _parse_duration(value: object) -> tuple[int, int]:
    if pd.isna(value):
        return 0, 0
    text = str(value)
    if "min" in text:
        return int(text.split()[0]), 0
    if "Season" in text:
        return 0, int(text.split()[0])
    return 0, 0


def _rating_bucket(rating: object) -> str:
    kids = {"TV-Y", "TV-Y7", "G", "TV-G"}
    general = {"PG", "PG-13", "TV-PG", "TV-14"}
    mature = {"R", "TV-MA", "NC-17", "UR"}
    if pd.isna(rating):
        return "unknown"
    if rating in kids:
        return "kids"
    if rating in general:
        return "general"
    if rating in mature:
        return "mature"
    return "unknown"


@dataclass
class Recommendation:
    title: str
    release_year: int
    content_type: str
    rating: str
    duration: str
    genres: list[str]
    description: str
    score: float
    reason: str


class NetflixRecommender:
    def __init__(self, data_path: Path = DATA_PATH) -> None:
        self.data_path = data_path
        self.df = self._prepare_dataframe(pd.read_csv(data_path))
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=15000,
        )
        self.feature_matrix = self.vectorizer.fit_transform(self.df["feature_soup"])
        popularity_frame = self.df[["release_year", "cast_size", "director_known", "description_length"]]
        self.popularity_signal = MinMaxScaler().fit_transform(popularity_frame).mean(axis=1)
        self.indices = pd.Series(self.df.index, index=self.df["title_key"]).drop_duplicates()

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        prepared["director"] = prepared["director"].fillna("")
        prepared["cast"] = prepared["cast"].fillna("")
        prepared["country"] = prepared["country"].fillna("Unknown")
        prepared["rating"] = prepared["rating"].fillna("Unrated")
        prepared["description"] = prepared["description"].fillna("")
        prepared["listed_in"] = prepared["listed_in"].fillna("")
        prepared["title_key"] = prepared["title"].map(_slug)

        duration_parts = prepared["duration"].apply(_parse_duration)
        prepared["duration_minutes"] = duration_parts.map(lambda item: item[0])
        prepared["season_count"] = duration_parts.map(lambda item: item[1])
        prepared["genres"] = prepared["listed_in"].apply(_safe_split)
        prepared["cast_members"] = prepared["cast"].apply(lambda value: _safe_split(value, limit=5))
        prepared["directors"] = prepared["director"].apply(_safe_split)
        prepared["countries"] = prepared["country"].apply(lambda value: _safe_split(value, limit=3))
        prepared["decade"] = (prepared["release_year"] // 10 * 10).astype(str) + "s"
        prepared["cast_size"] = prepared["cast"].apply(lambda value: len(_safe_split(value)))
        prepared["director_known"] = (prepared["director"].str.len() > 0).astype(int)
        prepared["description_length"] = prepared["description"].str.split().map(len)
        prepared["rating_bucket"] = prepared["rating"].apply(_rating_bucket)
        prepared["feature_soup"] = prepared.apply(self._build_feature_soup, axis=1)
        return prepared

    def _build_feature_soup(self, row: pd.Series) -> str:
        title_tokens = _slug(row["title"]).split()
        boosted_genres = [genre.lower().replace(" ", "_") for genre in row["genres"] for _ in range(3)]
        boosted_type = [row["type"].lower().replace(" ", "_")] * 2
        boosted_directors = [director.lower().replace(" ", "_") for director in row["directors"] for _ in range(2)]
        cast_tokens = [member.lower().replace(" ", "_") for member in row["cast_members"]]
        country_tokens = [country.lower().replace(" ", "_") for country in row["countries"]]
        description = row["description"].lower()
        extras = [
            f"decade_{row['decade']}",
            f"audience_{row['rating_bucket']}",
            f"duration_{'short' if row['duration_minutes'] and row['duration_minutes'] < 90 else 'feature'}",
            f"seasons_{'series' if row['season_count'] else 'single'}",
        ]
        parts = title_tokens + boosted_genres + boosted_type + boosted_directors + cast_tokens + country_tokens + extras
        return " ".join(parts) + " " + description

    def suggest_titles(self, query: str, limit: int = 8) -> list[str]:
        if not query.strip():
            return self.df["title"].head(limit).tolist()
        lowered = _slug(query)
        title_keys = self.df["title_key"].tolist()
        matches = get_close_matches(lowered, title_keys, n=limit, cutoff=0.35)
        exact_like = self.df[self.df["title_key"].str.contains(re.escape(lowered), na=False)]["title"].tolist()
        suggestions: list[str] = []
        for key in matches:
            suggestions.append(self.df.loc[self.indices[key], "title"])
        suggestions.extend(exact_like)
        deduped: list[str] = []
        seen = set()
        for title in suggestions:
            if title not in seen:
                seen.add(title)
                deduped.append(title)
        return deduped[:limit]

    def random_title(self) -> str:
        return self.df.sample(1, random_state=np.random.randint(0, 100000))["title"].iat[0]

    def recommend(
        self,
        title: str,
        top_n: int = 12,
        content_type: str = "Any",
        release_window: int = 30,
    ) -> list[Recommendation]:
        key = _slug(title)
        if key not in self.indices:
            suggestions = self.suggest_titles(title, limit=5)
            hint = f"Try: {', '.join(suggestions)}" if suggestions else "Try another title from the catalog."
            raise ValueError(f"Title not found. {hint}")

        idx = int(self.indices[key])
        query_vector = self.feature_matrix[idx]
        similarities = linear_kernel(query_vector, self.feature_matrix).flatten()

        base_year = int(self.df.loc[idx, "release_year"])
        year_distance = np.abs(self.df["release_year"] - base_year)
        year_bonus = 1 - np.clip(year_distance / max(release_window, 1), 0, 1)
        same_type_bonus = (self.df["type"] == self.df.loc[idx, "type"]).astype(float).to_numpy()
        final_scores = similarities * 0.82 + year_bonus.to_numpy() * 0.10 + same_type_bonus * 0.04 + self.popularity_signal * 0.04

        scored = self.df.assign(score=final_scores).drop(index=idx)
        if content_type != "Any":
            scored = scored[scored["type"] == content_type]

        ranked = scored.sort_values("score", ascending=False).head(top_n)
        selected = self.df.loc[idx]
        return [
            Recommendation(
                title=row["title"],
                release_year=int(row["release_year"]),
                content_type=row["type"],
                rating=row["rating"],
                duration=row["duration"],
                genres=row["genres"],
                description=row["description"],
                score=float(row["score"]),
                reason=self._build_reason(selected, row),
            )
            for _, row in ranked.iterrows()
        ]

    def _build_reason(self, selected: pd.Series, candidate: pd.Series) -> str:
        shared_genres = [genre for genre in candidate["genres"] if genre in selected["genres"]]
        shared_cast = [name for name in candidate["cast_members"] if name in selected["cast_members"]]
        shared_country = [country for country in candidate["countries"] if country in selected["countries"]]

        reasons: list[str] = []
        if shared_genres:
            reasons.append("shares " + ", ".join(shared_genres[:2]).lower())
        if shared_cast:
            reasons.append("features familiar cast")
        if candidate["type"] == selected["type"]:
            reasons.append(f"stays in the {candidate['type'].lower()} lane")
        if shared_country:
            reasons.append("comes from a similar production region")

        if not reasons:
            reasons.append("matches the overall tone and metadata profile")

        return " and ".join(reasons[:2]).capitalize() + "."
