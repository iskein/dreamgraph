"""Synthetic 500-document news corpus with synonym overlap."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

N_DOCS = 500
SEED = 42
CORPUS_PATH = Path(__file__).parent / "generated" / "corpus.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

SYNONYMS = ("car", "automobile", "vehicle")


@dataclass
class Document:
    id: str
    title: str
    body: str
    terms: list[str]
    days_since_publish: int
    tags: list[str] = field(default_factory=list)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens from title/body text."""
    return _TOKEN_RE.findall(text.lower())


def corpus_path() -> Path:
    return CORPUS_PATH


def generate(path: Path | None = None) -> list[Document]:
    """Generate 500 synthetic documents and persist them as JSON."""
    rng = random.Random(SEED)
    docs: list[Document] = []
    next_id = 0

    auto_groups = [
        (30, ["car"]),
        (30, ["automobile"]),
        (30, ["vehicle"]),
        (15, ["car", "automobile"]),
        (15, ["car", "vehicle"]),
        (15, ["automobile", "vehicle"]),
        (10, ["car", "automobile", "vehicle"]),
    ]
    auto_extra = [
        "engine", "wheels", "fuel", "traffic", "road", "parking",
        "driver", "motor", "mileage", "safety", "highway", "fleet",
        "sedan", "dealership", "transport", "assembly",
    ]
    for count, syns in auto_groups:
        for _ in range(count):
            extras = rng.sample(auto_extra, k=4)
            title = _auto_title(syns, extras, rng)
            body = _auto_body(syns, extras)
            docs.append(_make_doc(next_id, title, body, rng, tags=["automotive", *syns]))
            next_id += 1

    topics: list[tuple[str, list[str]]] = [
        ("sports", ["team", "match", "score", "player", "league", "tournament", "coach", "stadium"]),
        ("tech", ["software", "chip", "startup", "cloud", "algorithm", "device", "network", "update"]),
        ("food", ["recipe", "restaurant", "flavor", "kitchen", "chef", "ingredient", "dinner", "market"]),
        ("travel", ["flight", "hotel", "destination", "passport", "luggage", "tour", "beach", "city"]),
        ("health", ["doctor", "clinic", "vaccine", "fitness", "nutrition", "hospital", "patient", "study"]),
        ("finance", ["stock", "market", "bank", "credit", "inflation", "investor", "portfolio", "rate"]),
        ("science", ["research", "climate", "genome", "physics", "lab", "experiment", "satellite", "data"]),
    ]
    remaining = N_DOCS - len(docs)
    base, extra = divmod(remaining, len(topics))
    for ti, (topic, vocab) in enumerate(topics):
        n = base + (1 if ti < extra else 0)
        for _ in range(n):
            words = rng.sample(vocab, k=min(5, len(vocab)))
            title = f"{topic.title()} report: {words[0]} {words[1]}"
            body = (
                f"A {topic} update covering {', '.join(words)}. "
                f"Analysts note {words[0]} trends around {words[-1]} "
                f"and related {topic} developments this season."
            )
            docs.append(_make_doc(next_id, title, body, rng, tags=[topic, *words[:2]]))
            next_id += 1

    assert len(docs) == N_DOCS
    dest = path or CORPUS_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps([asdict(d) for d in docs], indent=2), encoding="utf-8")
    return docs


def load(path: Path | None = None) -> list[Document]:
    """Load the persisted corpus, generating it if needed."""
    dest = path or CORPUS_PATH
    if not dest.exists():
        return generate(dest)
    raw = json.loads(dest.read_text(encoding="utf-8"))
    return [Document(**row) for row in raw]


def ensure_corpus() -> list[Document]:
    """Create the corpus JSON if it does not already exist."""
    if CORPUS_PATH.exists():
        return load()
    return generate()


def _make_doc(i: int, title: str, body: str, rng: random.Random, tags: list[str]) -> Document:
    days = rng.randint(0, 365)
    terms = tokenize(f"{title} {body}")
    return Document(
        id=f"d{i:03d}",
        title=title,
        body=body,
        terms=terms,
        days_since_publish=days,
        tags=tags,
    )


def _auto_title(syns: list[str], extras: list[str], rng: random.Random) -> str:
    focus = rng.choice(syns)
    return f"{focus.title()} news: {extras[0]} and {extras[1]}"


def _auto_body(syns: list[str], extras: list[str]) -> str:
    mentioned = " ".join(syns)
    return (
        f"Coverage of {mentioned} issues including {', '.join(extras)}. "
        f"The {syns[0]} segment also discusses {extras[-1]} and {extras[0]}."
    )
