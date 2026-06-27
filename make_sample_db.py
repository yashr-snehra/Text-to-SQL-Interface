"""Create sample.db so the app works without the full Spider download."""
import sqlite3

SCHEMA = """
CREATE TABLE artist (
    id INTEGER PRIMARY KEY,
    name TEXT,
    country TEXT
);
CREATE TABLE album (
    id INTEGER PRIMARY KEY,
    title TEXT,
    year INTEGER,
    artist_id INTEGER REFERENCES artist(id)
);
CREATE TABLE track (
    id INTEGER PRIMARY KEY,
    title TEXT,
    duration INTEGER,
    genre TEXT,
    album_id INTEGER REFERENCES album(id)
);
"""

ARTISTS = [
    (1, "The Verve", "UK"),
    (2, "Daft Punk", "France"),
    (3, "Radiohead", "UK"),
]
ALBUMS = [
    (1, "Urban Hymns", 1997, 1),
    (2, "Discovery", 2001, 2),
    (3, "OK Computer", 1997, 3),
    (4, "In Rainbows", 2007, 3),
]
TRACKS = [
    (1, "Bitter Sweet Symphony", 360, "Rock", 1),
    (2, "The Drugs Don't Work", 309, "Rock", 1),
    (3, "One More Time", 320, "Electronic", 2),
    (4, "Harder Better Faster Stronger", 224, "Electronic", 2),
    (5, "Paranoid Android", 383, "Alternative", 3),
    (6, "Karma Police", 261, "Alternative", 3),
    (7, "15 Step", 237, "Alternative", 4),
]


def build(path="sample.db"):
    conn = sqlite3.connect(path)
    try:
        conn.executescript("DROP TABLE IF EXISTS track; DROP TABLE IF EXISTS album; "
                           "DROP TABLE IF EXISTS artist;")
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO artist VALUES (?,?,?)", ARTISTS)
        conn.executemany("INSERT INTO album VALUES (?,?,?,?)", ALBUMS)
        conn.executemany("INSERT INTO track VALUES (?,?,?,?,?)", TRACKS)
        conn.commit()
    finally:
        conn.close()
    print(f"Wrote {path}")


if __name__ == "__main__":
    build()
