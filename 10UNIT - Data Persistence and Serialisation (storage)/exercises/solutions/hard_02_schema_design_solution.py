#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Solution: Hard 02 - Database Schema Design
═══════════════════════════════════════════════════════════════════════════════
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Researcher:
    """Researcher data class."""
    name: str
    email: str
    affiliation: str
    researcher_id: int | None = None


@dataclass
class Journal:
    """Journal data class."""
    name: str
    issn: str
    impact_factor: float
    journal_id: int | None = None


@dataclass
class Publication:
    """Publication data class."""
    title: str
    abstract: str
    doi: str
    publication_date: str
    journal_id: int
    publication_id: int | None = None


def create_publication_schema(conn: sqlite3.Connection) -> None:
    """Create the normalised publication database schema."""
    conn.executescript("""
        -- Researchers table
        CREATE TABLE IF NOT EXISTS researchers (
            researcher_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            affiliation TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Journals table
        CREATE TABLE IF NOT EXISTS journals (
            journal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            issn TEXT UNIQUE NOT NULL,
            impact_factor REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Publications table
        CREATE TABLE IF NOT EXISTS publications (
            publication_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            abstract TEXT,
            doi TEXT UNIQUE NOT NULL,
            publication_date TEXT,
            journal_id INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (journal_id) REFERENCES journals(journal_id)
                ON DELETE RESTRICT ON UPDATE CASCADE
        );
        
        -- Publication authors (many-to-many with ordering)
        CREATE TABLE IF NOT EXISTS publication_authors (
            publication_id INTEGER NOT NULL,
            researcher_id INTEGER NOT NULL,
            author_order INTEGER NOT NULL,
            PRIMARY KEY (publication_id, researcher_id),
            FOREIGN KEY (publication_id) REFERENCES publications(publication_id)
                ON DELETE CASCADE ON UPDATE CASCADE,
            FOREIGN KEY (researcher_id) REFERENCES researchers(researcher_id)
                ON DELETE RESTRICT ON UPDATE CASCADE
        );
        
        -- Keywords table
        CREATE TABLE IF NOT EXISTS keywords (
            keyword_id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT UNIQUE NOT NULL
        );
        
        -- Publication keywords (many-to-many)
        CREATE TABLE IF NOT EXISTS publication_keywords (
            publication_id INTEGER NOT NULL,
            keyword_id INTEGER NOT NULL,
            PRIMARY KEY (publication_id, keyword_id),
            FOREIGN KEY (publication_id) REFERENCES publications(publication_id)
                ON DELETE CASCADE ON UPDATE CASCADE,
            FOREIGN KEY (keyword_id) REFERENCES keywords(keyword_id)
                ON DELETE CASCADE ON UPDATE CASCADE
        );
        
        -- Indices for common queries
        CREATE INDEX IF NOT EXISTS idx_publications_journal 
            ON publications(journal_id);
        CREATE INDEX IF NOT EXISTS idx_publications_date 
            ON publications(publication_date);
        CREATE INDEX IF NOT EXISTS idx_pub_authors_researcher 
            ON publication_authors(researcher_id);
        CREATE INDEX IF NOT EXISTS idx_pub_keywords_keyword 
            ON publication_keywords(keyword_id);
    """)
    conn.commit()


def insert_researcher(conn: sqlite3.Connection, researcher: Researcher) -> int:
    """Insert a researcher and return their ID."""
    cursor = conn.execute(
        """
        INSERT INTO researchers (name, email, affiliation)
        VALUES (?, ?, ?)
        """,
        (researcher.name, researcher.email, researcher.affiliation)
    )
    conn.commit()
    return cursor.lastrowid


def insert_journal(conn: sqlite3.Connection, journal: Journal) -> int:
    """Insert a journal and return its ID."""
    cursor = conn.execute(
        """
        INSERT INTO journals (name, issn, impact_factor)
        VALUES (?, ?, ?)
        """,
        (journal.name, journal.issn, journal.impact_factor)
    )
    conn.commit()
    return cursor.lastrowid


def insert_publication(conn: sqlite3.Connection, publication: Publication) -> int:
    """Insert a publication and return its ID."""
    cursor = conn.execute(
        """
        INSERT INTO publications (title, abstract, doi, publication_date, journal_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (publication.title, publication.abstract, publication.doi,
         publication.publication_date, publication.journal_id)
    )
    conn.commit()
    return cursor.lastrowid


def add_author_to_publication(
    conn: sqlite3.Connection,
    publication_id: int,
    researcher_id: int,
    author_order: int
) -> None:
    """Link an author to a publication with specified order."""
    conn.execute(
        """
        INSERT INTO publication_authors (publication_id, researcher_id, author_order)
        VALUES (?, ?, ?)
        """,
        (publication_id, researcher_id, author_order)
    )
    conn.commit()


def add_keyword_to_publication(
    conn: sqlite3.Connection,
    publication_id: int,
    keyword: str
) -> None:
    """Add a keyword to a publication, creating the keyword if needed."""
    # Insert keyword if not exists
    conn.execute(
        "INSERT OR IGNORE INTO keywords (keyword) VALUES (?)",
        (keyword,)
    )
    
    # Get keyword ID
    cursor = conn.execute(
        "SELECT keyword_id FROM keywords WHERE keyword = ?",
        (keyword,)
    )
    keyword_id = cursor.fetchone()[0]
    
    # Link to publication
    conn.execute(
        """
        INSERT OR IGNORE INTO publication_keywords (publication_id, keyword_id)
        VALUES (?, ?)
        """,
        (publication_id, keyword_id)
    )
    conn.commit()


def get_publications_by_researcher(
    conn: sqlite3.Connection,
    researcher_email: str
) -> list[dict[str, Any]]:
    """Get all publications for a researcher."""
    cursor = conn.execute(
        """
        SELECT p.title, p.doi, p.publication_date, j.name as journal_name,
               pa.author_order
        FROM researchers r
        JOIN publication_authors pa ON r.researcher_id = pa.researcher_id
        JOIN publications p ON pa.publication_id = p.publication_id
        JOIN journals j ON p.journal_id = j.journal_id
        WHERE r.email = ?
        ORDER BY p.publication_date DESC
        """,
        (researcher_email,)
    )
    
    return [dict(row) for row in cursor.fetchall()]


def get_publication_with_authors(
    conn: sqlite3.Connection,
    doi: str
) -> dict[str, Any] | None:
    """Get publication details with all authors in order."""
    # Get basic publication info
    cursor = conn.execute(
        """
        SELECT p.title, p.abstract, p.doi, p.publication_date, j.name as journal_name
        FROM publications p
        JOIN journals j ON p.journal_id = j.journal_id
        WHERE p.doi = ?
        """,
        (doi,)
    )
    row = cursor.fetchone()
    if row is None:
        return None
    
    result = dict(row)
    
    # Get publication ID
    cursor = conn.execute(
        "SELECT publication_id FROM publications WHERE doi = ?",
        (doi,)
    )
    pub_id = cursor.fetchone()[0]
    
    # Get authors in order
    cursor = conn.execute(
        """
        SELECT r.name
        FROM publication_authors pa
        JOIN researchers r ON pa.researcher_id = r.researcher_id
        WHERE pa.publication_id = ?
        ORDER BY pa.author_order
        """,
        (pub_id,)
    )
    result['authors'] = [row[0] for row in cursor.fetchall()]
    
    # Get keywords
    cursor = conn.execute(
        """
        SELECT k.keyword
        FROM publication_keywords pk
        JOIN keywords k ON pk.keyword_id = k.keyword_id
        WHERE pk.publication_id = ?
        """,
        (pub_id,)
    )
    result['keywords'] = [row[0] for row in cursor.fetchall()]
    
    return result


def get_coauthor_network(
    conn: sqlite3.Connection,
    researcher_email: str
) -> list[dict[str, Any]]:
    """Find all coauthors of a researcher."""
    cursor = conn.execute(
        """
        SELECT r2.name as coauthor_name, r2.email as coauthor_email,
               COUNT(*) as collaboration_count
        FROM researchers r1
        JOIN publication_authors pa1 ON r1.researcher_id = pa1.researcher_id
        JOIN publication_authors pa2 ON pa1.publication_id = pa2.publication_id
        JOIN researchers r2 ON pa2.researcher_id = r2.researcher_id
        WHERE r1.email = ? AND r2.email != ?
        GROUP BY r2.researcher_id
        ORDER BY collaboration_count DESC
        """,
        (researcher_email, researcher_email)
    )
    
    return [dict(row) for row in cursor.fetchall()]


def get_journal_statistics(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Get statistics for each journal."""
    cursor = conn.execute(
        """
        SELECT j.name as journal_name, j.issn, j.impact_factor,
               COUNT(DISTINCT p.publication_id) as publication_count,
               COUNT(DISTINCT pa.researcher_id) as unique_author_count,
               CAST(COUNT(pa.researcher_id) AS REAL) / 
                   NULLIF(COUNT(DISTINCT p.publication_id), 0) as avg_authors_per_paper
        FROM journals j
        LEFT JOIN publications p ON j.journal_id = p.journal_id
        LEFT JOIN publication_authors pa ON p.publication_id = pa.publication_id
        GROUP BY j.journal_id
        ORDER BY publication_count DESC
        """
    )
    
    return [dict(row) for row in cursor.fetchall()]


def validate_schema_normalisation(conn: sqlite3.Connection) -> list[str]:
    """Check if the schema follows normalisation principles."""
    errors = []
    
    expected_tables = ['researchers', 'journals', 'publications',
                       'publication_authors', 'keywords', 'publication_keywords']
    
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    actual_tables = [row[0] for row in cursor.fetchall()]
    
    for table in expected_tables:
        if table not in actual_tables:
            errors.append(f"Missing table: {table}")
    
    # Check primary keys
    for table in actual_tables:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        has_pk = any(col[5] > 0 for col in cursor.fetchall())
        if not has_pk:
            errors.append(f"Table {table} missing primary key")
    
    return errors


def get_index_usage_report(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Generate a report of all indices in the database."""
    indices = []
    
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = [row[0] for row in cursor.fetchall()]
    
    for table in tables:
        cursor = conn.execute(f"PRAGMA index_list({table})")
        for idx_row in cursor.fetchall():
            idx_name = idx_row[1]
            is_unique = idx_row[2] == 1
            
            # Get columns
            col_cursor = conn.execute(f"PRAGMA index_info({idx_name})")
            columns = [col[2] for col in col_cursor.fetchall()]
            
            indices.append({
                'table_name': table,
                'index_name': idx_name,
                'columns': columns,
                'is_unique': is_unique
            })
    
    return indices


def run_tests() -> None:
    """Run comprehensive tests for the schema design."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / 'publications.db'

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

        print("Testing schema creation")
        create_publication_schema(conn)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row['name'] for row in cursor.fetchall()}
        required = {'researchers', 'journals', 'publications',
                    'publication_authors', 'keywords', 'publication_keywords'}
        assert required.issubset(tables)
        print("  ✓ Schema created")

        print("Testing data insertion")
        r1 = insert_researcher(conn, Researcher('Alice Smith', 'alice@uni.edu', 'MIT'))
        r2 = insert_researcher(conn, Researcher('Bob Jones', 'bob@uni.edu', 'Stanford'))
        j1 = insert_journal(conn, Journal('Nature', '0028-0836', 49.962))
        p1 = insert_publication(conn, Publication(
            'Groundbreaking Research', 'This paper presents...',
            '10.1000/test.001', '2024-01-15', j1
        ))
        add_author_to_publication(conn, p1, r1, 1)
        add_author_to_publication(conn, p1, r2, 2)
        add_keyword_to_publication(conn, p1, 'machine learning')
        print("  ✓ Data inserted")

        print("Testing queries")
        pubs = get_publications_by_researcher(conn, 'alice@uni.edu')
        assert len(pubs) == 1
        print(f"  ✓ Found {len(pubs)} publications for Alice")

        pub = get_publication_with_authors(conn, '10.1000/test.001')
        assert pub['authors'][0] == 'Alice Smith'
        print(f"  ✓ Publication has {len(pub['authors'])} authors")

        coauthors = get_coauthor_network(conn, 'alice@uni.edu')
        assert len(coauthors) == 1
        print(f"  ✓ Found {len(coauthors)} coauthors")

        stats = get_journal_statistics(conn)
        assert stats[0]['publication_count'] == 1
        print("  ✓ Journal statistics computed")

        errors = validate_schema_normalisation(conn)
        assert len(errors) == 0
        print("  ✓ Schema validation passed")

        conn.close()
        print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    run_tests()
