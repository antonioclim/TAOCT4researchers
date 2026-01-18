#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Practice Exercise: Hard 02 - Database Schema Design
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
"The purpose of an index is to speed up access to data. Given a specific
search condition, an index can be used to locate records quickly, without
examining all the data."
— Garcia-Molina et al., 2008

Effective database schema design requires understanding normalisation,
referential integrity, and query optimisation. This exercise challenges
you to design and implement a normalised schema for a complex research
data management scenario.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Design normalised database schemas (up to 3NF)
2. Implement foreign key constraints and cascading rules
3. Create indices for query optimisation
4. Write complex JOIN queries across multiple tables

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 45 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO: Research Publication Database
# ═══════════════════════════════════════════════════════════════════════════════
#
# Design a database for managing research publications with the following
# requirements:
#
# - Researchers have names, emails, and affiliations
# - Publications have titles, abstracts, DOIs, and publication dates
# - A publication can have multiple authors (researchers)
# - Authors have an order (first author, second author, etc.)
# - Publications belong to journals
# - Journals have names, ISSN numbers, and impact factors
# - Publications can have multiple keywords
# - Keywords are shared across publications
#
# Your schema must:
# - Be in Third Normal Form (3NF)
# - Use appropriate foreign key constraints
# - Include indices for common query patterns
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Create Schema
# ═══════════════════════════════════════════════════════════════════════════════

def create_publication_schema(conn: sqlite3.Connection) -> None:
    """
    Create the normalised publication database schema.

    Required tables:
    - researchers: id, name, email (unique), affiliation
    - journals: id, name, issn (unique), impact_factor
    - publications: id, title, abstract, doi (unique), publication_date, journal_id (FK)
    - publication_authors: publication_id (FK), researcher_id (FK), author_order
    - keywords: id, keyword (unique)
    - publication_keywords: publication_id (FK), keyword_id (FK)

    Must include:
    - Primary keys on all tables
    - Foreign keys with appropriate CASCADE rules
    - Unique constraints where specified
    - Indices on foreign keys and commonly queried columns

    Args:
        conn: SQLite database connection.
    """
    # TODO: Implement the schema creation
    # Hint: Use CREATE TABLE IF NOT EXISTS with appropriate constraints
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Data Insertion Functions
# ═══════════════════════════════════════════════════════════════════════════════

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


def insert_researcher(conn: sqlite3.Connection, researcher: Researcher) -> int:
    """
    Insert a researcher and return their ID.

    Args:
        conn: Database connection.
        researcher: Researcher to insert.

    Returns:
        ID of inserted researcher.
    """
    # TODO: Implement this function
    pass


def insert_journal(conn: sqlite3.Connection, journal: Journal) -> int:
    """
    Insert a journal and return its ID.

    Args:
        conn: Database connection.
        journal: Journal to insert.

    Returns:
        ID of inserted journal.
    """
    # TODO: Implement this function
    pass


def insert_publication(conn: sqlite3.Connection, publication: Publication) -> int:
    """
    Insert a publication and return its ID.

    Args:
        conn: Database connection.
        publication: Publication to insert.

    Returns:
        ID of inserted publication.
    """
    # TODO: Implement this function
    pass


def add_author_to_publication(
    conn: sqlite3.Connection,
    publication_id: int,
    researcher_id: int,
    author_order: int
) -> None:
    """
    Link an author to a publication with specified order.

    Args:
        conn: Database connection.
        publication_id: ID of the publication.
        researcher_id: ID of the researcher/author.
        author_order: Position in author list (1 = first author).
    """
    # TODO: Implement this function
    pass


def add_keyword_to_publication(
    conn: sqlite3.Connection,
    publication_id: int,
    keyword: str
) -> None:
    """
    Add a keyword to a publication, creating the keyword if needed.

    Args:
        conn: Database connection.
        publication_id: ID of the publication.
        keyword: Keyword text.
    """
    # TODO: Implement this function
    # Hint: Use INSERT OR IGNORE for the keyword, then link
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Complex Queries
# ═══════════════════════════════════════════════════════════════════════════════

def get_publications_by_researcher(
    conn: sqlite3.Connection,
    researcher_email: str
) -> list[dict[str, Any]]:
    """
    Get all publications for a researcher.

    Returns publications with: title, doi, publication_date, journal_name,
    author_order.

    Args:
        conn: Database connection.
        researcher_email: Email of the researcher.

    Returns:
        List of publication dictionaries.
    """
    # TODO: Implement this function
    # Hint: Use JOINs across researchers, publication_authors, publications, journals
    pass


def get_publication_with_authors(
    conn: sqlite3.Connection,
    doi: str
) -> dict[str, Any] | None:
    """
    Get publication details with all authors in order.

    Returns: title, abstract, doi, publication_date, journal_name,
    authors (list of names in order), keywords (list).

    Args:
        conn: Database connection.
        doi: DOI of the publication.

    Returns:
        Publication dictionary or None if not found.
    """
    # TODO: Implement this function
    pass


def get_coauthor_network(
    conn: sqlite3.Connection,
    researcher_email: str
) -> list[dict[str, Any]]:
    """
    Find all coauthors of a researcher.

    Returns: coauthor_name, coauthor_email, collaboration_count
    (number of shared publications).

    Args:
        conn: Database connection.
        researcher_email: Email of the researcher.

    Returns:
        List of coauthor dictionaries.
    """
    # TODO: Implement this function
    # Hint: Self-join through publication_authors
    pass


def get_journal_statistics(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    Get statistics for each journal.

    Returns: journal_name, issn, impact_factor, publication_count,
    unique_author_count, avg_authors_per_paper.

    Args:
        conn: Database connection.

    Returns:
        List of journal statistics dictionaries.
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Schema Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_schema_normalisation(conn: sqlite3.Connection) -> list[str]:
    """
    Check if the schema follows normalisation principles.

    Validates:
    - All tables have primary keys
    - All foreign keys are properly defined
    - No obvious redundancies exist

    Args:
        conn: Database connection.

    Returns:
        List of validation error messages (empty if valid).
    """
    # TODO: Implement this function
    # Hint: Use PRAGMA table_info and PRAGMA foreign_key_list
    pass


def get_index_usage_report(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    Generate a report of all indices in the database.

    Returns: table_name, index_name, columns, is_unique.

    Args:
        conn: Database connection.

    Returns:
        List of index information dictionaries.
    """
    # TODO: Implement this function
    # Hint: Use PRAGMA index_list and PRAGMA index_info
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Run comprehensive tests for the schema design."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / 'publications.db'

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

        print("Testing Exercise 1: Schema creation")
        create_publication_schema(conn)
        # Verify tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row['name'] for row in cursor.fetchall()}
        required = {'researchers', 'journals', 'publications',
                    'publication_authors', 'keywords', 'publication_keywords'}
        assert required.issubset(tables), f"Missing tables: {required - tables}"
        print("  ✓ Schema created successfully")

        print("\nTesting Exercise 2: Data insertion")
        # Insert test data
        r1 = insert_researcher(conn, Researcher(
            name='Alice Smith', email='alice@uni.edu', affiliation='MIT'
        ))
        r2 = insert_researcher(conn, Researcher(
            name='Bob Jones', email='bob@uni.edu', affiliation='Stanford'
        ))

        j1 = insert_journal(conn, Journal(
            name='Nature', issn='0028-0836', impact_factor=49.962
        ))

        p1 = insert_publication(conn, Publication(
            title='Groundbreaking Research',
            abstract='This paper presents...',
            doi='10.1000/test.001',
            publication_date='2024-01-15',
            journal_id=j1
        ))

        add_author_to_publication(conn, p1, r1, 1)
        add_author_to_publication(conn, p1, r2, 2)
        add_keyword_to_publication(conn, p1, 'machine learning')
        add_keyword_to_publication(conn, p1, 'research')
        print("  ✓ Data insertion works correctly")

        print("\nTesting Exercise 3: Complex queries")
        pubs = get_publications_by_researcher(conn, 'alice@uni.edu')
        assert len(pubs) == 1, "Should find 1 publication"
        assert pubs[0]['title'] == 'Groundbreaking Research'
        print("  ✓ get_publications_by_researcher works")

        pub_details = get_publication_with_authors(conn, '10.1000/test.001')
        assert pub_details is not None
        assert len(pub_details['authors']) == 2
        assert pub_details['authors'][0] == 'Alice Smith'  # First author
        print("  ✓ get_publication_with_authors works")

        coauthors = get_coauthor_network(conn, 'alice@uni.edu')
        assert len(coauthors) == 1
        assert coauthors[0]['coauthor_email'] == 'bob@uni.edu'
        print("  ✓ get_coauthor_network works")

        stats = get_journal_statistics(conn)
        assert len(stats) == 1
        assert stats[0]['publication_count'] == 1
        print("  ✓ get_journal_statistics works")

        print("\nTesting Exercise 4: Schema validation")
        errors = validate_schema_normalisation(conn)
        assert len(errors) == 0, f"Validation errors: {errors}"
        print("  ✓ Schema validation passes")

        indices = get_index_usage_report(conn)
        assert len(indices) > 0, "Should have indices"
        print(f"  ✓ Found {len(indices)} indices")

        conn.close()

        print("\n" + "=" * 60)
        print("All tests passed! 🎉")
        print("=" * 60)


if __name__ == "__main__":
    run_tests()
