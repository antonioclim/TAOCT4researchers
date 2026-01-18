#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Practice Exercise: medium_03_scraping.py

Difficulty: ★★★☆☆ (Medium)
Estimated Time: 30 minutes
Prerequisites: HTML basics, BeautifulSoup

Learning Objectives:
- LO5: Perform ethical web scraping with BeautifulSoup

This exercise focuses on HTML parsing and data extraction.

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Note: BeautifulSoup must be imported as:
# from bs4 import BeautifulSoup


# =============================================================================
# SAMPLE HTML FOR EXERCISES
# =============================================================================

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Research Publications</title>
</head>
<body>
    <header>
        <h1>Journal of Computational Research</h1>
        <nav>
            <a href="/about">About</a>
            <a href="/submit">Submit</a>
            <a href="/archive">Archive</a>
        </nav>
    </header>
    
    <main>
        <section class="publications">
            <article class="paper" data-id="P001">
                <h2 class="title">Machine Learning for Climate Modelling</h2>
                <div class="authors">
                    <span class="author">Smith, J.</span>
                    <span class="author">Chen, L.</span>
                </div>
                <p class="abstract">We present novel approaches to climate prediction...</p>
                <div class="metadata">
                    <span class="year">2024</span>
                    <span class="citations">42</span>
                    <a href="/papers/P001" class="link">Full Text</a>
                </div>
            </article>
            
            <article class="paper" data-id="P002">
                <h2 class="title">Deep Learning in Genomics</h2>
                <div class="authors">
                    <span class="author">Johnson, M.</span>
                </div>
                <p class="abstract">A comprehensive survey of deep learning applications...</p>
                <div class="metadata">
                    <span class="year">2023</span>
                    <span class="citations">87</span>
                    <a href="/papers/P002" class="link">Full Text</a>
                </div>
            </article>
            
            <article class="paper" data-id="P003">
                <h2 class="title">Natural Language Processing for Science</h2>
                <div class="authors">
                    <span class="author">Garcia, M.</span>
                    <span class="author">Lee, S.</span>
                    <span class="author">Brown, R.</span>
                </div>
                <p class="abstract">Exploring transformer models for scientific literature...</p>
                <div class="metadata">
                    <span class="year">2024</span>
                    <span class="citations">23</span>
                    <a href="/papers/P003" class="link">Full Text</a>
                </div>
            </article>
        </section>
        
        <aside class="stats">
            <table class="statistics">
                <tr><th>Year</th><th>Papers</th><th>Citations</th></tr>
                <tr><td>2024</td><td>156</td><td>1234</td></tr>
                <tr><td>2023</td><td>142</td><td>3456</td></tr>
                <tr><td>2022</td><td>128</td><td>5678</td></tr>
            </table>
        </aside>
    </main>
    
    <footer>
        <p>&copy; 2024 Journal of Computational Research</p>
    </footer>
</body>
</html>
"""


# =============================================================================
# DATA CLASS FOR PAPERS
# =============================================================================

@dataclass
class Paper:
    """Represents a research paper."""
    id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    citations: int
    link: str


# =============================================================================
# EXERCISE 1: Extract Page Title
# =============================================================================

def extract_page_title(html: str) -> str | None:
    """
    Extract the page title from HTML.
    
    Args:
        html: Raw HTML string
        
    Returns:
        Page title text or None if not found
        
    Example:
        >>> extract_page_title(SAMPLE_HTML)
        'Research Publications'
    """
    # TODO: Implement this function
    # 1. Parse HTML with BeautifulSoup
    # 2. Find the <title> tag
    # 3. Return its text content (stripped)
    pass


# =============================================================================
# EXERCISE 2: Extract All Links
# =============================================================================

def extract_all_links(html: str, base_url: str = '') -> list[dict[str, str]]:
    """
    Extract all links from HTML with their text and href.
    
    Args:
        html: Raw HTML string
        base_url: Base URL to prepend to relative links
        
    Returns:
        List of dicts with 'text' and 'href' keys
        
    Example:
        >>> links = extract_all_links(SAMPLE_HTML, 'https://journal.example.com')
        >>> links[0]
        {'text': 'About', 'href': 'https://journal.example.com/about'}
    """
    # TODO: Implement this function
    # 1. Parse HTML and find all <a> tags
    # 2. For each, extract text and href attribute
    # 3. Prepend base_url to relative hrefs (those starting with '/')
    pass


# =============================================================================
# EXERCISE 3: Extract Papers
# =============================================================================

def extract_papers(html: str) -> list[Paper]:
    """
    Extract all papers from the HTML.
    
    Parse each article.paper element and extract all fields
    into Paper dataclass instances.
    
    Args:
        html: Raw HTML string
        
    Returns:
        List of Paper objects
        
    Example:
        >>> papers = extract_papers(SAMPLE_HTML)
        >>> len(papers)
        3
        >>> papers[0].title
        'Machine Learning for Climate Modelling'
        >>> papers[0].authors
        ['Smith, J.', 'Chen, L.']
    """
    # TODO: Implement this function
    # 1. Parse HTML
    # 2. Find all article.paper elements
    # 3. For each, extract: id (data-id), title, authors, abstract, year, citations, link
    # 4. Convert year and citations to int
    # 5. Return list of Paper objects
    pass


# =============================================================================
# EXERCISE 4: Extract Table Data
# =============================================================================

def extract_table_data(html: str, table_class: str) -> list[dict[str, str]]:
    """
    Extract data from an HTML table into list of dictionaries.
    
    The first row (th elements) provides the keys for each dict.
    
    Args:
        html: Raw HTML string
        table_class: CSS class of the table to extract
        
    Returns:
        List of dicts, one per data row
        
    Example:
        >>> data = extract_table_data(SAMPLE_HTML, 'statistics')
        >>> data[0]
        {'Year': '2024', 'Papers': '156', 'Citations': '1234'}
    """
    # TODO: Implement this function
    # 1. Find table with specified class
    # 2. Extract headers from th elements
    # 3. For each data row (tr with td), create dict mapping headers to values
    pass


# =============================================================================
# EXERCISE 5: Filter Papers by Criteria
# =============================================================================

def filter_papers(
    papers: list[Paper],
    min_citations: int | None = None,
    year: int | None = None,
    author_contains: str | None = None
) -> list[Paper]:
    """
    Filter papers based on criteria.
    
    All provided criteria must match (AND logic).
    
    Args:
        papers: List of Paper objects
        min_citations: Minimum citation count
        year: Exact year to match
        author_contains: Substring that must appear in at least one author
        
    Returns:
        Filtered list of papers
        
    Example:
        >>> papers = extract_papers(SAMPLE_HTML)
        >>> filtered = filter_papers(papers, min_citations=50)
        >>> len(filtered)
        1
        >>> filtered[0].title
        'Deep Learning in Genomics'
    """
    # TODO: Implement this function
    pass


# =============================================================================
# TESTS
# =============================================================================

def test_extract_page_title():
    """Test title extraction."""
    title = extract_page_title(SAMPLE_HTML)
    assert title == 'Research Publications'
    
    # No title
    assert extract_page_title('<html><body></body></html>') is None
    
    print('✓ extract_page_title tests passed')


def test_extract_all_links():
    """Test link extraction."""
    links = extract_all_links(SAMPLE_HTML, 'https://example.com')
    
    assert len(links) >= 6  # nav links + paper links
    
    # Check first nav link
    about_link = next(l for l in links if l['text'] == 'About')
    assert about_link['href'] == 'https://example.com/about'
    
    print('✓ extract_all_links tests passed')


def test_extract_papers():
    """Test paper extraction."""
    papers = extract_papers(SAMPLE_HTML)
    
    assert len(papers) == 3
    
    # First paper
    p1 = papers[0]
    assert p1.id == 'P001'
    assert p1.title == 'Machine Learning for Climate Modelling'
    assert p1.authors == ['Smith, J.', 'Chen, L.']
    assert p1.year == 2024
    assert p1.citations == 42
    
    # Second paper
    p2 = papers[1]
    assert len(p2.authors) == 1
    assert p2.citations == 87
    
    print('✓ extract_papers tests passed')


def test_extract_table_data():
    """Test table extraction."""
    data = extract_table_data(SAMPLE_HTML, 'statistics')
    
    assert len(data) == 3
    assert data[0] == {'Year': '2024', 'Papers': '156', 'Citations': '1234'}
    assert data[2]['Year'] == '2022'
    
    print('✓ extract_table_data tests passed')


def test_filter_papers():
    """Test paper filtering."""
    papers = extract_papers(SAMPLE_HTML)
    
    # By citations
    high_cited = filter_papers(papers, min_citations=50)
    assert len(high_cited) == 1
    assert high_cited[0].id == 'P002'
    
    # By year
    papers_2024 = filter_papers(papers, year=2024)
    assert len(papers_2024) == 2
    
    # By author
    garcia_papers = filter_papers(papers, author_contains='Garcia')
    assert len(garcia_papers) == 1
    
    # Combined
    combined = filter_papers(papers, year=2024, min_citations=30)
    assert len(combined) == 1
    
    print('✓ filter_papers tests passed')


if __name__ == '__main__':
    test_extract_page_title()
    test_extract_all_links()
    test_extract_papers()
    test_extract_table_data()
    test_filter_papers()
    print('\n✓ All tests passed!')
