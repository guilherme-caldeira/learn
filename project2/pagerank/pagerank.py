import os
import random
import re
import sys

import pandas as pd
import numpy as np
from collections import Counter

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_dist = {}

    number_pages_corpus = len(corpus)
    number_links = len(corpus[page])
    
    for pagex in corpus:
        if pagex in corpus[page]:
            prob = (damping_factor / number_links) + ((1 - damping_factor) / number_pages_corpus)
        else:
            prob = (1 - damping_factor) / number_pages_corpus
        prob_dist[pagex] = prob

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # List of pages
    pages = []

    # For the first time, choose a random page
    corpus_page = []
    for x in corpus:
        corpus_page.append(x)
    page = random.choice(corpus_page)
    pages.append(page)

    # List of pages to choose and their weight
    pages_to_choose = []
    pages_choose_weight = []

    trans_mod = {}

    # Repeat sampling N - 1 times because one page was already chosen
    for i in range(n - 1):
        # Get the transition model
        trans_mod = transition_model(corpus, page, damping_factor)

        # Loop through all items in the dictionary and add the key and the value to a list
        for x, y in trans_mod.items():
            pages_to_choose.append(x)
            pages_choose_weight.append(y)
        
        # Draw a new page based on the transition model
        new_page = random.choices(pages_to_choose, weights=pages_choose_weight, k=1)

        # Add the chosen page to the list with all pages chosen
        pages.append(new_page[0])

        # Clear all collections
        trans_mod.clear()
        pages_to_choose.clear()
        pages_choose_weight.clear()
    
    # Summarize
    final_score = {}
    final_score = Counter(pages)

    # Gets the proportional value
    for x, y in final_score.items():
        final_score[x] = y / n

    return final_score


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
