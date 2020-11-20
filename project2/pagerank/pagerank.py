import os
import random
import re
import sys


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
    # Page rank
    page_rank = {}
    for x in corpus:
        page_rank[x] = 0

    # For the first time, choose a random page
    corpus_pages = []
    for x in corpus:
        corpus_pages.append(x)
    page = random.choice(corpus_pages)
    page_rank[page] = 1

    # List of pages to choose and their weight
    pages_to_choose = []
    pages_choose_weight = []

    # Transition model
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
        page = random.choices(pages_to_choose, weights=pages_choose_weight, k=1)[0]

        # Add 1 to the chosen page
        page_rank[page] += 1

        # Clear all collections
        trans_mod.clear()
        pages_to_choose.clear()
        pages_choose_weight.clear()
    
    # Summarize and return
    sample_rank = {}
    for page in page_rank:
        sample_rank[page] = page_rank[page] / sum(page_rank.values())
    return sample_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Each possible page i that links to page p
    incoming = {}
    for p in corpus:
        incoming[p] = set()
    for p in corpus:
        for i in corpus[p]:
            incoming[i].add(p)
    
    # Pages in corpus
    corpus_pages = []
    for x in corpus:
        corpus_pages.append(x)
    n = len(corpus_pages)

    # Page rank
    page_rank = {}
    for page in corpus_pages:
        page_rank[page] = 1 / n
    
    diff = float('inf')
    while diff > 0.001:
        diff = 0
        for page in corpus_pages:
            # Apply page rank formula
            p = (1 - damping_factor) / n
            for i in incoming[page]:
                if len(corpus[i]) > 0:
                    p += damping_factor * sum([page_rank[i] / len(corpus[i])])
                else:
                    # A page that has no links at all should be interpreted 
                    # as having one link for every page in the corpus (including itself)
                    p += 1 / n
            # Check whether the iteration is converging
            diff = max(diff, abs(page_rank[page] - p))
            # Update page_rank with the new p value
            page_rank[page] = p

    # Summarize and return
    iterate_rank = {}
    for page in page_rank:
        iterate_rank[page] = page_rank[page] / sum(page_rank.values())
    return iterate_rank


if __name__ == "__main__":
    main()
