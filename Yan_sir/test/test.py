"""
Generalized ranking implementation using JSON Lines input.

Input files (JSONL, one JSON object per line):
  - authors.jsonl     : { researcher_id, index }
  - prizes.jsonl      : { prize_id, prestige, year, researcher_id }
  - citations.jsonl   : { year, citer_id, citee_id, cite_count }
  - collabs.jsonl     : { year, author_id, coauthor_id, paper_count }

Phases:
  1) Seed weights from awards (Equation 2) up to SEED_YEAR.
  2) Year-by-year propagate citations + collabs (Equation 6).

Designed to stream large datasets without loading entire files.
"""

import json
from collections import defaultdict

# --------------- CONFIGURATION ---------------

# Filenames
AUTHOR_FILE = "authors.jsonl"
PRIZE_FILE  = "prizes.jsonl"
CITE_FILE   = "citations.jsonl"
COLLAB_FILE = "collabs.jsonl"

# Seed cutoff year
SEED_YEAR = 2023
# Years to propagate (example: only 2024)
ALL_YEARS = [2024]

# Tuning parameters
ALPHA = 0.7
BETA  = 0.3

# --------------- GLOBALS ---------------

# researcher_id -> integer index
author_to_index = {}
# index -> researcher_id
index_to_author = {}

# Current and next weight vectors
W_current = []
W_next    = []

# prize prestige lookup and counts
prize_prestige      = {}
prize_winner_counts = defaultdict(int)

# --------------- UTILITY FUNCTIONS ---------------

def load_authors(path):
    """Load author index mapping and initialize weight arrays."""
    global W_current, W_next
    max_idx = -1
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            ridx = obj['researcher_id']
            idx  = obj['index']
            author_to_index[ridx] = idx
            index_to_author[idx] = ridx
            max_idx = max(max_idx, idx)
    N = max_idx + 1
    W_current = [0.0] * N
    W_next    = [0.0] * N

def seed_awards(path, seed_year):
    """Seed W_current from prizes up to seed_year (two-pass)."""
    # First pass: record prestige & count winners
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            year = rec['year']
            if year > seed_year:
                continue
            p = rec['prize_id']
            prestige = rec['prestige']
            prize_prestige[p] = prestige
            prize_winner_counts[(p, year)] += 1

    # Second pass: distribute prestige shares
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            year = rec['year']
            if year > seed_year:
                continue
            p = rec['prize_id']
            rid = rec['researcher_id']
            count = prize_winner_counts[(p, year)]
            if count == 0:
                continue
            share = prize_prestige[p] / count
            idx = author_to_index.get(rid)
            if idx is not None:
                W_current[idx] += share

def propagate_year(year):
    """One-step propagation for a given year using JSONL citation + collab."""
    N = len(W_current)
    # reset next weights
    for i in range(N):
        W_next[i] = 0.0

    # 1) tally total cites by j in this year
    total_cites = defaultdict(int)
    with open(CITE_FILE, 'r') as f:
        for line in f:
            rec = json.loads(line)
            if rec['year'] != year:
                continue
            total_cites[rec['citer_id']] += rec['cite_count']

    # 2) citation contributions
    with open(CITE_FILE, 'r') as f:
        for line in f:
            rec = json.loads(line)
            if rec['year'] != year:
                continue
            j = rec['citer_id']
            i = rec['citee_id']
            cij = rec['cite_count']
            idx_j = author_to_index.get(j)
            idx_i = author_to_index.get(i)
            if idx_j is None or idx_i is None:
                continue
            wj = W_current[idx_j]
            if wj == 0.0:
                continue
            tc = total_cites[j]
            cite_frac = cij / tc if tc > 0 else 0.0
            W_next[idx_i] += wj * ALPHA * cite_frac

    # 3) tally total papers by j in this year
    total_pubs = defaultdict(int)
    with open(COLLAB_FILE, 'r') as f:
        for line in f:
            rec = json.loads(line)
            if rec['year'] != year:
                continue
            total_pubs[rec['author_id']] += rec['paper_count']

    # 4) collaboration contributions
    with open(COLLAB_FILE, 'r') as f:
        for line in f:
            rec = json.loads(line)
            if rec['year'] != year:
                continue
            j = rec['author_id']
            i = rec['coauthor_id']
            kij = rec['paper_count']
            idx_j = author_to_index.get(j)
            idx_i = author_to_index.get(i)
            if idx_j is None or idx_i is None:
                continue
            wj = W_current[idx_j]
            if wj == 0.0:
                continue
            tp = total_pubs[j]
            collab_frac = kij / tp if tp > 0 else 0.0
            W_next[idx_i] += wj * BETA * collab_frac

    # swap buffers
    for i in range(N):
        W_current[i] = W_next[i]

def main():
    # 1) load authors and init weights
    load_authors(AUTHOR_FILE)
    print(f"Loaded {len(author_to_index)} authors.")

    # 2) seed from awards
    seed_awards(PRIZE_FILE, SEED_YEAR)
    print("Seeding from awards complete.")

    # 3) propagate for each year
    for yr in ALL_YEARS:
        print(f"Propagating year {yr} ...")
        propagate_year(yr)
    print("Propagation complete.")

    # 4) output final ranks
    out = []
    for idx, score in enumerate(W_current):
        out.append({"researcher_id": index_to_author[idx], "rank": score})

    print("Final ranks:")
    for rec in out:
        print(rec)

if __name__ == "__main__":
    main()