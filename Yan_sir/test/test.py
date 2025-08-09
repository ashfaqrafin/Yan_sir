"""
Generalized implementation of the researcher ranking algorithm
for large-scale datasets (millions of authors, hundreds of millions of edges).

Phases:
  1) Seed weights from awards (Equation 2).
  2) Incrementally propagate citations + collaboration year-by-year (Equation 6).

Assumes three CSV files (or other tabular source) with schema:

  prizes.csv:
    prize_id, prestige, year, researcher_id

  citations.csv:
    year, citer_id, citee_id, cite_count

  collabs.csv:
    year, author_id, coauthor_id, paper_count

And an authors.csv listing all researcher_id → numeric index mapping:
  researcher_id,index

All CSVs can be very large; we process them in streaming/chunks.
"""

import csv
from collections import defaultdict
import os

# --------------- CONFIGURATION ---------------

# Input data files (CSV)
PRIZE_FILE    = "prizes.csv"
CITE_FILE     = "citations.csv"
COLLAB_FILE   = "collabs.csv"
AUTHOR_INDEX  = "authors.csv"

# Cutoff year for seeding from awards
SEED_YEAR = 2023

# List of all years to propagate (inclusive)
ALL_YEARS = list(range(SEED_YEAR + 1, 2025))  # e.g. 2024, 2025

# Algorithm parameters
ALPHA = 0.7    # weight for citations
BETA  = 0.3    # weight for collaborations

# --------------- DATA STRUCTURES ---------------

# Map researcher_id -> small integer index [0..N-1]
author_to_index = {}
# Reverse map, if needed
index_to_author = {}

# Seed weights vector w[i] for i in [0..N-1]
W_current = []

# Temporary storage for new weights each year
W_next = []

# Prize prestige lookup: prize_id -> prestige value (float)
prize_prestige = {}

# For seeding, count winners per (prize_id, year)
prize_winner_counts = defaultdict(int)

# --------------- UTILITY FUNCTIONS ---------------

def load_author_index(path):
    """
    Reads authors.csv and fills author_to_index,
    and initializes global W_current to zeros.
    """
    global W_current, W_next
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for researcher_id, idx in reader:
            idx = int(idx)
            author_to_index[researcher_id] = idx
            index_to_author[idx] = researcher_id

    N = len(author_to_index)
    W_current = [0.0] * N
    W_next    = [0.0] * N


def seed_from_awards(prize_file, seed_year):
    """
    Implements Equation 2:
      w_i,T0 = sum_{p,τ≤T0} prestige_p * (count_i_wins(p,τ) / total_winners(p,τ))
    Streaming through prizes.csv once to accumulate counts.
    """
    # First pass: read prestige & count winners per (prize,year)
    with open(prize_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row['prize_id']
            year = int(row['year'])
            if year > seed_year:
                continue
            prestige = float(row['prestige'])
            prize_prestige[p] = prestige
            prize_winner_counts[(p, year)] += 1

    # Second pass: distribute prestige among winners
    with open(prize_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row['year'])
            if year > seed_year:
                continue
            p   = row['prize_id']
            pid = row['researcher_id']
            count = prize_winner_counts[(p, year)]
            if count == 0:
                continue
            prestige = prize_prestige[p]
            share = prestige / count

            idx = author_to_index.get(pid)
            if idx is not None:
                W_current[idx] += share


def propagate_year(year, cite_file, collab_file):
    """
    Implements one step of Equation 6 for a single year.
    Only reads citation and collaboration edges for that year,
    and updates W_next from W_current.
    """
    # Reset new weights
    N = len(W_current)
    for i in range(N):
        W_next[i] = 0.0

    # Process citations: we need total cites by j in that year
    total_cites = defaultdict(int)
    with open(cite_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['year']) != year:
                continue
            j = row['citer_id']
            cnt = int(row['cite_count'])
            total_cites[j] += cnt

    # First pass through citations.csv: citation contributions
    # We also combine collaboration edges in the same loop if possible
    with open(cite_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['year']) != year:
                continue
            j = row['citer_id']
            i = row['citee_id']
            cij = int(row['cite_count'])
            idx_j = author_to_index.get(j)
            idx_i = author_to_index.get(i)
            if idx_j is None or idx_i is None or W_current[idx_j] == 0.0:
                continue

            # citation fraction
            tc = total_cites[j]
            cite_frac = (cij / tc) if tc > 0 else 0.0

            # we'll fetch collaboration fraction below
            # accumulate partial vote
            W_next[idx_i] += W_current[idx_j] * ALPHA * cite_frac

    # Process collaborations: need total pubs by j in that year
    total_pubs = defaultdict(int)
    with open(collab_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['year']) != year:
                continue
            j = row['author_id']
            cnt = int(row['paper_count'])
            total_pubs[j] += cnt

    # Second pass through collabs.csv: collaboration contributions
    with open(collab_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['year']) != year:
                continue
            j = row['author_id']
            i = row['coauthor_id']
            kij = int(row['paper_count'])
            idx_j = author_to_index.get(j)
            idx_i = author_to_index.get(i)
            if idx_j is None or idx_i is None or W_current[idx_j] == 0.0:
                continue

            pub_total = total_pubs[j]
            collab_frac = (kij / pub_total) if pub_total > 0 else 0.0
            W_next[idx_i] += W_current[idx_j] * BETA * collab_frac

    # At end of year, swap W_next → W_current
    for i in range(N):
        W_current[i] = W_next[i]


# --------------- MAIN WORKFLOW ---------------

def main():
    # 1) Load author index
    load_author_index(AUTHOR_INDEX)
    print(f"Loaded {len(author_to_index)} authors.")

    # 2) Seed initial weights from awards up to SEED_YEAR
    seed_from_awards(PRIZE_FILE, SEED_YEAR)
    print("Seed weights from awards completed.")

    # 3) Propagate year-by-year
    for yr in ALL_YEARS:
        print(f"Propagating for year {yr} ...", end="")
        propagate_year(yr, CITE_FILE, COLLAB_FILE)
        print(" done.")

    # 4) Output final weights
    out_path = "researcher_rankings.csv"
    with open(out_path, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["researcher_id", "rank"])
        for idx, score in enumerate(W_current):
            writer.writerow([ index_to_author[idx], f"{score:.6f}" ])
    print(f"Final rankings written to {out_path}")


if __name__ == "__main__":
    main()