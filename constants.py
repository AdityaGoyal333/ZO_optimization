"""Knobs that are part of the experiment protocol, not of a single run.

Change them here so every job flushes and evaluates on the same schedule.
"""

# Rewrite metrics.pkl and checkpoint.pt after this many minibatch updates.
# A killed job keeps the last flush instead of an empty output directory.
FLUSH_EVERY = 100

# Run the fixed test set after this many minibatch updates.
# 1000 is divisible by FLUSH_EVERY, so each test score is in that flush.
EVAL_EVERY = 100

# Hugging Face Trainer's default generator seed. The DataLoader reshuffles
# the fixed training pool every epoch with this seed. It is not the seed
# that chooses which examples are in the pool.
DATALOADER_SEED = 42

# Drop an SVD component when its singular value is this fraction of the
# largest one. Gaussian factors are full rank far above this cutoff;
# the cutoff only removes numerical junk.
SINGULAR_VALUE_REL_TOL = 1e-6

# FZOO draws each step's perturbation seed from this range.
PERTURBATION_SEED_MODULUS = 1_000_000_000

# Longer inputs are truncated by the tokenizer. 512 is the RoBERTa/BERT limit.
MAX_SEQUENCE_LENGTH = 512
