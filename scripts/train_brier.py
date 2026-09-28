"""Train the Brier-score model and write its validation performance.

w_default = 1.0 makes the weighted objective collapse to the plain Brier score.
"""

from _common import run

if __name__ == "__main__":
    run("brier", w_default=1.0)
