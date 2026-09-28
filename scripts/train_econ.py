"""Train the asymmetric economic-loss model and write its validation performance.

w_default = 3.0 puts three times the weight on errors made on actual defaults.
"""

from _common import run

if __name__ == "__main__":
    run("econ", w_default=3.0)
