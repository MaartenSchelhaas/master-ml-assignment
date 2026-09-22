"""Train the asymmetric economic-loss model and write its validation performance."""

from ml_assignment.data import load_raw, train_val_split

if __name__ == "__main__":
    X_trn, y_trn, X_test = load_raw()
    X_tr, X_val, y_tr, y_val = train_val_split(X_trn, y_trn)
    raise NotImplementedError("train the hand-rolled MLP to minimize the economic loss")
