"""Train the asymmetric economic-loss model and write its validation performance."""

from ml_assignment.data import load_raw, train_val_split
from ml_assignment.losses import EconomicLoss
from ml_assignment.model import MLP
from ml_assignment.optim import SGD

if __name__ == "__main__":
    X_trn, y_trn, X_test = load_raw()
    X_tr, X_val, y_tr, y_val = train_val_split(X_trn, y_trn)

    # placeholder hyperparameters, replace once tuning picks real values
    model = MLP(
        input_dim=X_tr.shape[1],
        hidden_sizes=[16, 8],
        optimizer=SGD(lr=0.01),
        batch_size=256,
        n_epochs=200,
        loss=EconomicLoss(),
        seed=0,
    )
    history = model.fit(X_tr, y_tr, X_val, y_val)
    print(f"final val economic loss: {history['val_loss'][-1]}")