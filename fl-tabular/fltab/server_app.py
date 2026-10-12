"""Flower ServerApp for Fed-Heart-Disease. FedAvg, evaluated centrally."""
from collections import OrderedDict
from logging import WARNING

from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.common.logger import log as flwr_log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fltab.harness import (NUM_CLIENTS, array_bytes, evaluate_all, log, make_model,
                              set_seed)

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    seed, lr = int(cfg["seed"]), float(cfg["lr"])
    epochs, rounds = int(cfg["local-epochs"]), int(cfg["num-server-rounds"])

    # One simulated client per hospital. Extra clients get a partition-id with
    # no hospital behind it and crash every round (FedAvg just skips them).
    n_nodes = len(list(grid.get_node_ids()))
    if n_nodes != NUM_CLIENTS:
        flwr_log(WARNING, "Simulation has %s clients but there are %s hospitals. "
                 "Fix: flwr federation simulation-config --num-supernodes %s",
                 n_nodes, NUM_CLIENTS, NUM_CLIENTS)

    set_seed(seed)                       # same initial model as the baselines
    model = make_model()
    init = OrderedDict(model.state_dict())
    upload_per_client = array_bytes(init)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        m = make_model()
        m.load_state_dict(arrays.to_torch_state_dict())
        accs = evaluate_all(m)
        log("fedavg", seed, accs, round_=server_round,
            bytes_uploaded=server_round * NUM_CLIENTS * upload_per_client,
            lr=lr, local_epochs=epochs)
        return MetricRecord({f"acc_{k}": v for k, v in accs.items()})

    # All 4 hospitals train every round; no client-side evaluation.
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=0.0)
    strategy.start(grid=grid, initial_arrays=ArrayRecord(init),
                   num_rounds=rounds, evaluate_fn=global_evaluate)
