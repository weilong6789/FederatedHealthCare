"""Flower ClientApp for Fed-Heart-Disease: one hospital's side of FedAvg."""
from collections import OrderedDict

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fltab.harness import (evaluate, make_model, set_seed, test_loaders,
                              train_loader, train_sgd)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    cfg = context.run_config
    lr, epochs, seed = float(cfg["lr"]), int(cfg["local-epochs"]), int(cfg["seed"])
    pid = int(context.node_config["partition-id"])          # hospital 0..3
    server_round = int(msg.content["config"].get("server-round", 0))

    shuffle_seed = seed * 1000 + server_round * 10 + pid
    set_seed(shuffle_seed)
    loader = train_loader(center=pid, seed=shuffle_seed)

    # Start from the global weights, train on this hospital's data, send back
    model = make_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    train_sgd(model, loader, lr, epochs)

    content = RecordDict({
        "arrays": ArrayRecord(OrderedDict(model.state_dict())),
        "metrics": MetricRecord({"num-examples": len(loader.dataset)}),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def client_evaluate(msg: Message, context: Context):
    """Not used (server evaluates centrally), kept so the app stays valid."""
    pid = int(context.node_config["partition-id"])
    model = make_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    loader = test_loaders()[str(pid)]
    metrics = MetricRecord({"accuracy": evaluate(model, loader),
                            "num-examples": len(loader.dataset)})
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
