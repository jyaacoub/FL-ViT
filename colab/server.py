from typing import Dict
import flwr as fl
import multiprocessing as mp
import argparse
from flower_helpers import Net, FedAvgMp, get_weights, test

def get_eval_fn():
    """Get the evaluation function for server side.

    Returns
    -------
    evaluate
        The evaluation function
    """

    def evaluate(server_round: int, params: fl.common.NDArrays,
             config: Dict[str, fl.common.Scalar]):
        """Evaluation function for server side."""

        # Prepare multiprocess
        manager = mp.Manager()
        # We receive the results through a shared dictionary
        return_dict = manager.dict()
        # Create the process
        p = mp.Process(target=test, args=(params, return_dict))
        # Start the process
        p.start()
        # Wait for it to end
        p.join()
        # Close it
        try:
            p.close()
        except ValueError as e:
            print(f"Coudln't close the evaluating process: {e}")
        # Get the return values
        loss = return_dict["loss"]
        accuracy = return_dict["accuracy"]
        # Del everything related to multiprocessing
        del (manager, return_dict, p)
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", type=int, default=3, help="Number of rounds for the federated training"
    )
    parser.add_argument(
        "-fc",
        type=int,
        default=2,
        help="Min fit clients, min number of clients to be sampled next round",
    )
    parser.add_argument(
        "-ac",
        type=int,
        default=2,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )
    args = parser.parse_args()
    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)
    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn")
    
    # Create a new fresh model to initialize parameters
    net = Net()
    init_weights = get_weights(net)
    # Convert the weights (np.ndarray) to parameters (bytes)
    init_param = fl.common.ndarrays_to_parameters(init_weights)
    # del the net as we don't need it anymore
    del net
    
    # Define the strategy
    strategy = FedAvgMp(
        fraction_fit=float(fc / ac),
        min_fit_clients=fc,
        min_available_clients=ac,
        evaluate_fn=get_eval_fn(),
        initial_parameters=init_param,
    )
    fl.server.start_server(server_address="127.0.0.1:5000",
                           config=fl.server.ServerConfig(num_rounds=rounds),
                           strategy=strategy,
                           num_clients=5)

