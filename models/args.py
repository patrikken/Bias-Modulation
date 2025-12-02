from configargparse import Parser
from lightning import Trainer


def get_args():
    parser = Parser()
    parser.add(
        "-c",
        "--config",
        is_config_file=True,
        help="config file path",
        default="configs/default.yaml",
    )
    parser = add_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    return args


def add_args(parser):
    # Data parameters
    parser.add("--dataset", type=str, default="adult", help="Dataset to use")
    parser.add("--sensitive_attr_name", type=str, default="")
    parser.add("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add(
        "--data_dir", type=str, default="./data", help="Directory for storing data"
    )
    parser.add(
        "--output_dir", type=str, default="./outputs", help="Directory for outputs"
    )
    parser.add("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add("model_name", type=str, default="mlp", help="Model architecture to use")
    parser.add(
        "hidden_layers",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer sizes",
    )
    parser.add(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer"
    )
    parser.add("--dropout", type=float, default=0.5, help="Dropout rate for model")
    parser.add(
        "early_stopping_patience", type=int, default=10, help="Early stopping patience"
    )

    return parser
