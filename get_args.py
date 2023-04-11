import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
    )
    parser.add_argument(
        '--lr',
        default=1e-4,
        type=int,
    )
    parser.add_argument(
        '--input_size',
        default=28 * 28,
        type=int,
    )
    parser.add_argument(
        '--hidden_size',
        default=512,
        type=int,
    )
    parser.add_argument(
        '--num_class',
        default=10,
    )
    parser.add_argument(
        '--epochs',
        default=5,
        type=int,
    )
    parser.add_argument(
        '--use_log',
        default=True,
        type=bool,
    )

    args = parser.parse_args()
    return args
