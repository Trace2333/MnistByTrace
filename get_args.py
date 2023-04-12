import argparse


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--device_type',
        default='gpu',
        type=str,
        help="Select between cpu and gpu",
)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
    )
    parser.add_argument(
        '--test_batch_size',
        default=16,
        type=int,
    )
    parser.add_argument(
        '--lr',
        default=2e-3,
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
        '--num_classes',
        default=10,
    )
    parser.add_argument(
        '--epochs',
        default=20,
        type=int,
    )
    parser.add_argument(
        '--test_epochs',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--use_log',
        default=True,
        type=bool,
    )
    parser.add_argument(
        '--optim_type',
        default='SGD',
        type=str,
    )
    parser.add_argument(
        '--loss_type',
        default='CrossEntropyLoss',
        type=str,
    )
    parser.add_argument(
        '--warm_up',
        default=False,
        type=bool,
    )
    parser.add_argument(
        '--hf_dataset',
        default=False,
        type=bool,
        help="if use the huggingface datasets module for the fashion mnist dataset"
             ".Use manual built dataset is much faster than the hf,but it need prepare the data in advance",
    )

    args = parser.parse_args()
    return args
