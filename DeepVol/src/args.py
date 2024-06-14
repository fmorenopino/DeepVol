import argparse

def get_args():
    """
    Function for handling command line arguments
    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='WaveNet')

    # seed
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='random seed')

    # model
    parser.add_argument('--num-blocks', type=int, default=2, metavar='NUM_BLOCKS',
                        help='number of blocks')
    parser.add_argument('--num-layers', type=int, default=6, metavar='NUM_LAYERS',
                        help='number of layers')

    parser.add_argument('--ch-residual', type=int, default=64, metavar='CH_RESIDUAL',
                            help='residual channels')
    parser.add_argument('--ch-dilation', type=int, default=64, metavar='CH_DILATION',
                            help='dilation channels')
    parser.add_argument('--ch-skip', type=int, default=128, metavar='CH_SKIP',
                            help='skip channels')
    parser.add_argument('--ch-end', type=int, default=64, metavar='CH_END',
                            help='end channels')

    parser.add_argument('--kernel-size', type=int, default=3, metavar='KERNEL_SIZE',
                            help='number of blocks')

    # dataset
    parser.add_argument('--win-len', type=int, default=128, metavar='WIN_LEN',
                        help='window length')
    parser.add_argument('--out-len', type=int, default=1, metavar='OUT_LEN',
                        help='output length')
    parser.add_argument('--emb-dim', type=int, default=56, metavar='EMB_DIM',
                        help='embedding dimension')
    parser.add_argument('--save-dataset-heavy-format', type=bool, default=False, metavar='HEAVY_FORMAT',
                    help='save-dataset-heavy-format')
    parser.add_argument('--granularity', type=str, default="5min", metavar='GRANULARITY',
                    help='granularity') #1min, 5min, 15min, 60min
    parser.add_argument('--conditioning-range', type=int, default=1, metavar='CONDITIONING_RANGE', 
                        help='conditioning range') 
                        #max 5min=15


    # training
    parser.add_argument('--batch-size', type=int, default=491, metavar='BATCH_SIZE',
                        help='batch size')#256

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--optimizer', type=str, default="Adam", metavar='OPTIMIZER', 
                        help='optimizer') #ASGD, LBFGS, Adam
    parser.add_argument('--check-val-every-n-epoch', type=int, default=100, metavar='CHECK_VAL', 
                    help='check-val-every-n-epoch')
    parser.add_argument('--log-preds-every-n-epochs', type=int, default=2000, metavar='LOG_PREDS', 
                    help='log-preds-every-n-epochs')
    parser.add_argument('--n-epochs', type=int, default=50000, metavar='N_EPOCHS', 
                help='n-epochs')
    parser.add_argument('--patience', type=int, default=10000, metavar='PATIENCE', 
                help='patience')
    parser.add_argument('--loss-function', type=str, default="qlike", metavar='LOSS_FUNCTION', 
            help='loss-function')

    # Argument parsing
    return parser.parse_args()
