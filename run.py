import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Process some integers.')

    # mode
    parser.add_argument('--train', dest='train', action='store_true', help='enable training mode (default)')
    parser.add_argument('--no-train', dest='train', action='store_false', help='disable training mode')
    parser.set_defaults(train=True)
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='enable evaluating mode (default)')
    parser.add_argument('--no-evaluate', dest='evaluate', action='store_false', help='disable evluating mode')
    parser.set_defaults(evaluate=True)

    # cache
    parser.add_argument('-m', '--model', default="rffm", type=str, help='using which model: "rffm", "linear" (default: "rffm")')
    parser.add_argument('-p', '--model-dir', default=None, type=str, help='path for caching model (default: None)')

    # train
    parser.add_argument('-b', '--batch', default=2048, type=int, help='batch size (default: 2048)')
    parser.add_argument('-e', '--epoch', default=2, type=int, help='epoch (default: 2)')
    parser.add_argument('-w', '--window-size', default=23, type=int, help='window size (default: 23)')

    # Optimizer
    parser.add_argument('-l', '--learning-rate', default=0.001, type=float, help='Adam Optimizer learning rate (default: 0.001)')

    # RFFM
    parser.add_argument('-d', '--dimension', default=31740, type=int, help='Ramdom Fouier Features Mapper output dimension (default: 31740)')
    parser.add_argument('-s', '--stddev', default=1.0, type=int, help='Ramdom Fouier Features Mapper standard deviation distribution (default: 1.0)')

    # Output
    parser.add_argument('-o', '--output', default="a.out", type=str, help='Output file')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()

    import main
    main.main(args)

