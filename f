#!/usr/bin/env python3
from fed.cli import parse, main
from fed.tune import hptune


if __name__ == '__main__':
    args = parse()
    if args.raytune:
        hptune(main, args)
    else:
        main(args)
