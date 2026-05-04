import sys

from cdc_priority.cli import main


if __name__ == "__main__":
    main(["pipeline", *sys.argv[1:]])
