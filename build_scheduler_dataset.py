import sys

from cdc_priority.cli import main


if __name__ == "__main__":
    main(["scheduler-dataset", *sys.argv[1:]])
