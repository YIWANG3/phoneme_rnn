import sys
import getopt


def parse():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:", ["config="])
    except getopt.GetoptError as err:
        print(err)  # will print something like "option -a not recognized"
        sys.exit(2)
    config_name = None
    for o, a in opts:
        if o in ("-c", "--config"):
            config_name = a
        else:
            assert False, "unhandled option"
    return config_name
