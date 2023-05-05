import argparse
import sys
from functools import reduce

import status
import scrapper
import processing
from app_preferences import AppPreferences

command_to_script = {
    "status": status.run,
    "scrapper": scrapper.run,
    "processing": processing.run,
}

commands_str = reduce(lambda acc, v: '%s\n\t%s' % (acc, v), [cmd for cmd in command_to_script.keys()], '\t')

TOP_LEVEL_USAGE_STR = "usage: brickutil <command> [<args>]\nCommand list:" + commands_str
UNKNOWN_CMD = "Unknown command: %s"
HEPL_FLAGS = ["-h", "--help"]

if __name__ == "__main__":
    # top level scrip selection
    if len(sys.argv) < 2 or sys.argv[1] in HEPL_FLAGS:
        print(TOP_LEVEL_USAGE_STR)
        exit()
    script_name = sys.argv[1]
    script = command_to_script.get(script_name)

    if script is None:
        print(UNKNOWN_CMD % script_name)
        exit()

    prefs = AppPreferences('data')
    # selected script processing
    script(sys.argv, script_name, prefs)
