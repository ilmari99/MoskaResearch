#!/bin/bash
# Must be run in the base directory
find ./Logs/ -type f \! -name "*($1)*" -delete