#!/bin/bash

case "$1" in
	s)
		shift
		PYTHONPATH=src python3 src/summarize.py "$@"
		;;
	e)
		shift
		PYTHONPATH=src python3 src/encode.py "$@"
		;;
	*)
		echo "invalid option for run.sh"
		;;
esac
