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
	c)
		shift
		PYTHONPATH=src python3 src/cluster.py "$@"
		;;
	d)
		shift
		PYTHONPATH=src python3 src/dimension_reduction.py "$@"
		;;
	v)
		shift
		PYTHONPATH=src python3 src/visualize.py "$@"
		;;
	*)
		echo "invalid option for run.sh"
		;;
esac
