#!/bin/bash

case "$1" in
	py)
		flatc --python -o py export.fbs
		;;
	ts)
		flatc --ts -o ts export.fbs
		;;
esac
