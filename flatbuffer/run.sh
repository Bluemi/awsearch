#!/bin/bash

case "$1" in
	py)
		flatc --python -o py export.fbs
		;;
	ts)
		flatc --ts -o ts export.fbs
		;;
	all)
		flatc --python -o py export.fbs
		flatc --ts -o ts export.fbs
		;;
	*)
		echo "invalid option"
		;;
esac
