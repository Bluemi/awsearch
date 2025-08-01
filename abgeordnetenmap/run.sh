#!/bin/bash

case "$1" in
	r)
		npm run dev
		;;
	*)
		echo "invalid choice"
		;;
esac
