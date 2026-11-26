#!/usr/bin/env bash
PathToScript="$( dirname "$(readlink -f "$0")")"
OutputDir="$PathToScript/Output"
cd "$PathToScript"
if [ ! -e "$OutputDir" ]; then
	if ! mkdir "$OutputDir"; then echo "Error create output dir..."; fi
fi

gcc ../src/ChimbaBenchCPU.c -O3 -fopenmp -Wall -static -o "$OutputDir/ChimbaBenchCPU-FBSD-x86_64"

if [ $? -eq 0 ]; then
	echo "Complete!"
else
	echo "Error"
fi
