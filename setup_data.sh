#!/bin/bash
# Unzips data
cd data
for f in *.zip
do
unzip "$f"
done
echo "Data unzipped!"
