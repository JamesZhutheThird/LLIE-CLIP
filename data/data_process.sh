#!/bin/bash

# list all the files from the subdirectories of the current directory
# and them copy them to the 'all' subdirectory with a prefix

# example
# source: ./difficult/234.png
# destination: all/difficult_234.png

# the prefix is the name of the subdirectory

# if all directory does exist, delete it

rm -r all
mkdir all

for i in `find . -type f -name "*.png"`; do
    cp $i all/`echo $i | sed 's/\.\///' | sed 's/\//_/'`
done