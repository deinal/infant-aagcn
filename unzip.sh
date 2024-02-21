#!/bin/bash

unzip helsinki.zip -d data/raw/helsinki
cd data/raw/helsinki
for file in *.zip
do
    dir=$(basename "$file" .zip)
    mkdir -p "$dir"
    unzip -o "$file" -d "$dir"
    rm "$file"
done
cd ../../..
find data/raw/helsinki -name "*.json" -type f -delete
rm -r data/raw/helsinki/v92_2


unzip NEL_Pisa_parameters.zip -d data/raw/pisa
mv data/raw/pisa/Positions\ CSV\ files/*.csv data/raw/pisa/
rmdir data/raw/pisa/Positions\ CSV\ files/
rm -r data/raw/pisa/json\ FILES/

unzip metadata.zip -d metadata