#!/usr/bin/env bash

OUTPUTPATH=${1:-"flists"}
DATA_FILE="URBAN-SED.tar.gz"
DATA_BASEPATH="URBAN-SED"

mkdir -p $OUTPUTPATH

if [[ ! -e "$DATA_FILE" ]]; then
    wget -c https://zenodo.org/record/1002874/files/URBAN-SED.tar.gz
else
    echo "File already downloaded!"
fi

if [[ ! -d "$DATA_BASEPATH" ]]; then
    tar xzvf "${DATA_FILE}"
    rm -v "${DATA_FILE}"
else
    echo "Data extracted!"
fi

#Produce filelists
for sub in train validate test; do
    output_strong="${OUTPUTPATH}/urban_sed_${sub}_strong.tsv"
    output_weak="${OUTPUTPATH}/urban_sed_${sub}_weak.tsv"
    echo "Producing $output_strong"
    echo -n ""> $output_strong; 
    for f in $(find URBAN-SED/annotations/$sub/ -type f -name "*txt"); do 
        wavpath=$(echo $f | sed -e 's/annotations/audio/' -e 's/txt/wav/')
        cat $f  | sed -e "s|^|$PWD\/$wavpath |g" >> $output_strong;  
    done; 
    sed -i '1ifilename onset offset event_label'  $output_strong; 
    sed -i 's/ \+/\t/g' $output_strong;

    echo "Producing weak labeled ${output_weak}"
    echo -n "" > $output_weak;
    for f in $(find URBAN-SED/annotations/$sub/ -type f -name "*txt"); do 
        wavpath=$(echo $f | sed -e 's/annotations/audio/' -e 's/txt/wav/')
        weak_labels=$(cat $f | awk -vORS=, '{print $NF}' |sed 's/,$//g')
        echo "$wavpath $weak_labels" >> $output_weak;
    done; 
    sed -i '1ifilename event_labels'  $output_weak; 
    sed -i 's/ \+/\t/g' $output_weak;


done
