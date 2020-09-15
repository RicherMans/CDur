#!/usr/bin/env bash

OUTPUTPATH=${1:-"flists"}
DATA_FILE="DCASE_2017_evaluation_set_audio_files.zip"
DATA_BASEPATH="DCASE2017_eval/evaluation_set_formatted_audio_segments/"

mkdir -p $OUTPUTPATH

if [[ ! -e "$DATA_FILE" ]]; then
    wget -c https://dl.dropboxusercontent.com/s/bbgqfd47cudwe9y/DCASE_2017_evaluation_set_audio_files.zip
else
    echo "File already downloaded!"
fi

if [[ ! -d "$DATA_BASEPATH" ]]; then
    unzip -P DCASE_2017_evaluation_set -d "DCASE2017_eval" "${DATA_FILE}" 
    rm -v "${DATA_FILE}"
else
    echo "Data extracted!"
fi

#Produce filellists

find DCASE2017_eval/evaluation_set_formatted_audio_segments/ -type f -name "*wav" | awk -v pwd="$PWD" 'NR==0{print "filename"}{print pwd"/"$1}' > DCASE2017_eval/filelists.txt

#Produce labellists
wget -q https://raw.githubusercontent.com/ankitshah009/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/master/groundtruth_release/groundtruth_strong_label_evaluation_set.csv -O DCASE2017_eval/dcase2017_eval_strong.tsv

#Merge filelists and labels
awk -F['/\t'] 'NR==FNR{a[$NF]=$0}NR!=FNR{print a["Y"$1]"$"$2"$"$3"$"$4}' DCASE2017_eval/filelists.txt DCASE2017_eval/dcase2017_eval_strong.tsv > flists/dcase2017_eval_strong.tsv

#Remove the commas and other uncenssary 符号
python3 -c "import pandas as pd; df= pd.read_csv('flists/dcase2017_eval_strong.tsv',sep='$',header=None); df[3] = df[3].str.strip().str.replace(',','').str.lower().str.replace(' ','_'); df.columns= ['filename','onset','offset','event_label']; df.to_csv('flists/dcase2017_eval_strong.tsv',index=False,sep='\t');"

awk '{print $1}' flists/dcase2017_eval_strong.tsv |sort -u | awk '{print $1}' > flists/dcase2017_eval_weak.tsv 

echo "Finished processing labels, file can be found in flists/dcase2017_eval_strong.tsv and flists/dcase2017_eval_weak.tsv"


