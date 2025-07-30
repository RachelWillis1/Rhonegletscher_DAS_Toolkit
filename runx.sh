#!/bin/sh
startall=$(date +%s)

echo ----
echo Preparing input/output folders
# prepare input folders:
mkdir -p input input/DAS_data output output/event_png
echo - done

echo ----
echo Preparing input data filelist
# check folders and files in S3:
aws s3 ls --summarize --human-readable s3://rhoneglacier-eth/new/ > ./output/S3folderlist.csv
aws s3 ls --summarize --human-readable --recursive s3://rhoneglacier-eth/new/ > ./output/S3filelist.csv

aws s3 cp s3://csm-luna/RG-output/codes/S3foldefileliststxt.py ./output/
python3 ./output/S3foldefileliststxt.py
rm -I output/S3foldefileliststxt.py
echo - done

echo ----
echo Selecting input data filelist
# USER: time - from to (from mintime to maxtime)
echo Define time range for input data:
echo minimum time: 20200706_071820.361
echo maximum time: 20200804_001520.542
#read -p 'Input your start time: ' starttime
#read -p 'Input your end time: ' endtime
# if running with nohup define starttime and endtime
starttime=20200707_174900.000
endtime=20200707_234900.000

# Convert endtime to a parsable format (YYYY-MM-DD HH:MM:SS.mmm)
endtime_formatted=$(echo "$endtime" | sed -E 's/(.{4})(.{2})(.{2})_(.{2})(.{2})(.{2})\.(.{3})/\1-\2-\3 \4:\5:\6.\7/')

# Subtract 2 minutes from the endtime
endtime_adj=$(date -d "$endtime_formatted UTC -2 minutes" +"%Y%m%d_%H%M%S.%3N" 2>/dev/null)

echo Your selected time range is: $starttime - $endtime_adj

# automatic py script creating filelist in the spec interval
export starttime
export endtime

mv output/S3folderlist.txt output/folderlist.txt
mv output/S3filelist.txt output/filelist.txt

aws s3 cp s3://csm-luna/RG-output/codes/S3filelist_selectiontxt.py ./output/
python3 ./output/S3filelist_selectiontxt.py
rm -I output/S3filelist_selectiontxt.py
echo - done

echo -----
echo START Processing of selected files
# 5120MB = 5GB # USER-specified or automatically check EBS
MINAVAIL="1"  #5000

LEN_FILELIST=$(cat ./output/S3selectedfilelist.txt |wc -l)
I_FILE="0"

for path in $(cat ./output/S3selectedfilelist.txt);
do
        #cp data
        aws s3 cp s3://rhoneglacier-eth/$path ./input/DAS_data/
        I_FILE=$((I_FILE +1))

        # check the available space:
        AVAIL=$(df -m --output=avail / | grep -v Avail)

     if (( $(echo "$AVAIL < $MINAVAIL" |bc) )); then

            echo "/input/DAS_data is full and ready for processing"

            python3 scripts-main/feature_extraction_and_clustering_wDASK.py

            # delete data from /input/DAS_data:
            rm -f input/DAS_data/idas*

        elif (( $(echo "$I_FILE == $LEN_FILELIST" |bc) )); then
            echo "last file in the list loaded!"

            python3 scripts-main/feature_extraction_and_clustering_wDASK.py

            # delete data from /input/DAS_data:
            rm -f input/DAS_data/idas*
      fi
done

# Transfer output files and clean up
echo "----"
echo "Transferring output files to S3 and cleaning up"

# Define S3 destination bucket
S3_BUCKET="s3://csm-luna/RG-output/Rachels_Outputs/"

# Rename the output files with the adjusted endtime
mv output/rf_pred.npy output/rf_pred_${starttime}_${endtime_adj}.npy 2>/dev/null
mv output/meta_data_coherency.npy output/meta_data_coherency_${starttime}_${endtime_adj}.npy 2>/dev/null
mv nohup.out nohup_${starttime}_${endtime_adj}.out

# Copy files to S3
aws s3 cp output/rf_pred_${starttime}_${endtime_adj}.npy $S3_BUCKET
aws s3 cp output/meta_data_coherency_${starttime}_${endtime_adj}.npy $S3_BUCKET
aws s3 cp nohup_${starttime}_${endtime_adj}.out $S3_BUCKET

echo "- DONE - Processing completed for time range: $starttime - $endtime_adj"
echo " Output files successfully uploaded to S3: $S3_BUCKET"

# Cleanup processed files
rm -rf input

endall=$(date +%s)

