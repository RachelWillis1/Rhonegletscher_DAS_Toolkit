"""Script to edit AWS S3 folder and file lists.

Reads and edits csv files and saves json/txt files
Prepared and hardcoded for RhoneGlacier dataset.
For different datasets, change args in find()
"""

import csv
import json
import os

# output_folder = "./output/"

#
# folderlist
#
with open('./output/S3folderlist.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    rows = []
    for row in reader:
        # print(', '.join(row))
        rows.append((', '.join(row)))

folderlist=[]
for i in range (len(rows)):
    if rows[i].find('2020')>0:
        folderlist.append(rows[i][(rows[i].find('2020')):(rows[i].find('/'))])
        
##json
# folderlistjson = json.dumps(folderlist)

# with open("folderlist.json", "w") as outfile:
#     json.dump(folderlistjson, outfile)

# print(folderlistjson)

##txt
with open("./output/S3folderlist.txt", "w") as outfile:
    outfile.write("\n".join(folderlist))

os.remove('./output/S3folderlist.csv')

#
# filelist
#
with open('./output/S3filelist.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    rows = []
    for row in reader:
        # print(', '.join(row))
        rows.append((', '.join(row)))

filelist=[]
for i in range (len(rows)):
    if rows[i].find('new/2020')>0:
        if len(rows[i][(rows[i].find('new/2020')):6+(rows[i].find('z.hdf5'))]) >10:
            filelist.append(rows[i][(rows[i].find('new/2020')):6+(rows[i].find('z.hdf5'))])
    else:
        pass
        
##json       
# filelistjson = json.dumps(filelist)

# with open("filelist.json", "w") as outfile:
#     json.dump(filelistjson, outfile)

# filelistjson

#txt
with open("./output/S3filelist.txt", "w", encoding="utf-8") as outfile:
    outfile.write("\n".join(filelist))

os.remove('./output/S3filelist.csv')