"""Script to create file list for selected time range.

Prepared and hardcoded for RhoneGlacier dataset.
For different datasets, changes might be needed.

"""
from datetime import datetime
import os

starttime = (os.environ["starttime"])
endtime = (os.environ["endtime"])

stime=(datetime.strptime(str(starttime), '%Y%m%d_%H%M%S.%f'))
etime=(datetime.strptime(str(endtime), '%Y%m%d_%H%M%S.%f'))

with open('./output/filelist.txt','r') as f:
    file = f.read().splitlines()
f.close()

selected_filelist=[]
for i in range(len(file)):
    filetime = (file[i][4+file[i].find('UTC'):23+file[i].find('UTC'):])
    time = (datetime.strptime(filetime, '%Y%m%d_%H%M%S.%f'))
    
    if time < etime and time > stime:
        selected_filelist.append(file[i])

##txt
with open("./output/selectedfilelist.txt", "w", encoding="utf-8") as outfile:
    outfile.write("\n".join(selected_filelist))

