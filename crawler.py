#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from icrawler.builtin import GoogleImageCrawler
from pathlib import Path
from tqdm import tqdm

import cv2
import dlib
import os
from pathlib import Path
detector = dlib.get_frontal_face_detector()


# In[12]:


def save(img,name, bbox, width=224,height=224):
    
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    imgCrop = cv2.resize(imgCrop, (width, height))
    cv2.imwrite(name+".jpg", imgCrop)

    
def faces(picpath):
    
    frame =cv2.imread(picpath)
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    fit =20
    for counter,face in enumerate(faces):
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        try:
            save(frame,picpath+os.path.split(picpath)[1].split('.')[0]+"__"+str(counter),(x1-fit,y1-fit,x2+fit,y2+fit))
        except:
            print("Fail  "+ str(picpath))

            
def cropFolder(List_of_all_pic):
    
    for i in range(len(List_of_all_pic)):
        faces(str(List_of_all_pic[i]))
        
        
def deleteOrginalPicFromFoler(List_of_all_pic):
    
    for i in range(len(List_of_all_pic)):
        os.remove(List_of_all_pic[i])
        
        
def rename(List_of_all_pic):
    
    for i in range(len(List_of_all_pic)):
        os.rename(str(List_of_all_pic[i]), str(os.path.split(List_of_all_pic[i])[0]+ '/'+str(i)+'.jpg'))
        
        
def fetchdata(csv_file_path):
    mitFilter=True
    filters = dict(type='face',
                   date=((2019, 1, 1), (2021, 9, 6))) 
    howmany= 100
    names=pd.read_csv(csv_file_path)
    subset=names.Name
    
    path = Path('./goodnight')
    n=0   
    for keyword in tqdm(subset):
        n=n+1
        print(n)
        crawler = GoogleImageCrawler(
            parser_threads=6,
            downloader_threads=6,
            storage={'root_dir': 'goodnight/{}'.format(keyword)}
        )    
        
        if mitFilter==True:
            crawler.crawl(keyword=keyword, filters=filters,max_num=howmany, min_size=(500, 500))
        else:
            crawler.crawl(keyword=keyword, max_num=howmany, min_size=(500, 500))
            
        List_of_all_pic = list(Path(path/keyword).glob(r'**/*.*'))
        cropFolder(List_of_all_pic)
        deleteOrginalPicFromFoler(List_of_all_pic)
        List_of_all_pic = list(Path(path/keyword)
                       .glob(r'**/*.*'))
        rename(List_of_all_pic)



# In[ ]:
fetchdata("act.csv")


# In[33]:


# from pathlib import Path
# import os
# path = Path('./google4')
# datasetfolder = next(os.walk(path))[1]

# for folder in tqdm(datasetfolder):
    
#     List_of_all_pic = list(Path(path/folder)
#                        .glob(r'**/*.*'))
#     cropFolder(List_of_all_pic)
#     deleteOrginalPicFromFoler(List_of_all_pic)
#     List_of_all_pic = list(Path(path/folder)
#                        .glob(r'**/*.*'))
#     rename(List_of_all_pic)


# In[ ]:




