
# coding: utf-8

# # 画像を収集します。

# In[26]:


# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
import re
import urllib2
import os
import cookielib
import json
#from tqdm import tqdm_notebook as tqdm

from tqdm import tqdm #Pythonで実行する場合


# In[39]:


def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),'html.parser')


# In[40]:


query="Kizuna Ai"
# and_query="jpg"
# not_query="small large orig php"
and_query=""
not_query="gif"
query.decode("utf-8")
label="0"
print "検索ワード:"+query


# In[41]:


#検索ワード
query= query.split()
query='+'.join(query)

#AND検索
and_query= and_query.split()
and_query='+'.join(and_query)

#NOT検索
not_query= not_query.split()
not_query='-'.join(not_query)

url="https://www.google.co.in/search?q="+query+"+"+and_query+"-"+not_query+"&source=lnms&tbm=isch"
print url


# In[42]:


#add the directory for your image here
DIR="data"
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
}
soup = get_soup(url,header)


# In[110]:


ActualImages=[]# contains the link for Large original images, type of  image
for a in soup.find_all("div",{"class":"rg_meta"}):
    link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
    ActualImages.append((link,Type))

print   len(ActualImages),"枚の画像が見つかりました。"


# In[111]:


if not os.path.exists(DIR):
            os.mkdir(DIR)
DIR = os.path.join(DIR, query.split()[0])

if not os.path.exists(DIR):
            os.mkdir(DIR)


# In[112]:


###print images
print ("画像をダウンロード中...")
for i , (img , Type) in enumerate(tqdm(ActualImages)):
    try:
        req = urllib2.Request(img, headers={'User-Agent' : header})
        raw_img = urllib2.urlopen(req).read()

        cntr = len([i for i in os.listdir(DIR) if label in i]) + 1
#         print cntr
        if len(Type)==0:
            f = open(os.path.join(DIR , label + "_"+ str(cntr)+".jpg"), 'wb')
        else :
#             f = open(os.path.join(DIR , label + "_"+ str(cntr)+"."+Type), 'wb')
            f = open(os.path.join(DIR , label + "_"+ str(cntr)+".jpg"), 'wb')


        f.write(raw_img)
        f.close()
    except Exception as e:
        print "could not load : "+img
        print e


