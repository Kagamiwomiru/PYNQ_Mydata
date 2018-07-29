# coding: utf-8
# 画像を収集します。
from bs4 import BeautifulSoup
import requests
import re
import urllib.request, urllib.error, urllib.parse
import os
import http.cookiejar
import json
from tqdm import tqdm
import sys
args = sys.argv

if len(args)<2:
    print("検索ワードを指定してください。")
    print("例:python3 GetImage.py Python")
    sys.exit()
if len(args)>2:
    print("AND検索を利用する場合は、スペースの代わりに'+'を使ってください。")
    print("例:python3 GetImage.py Python+入門")
    sys.exit()


def get_soup(url,header):
    return BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url,headers=header)),'html.parser')

query=args[1]

and_query=""
not_query="gif"
label="0"
print("検索ワード:"+query)

url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
print(url)

#add the directory for your image here
DIR="data"
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
}
soup = get_soup(url,header)

ActualImages=[]# contains the link for Large original images, type of  image
for a in soup.find_all("div",{"class":"rg_meta"}):
    link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
    ActualImages.append((link,Type))

print(len(ActualImages),"枚の画像が見つかりました。")

if not os.path.exists(DIR):
            os.mkdir(DIR)
DIR = os.path.join(DIR, query.split()[0])

if not os.path.exists(DIR):
            os.mkdir(DIR)

print ("画像をダウンロード中...")
for i , (img , Type) in enumerate(tqdm(ActualImages)):
    try:
        raw_img = urllib.request.urlopen(img).read()
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
        print("could not load : "+img)
        print(e)