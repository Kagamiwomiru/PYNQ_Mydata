# coding: utf-8
# アニメ顔検出
import os
import cv2
import datetime
import sys; sys.path.append('/home/kagamiwomiru/Tools/GetImages/ToolKit')
import ConvertTo32 as CT
import glob
from tqdm import tqdm #コマンドライン用
# from tqdm import tqdm_notebook as tqdm #jupyter notebook用
import sys
args = sys.argv

if len(args)!=2:
    print("生成するキャラ名を指定してください。")
    print("例:python3 AniFace.py KizunaAI")
    sys.exit()


# 特徴量ファイルをもとに分類器を作成
# https://github.com/nagadomi/lbpcascade_animeface 
classifier = cv2.CascadeClassifier('cascades/lbpcascade_animeface.xml')


#設定
# print("生成するキャラ名を入力:")
output_dir=args[1]


# ディレクトリを作成
output_root_dir = 'faces/'
if not os.path.exists(output_root_dir+output_dir):
    os.makedirs(output_root_dir+output_dir)

root_dir=("data/")
for sub_dir in glob.glob(root_dir+"*"):
    sys.stdout.write("\r処理中:"+sub_dir+"\n")
    for image in tqdm(glob.glob(sub_dir+"/*")):
        data=cv2.imread(image)
        # グレースケールで処理を高速化
        try:
            gray_image = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        except:
            print("失敗："+image)
            os.remove(image) #問題の画像を削除
        else:
            faces = classifier.detectMultiScale(gray_image)
            for i, (x,y,w,h) in enumerate(faces):
                # 顔を切り抜く
                face_image = data[y:y+h, x:x+w]
                now = datetime.datetime.now()
                output_path = os.path.join(output_root_dir+output_dir,'{0:%Y%m%d_%H%M%S}.jpg'.format(now))
                cv2.imwrite(output_path,face_image)

# 画像の加工

CT.convertTo32(output_root_dir+output_dir)

