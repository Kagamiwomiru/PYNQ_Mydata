# coding: utf-8
# 配列に検索したい文字列を列挙するとPYNQ-BNN用のデータセットを作成します。

# 注意:検索ワードでAND検索を利用する場合は、スペースの代わりに”＋”を使ってください。


from tqdm import tqdm
import subprocess


print("検索ワードを指定してください。終わったらCtrl+dを押してください。")
print("AND検索の場合、スペースの代わりに'+'を使ってください。")
print("例:KizunaAI+youtuber")

queries=[]

while True:
    try:
        queries.append(input())
    except EOFError:
        break

print("画像をダウンロード中です。この処理には時間がかかります。")

for query in queries:
    res=subprocess.call(['python3 ./GetImage.py '+query] ,shell=True)

print("データセット用に加工しています。この処理には時間がかかります。")

subprocess.call(['python3 ./AniFace.py KizunaAI'],shell=True)


subprocess.call(['rm -rf ./data'],shell=True) #このスクリプトを実行すると、dataディレクトリを消します。
