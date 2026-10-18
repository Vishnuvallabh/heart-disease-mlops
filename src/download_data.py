import os
import urllib.request

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")
OUT = "data/raw/processed.cleveland.data"

os.makedirs("data/raw", exist_ok=True)
urllib.request.urlretrieve(URL, OUT)
print("Downloaded to", OUT)
