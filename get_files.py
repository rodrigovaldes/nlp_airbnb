
import wget
import os
import time

url_listings = "http://data.insideairbnb.com/united-states/ny/new-york-city/2018-03-04/data/listings.csv.gz"
url_reviews = "http://data.insideairbnb.com/united-states/ny/new-york-city/2018-03-04/data/reviews.csv.gz"

directory = "data/"
if not os.path.exists(directory):
    os.makedirs(directory)

for url in [url_listings, url_reviews]:
    wget.download(url)
    filename = url.split("/")[-1]
    os.system("open {}".format(filename))
    time.sleep(5)
    os.system("rm {}".format(filename))
    os.rename(".".join(filename.split(".")[:-1]),
        "data/"+".".join(filename.split(".")[:-1]))
