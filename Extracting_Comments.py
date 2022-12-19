# For data extraction from youtube ie getting comments from youtube
from youtube_comment_scraper_python import *
import pandas as pd

link = input("Input links:") #link of the youtube video
saved = input("Output name:") #name of the csv file in which the comment,number of likes and time of the comment will be stored.
youtube.open(link)

response = youtube.video_comments()
all_data = []
for i in range(0, 5): #number of iterations to be made by the algo to scan the comments.Here (0,5) means that it will scroll the comment section 10 times.
    response = youtube.video_comments()
    data = response['body']
    all_data.extend(data)
df = pd.DataFrame(data)
df.to_csv(saved)
