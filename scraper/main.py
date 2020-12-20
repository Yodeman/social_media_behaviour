import time
import re
import json
from bs4 import BeautifulSoup
from selenium_driver import driver


output_file_path = r"C:\Users\USER\Desktop\social\social_data\test\\"

def get_comments(url, id):
    try:
        driver.implicitly_wait(30)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "lxml")
        
        comm_list = []
        comments_list = soup.find_all('div', attrs={'class':"narrow"})
        for i,x in enumerate(comments_list):
            # remove blockquote tag and its content.
            if x.blockquote:
                x.blockquote.decompose()
            comm_list.append({"id":f"id-{i}", "comment":x.get_text(separator='\n')})
            
            # write the comment into json file.
            with open(output_file_path+"comments"+str(id)+".json", "w") as json_file:
                json.dump(comm_list, json_file, indent=0)
    except Exception as e:
        print(e, ">>>>>>>>>>>>>Exception>>>>>>>>>>>>>")

url = "https://www.nairaland.com/6308930/buhari-wont-succumb-threats-take/"
last_id = 0
for i in range(3):
    last_id+=1
    get_comments(url+str(i), last_id)