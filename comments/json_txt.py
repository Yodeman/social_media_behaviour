# COnverts JSON files to text files.

import os
import json
   
def json_txt(input_file_dir):
    """
    converts json file into text file.
    """
    file_name = input_file_dir.split('\\')[-1]+"_comments.txt"    #get folder name.
    txt_output = open(file_name, 'w')
    with open(input_file_dir+"\\merged.json", 'r') as f:
        comments = json.load(f)
        for comment in comments:
            txt_output.write(comment["comment"]+'\n\n')
    txt_output.close()


if __name__ == "__main__":
    path = r'C:\Users\USER\Desktop\social\social_data\\'
    for folder in os.listdir(path):
        json_folder = path+folder
        if os.path.isdir(json_folder):
            json_txt(json_folder)
        
