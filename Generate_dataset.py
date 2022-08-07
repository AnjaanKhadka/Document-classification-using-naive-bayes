import fitz
from numpy import append
import pandas as pd
import os

def get_path():
    final_path = []
    
    path1 = "D:\\DataMiningLab\\classification\\Dataset\\AI"
    path2 = "D:\\DataMiningLab\\classification\\Dataset\\Web"
    
    # path1 = input("Enter the path for AI path")
    print("path regestered successfully")
    # path1 = input("Enter the path for Web path")
    print("path regestered successfully")
    final_path.append(path1)
    final_path.append(path2)
    return final_path
    
def get_final_dataframe(path,flag):
    df = pd.DataFrame(columns = ['text','labels'])
    content = []
    label = []
    for file in os.listdir(path):
            if file.endswith(".pdf"):
                doc = fitz.open(path+'/'+file)
                content_temp = ''
                for page in range(len(doc)):
                    content_temp = content_temp+doc[page].get_text()
                content.append(content_temp)
    df['text'] = content
    df['labels'] = flag
    
    return df

    
def get_contents_of_pdf(file_path):
    for path in file_path:
        if '\\AI' in path:
            df_ai = get_final_dataframe(path,1)
        if '\\Web' in path:
            df_web = get_final_dataframe(path,0)  
    df = df_ai.append(df_web) 
    # print(df)
    return df
    
def get_content(file_path):
    # df = pd.DataFrame(columns = ['text','labels'])
    df = get_contents_of_pdf(file_path)
    df.to_csv("tabular_dataset.csv")
    print(df)
    
     
    
def dataset_generate():
    file_path = get_path()
    print(file_path)
    dataset = get_content(file_path)

    
if __name__ == "__main__":
    dataset_generate()