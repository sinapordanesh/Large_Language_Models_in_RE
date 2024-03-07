#!/usr/bin/env python
# coding: utf-8

# In[1]:

"""
For running this heavy job in the background and non interuptable, use the following command:

    $ nohup python git_repo_code_extractor.py > output.log 2>&1 &

"""

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Use tqdm.notebook for Jupyter Notebooks


# In[7]:


# Load the main dataset
# df = pd.read_csv("data/java_test_dataset.csv")
df = pd.read_csv("data/java_train_dataset.csv", low_memory=False)


# - consistent downloading process and stable connection, retry if it gets down
# - process stage representation and showing in terminal
# - thread for faster internet connection and communication 
# - error handeling and pass if a file or repository dosnt exist (try main instead of master if a file content doesnt count). 

# In[9]:


def get_requests_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def get_raw_content_url(github_url, file_path, branch='master'):
    parts = github_url.split('/')
    user, repo = parts[3], parts[4]
    raw_url = f'https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}'
    return raw_url

def download_file_content(args):
    url, focal_file_path, test_file_path, session = args
    results = {}
    for file_type, file_path in [('focal_class_code', focal_file_path), ('test_class_code', test_file_path)]:
        for branch in ['master', 'main']:  # Try with 'master' first, then 'main'
            raw_url = get_raw_content_url(url, file_path, branch)
            try:
                response = session.get(raw_url)
                response.raise_for_status()
                results[file_type] = response.text
                break  # Break if successfully got the content
            except requests.RequestException:
                pass  # Try the next branch
        if file_type not in results:  # If both attempts fail
            results[file_type] = None
    return results
"""
def download_files_concurrently(df):
    session = get_requests_session()
    with ThreadPoolExecutor(max_workers=50) as executor:
        # Update to include both focal and test class file paths
        tasks = [(row['url'], row['focal_class_file'], row['test_class_file'], session) for index, row in df.iterrows()]
        futures = {executor.submit(download_file_content, task): index for index, task in enumerate(tasks)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            result = future.result()
            index = futures[future]
            # Assign the results to the new columns
            df.at[index, 'focal_class_code'] = result['focal_class_code']
            df.at[index, 'test_class_code'] = result['test_class_code']
"""

def download_files_concurrently(df, chunksize=1000):
    session = get_requests_session()
    for start in tqdm(range(0, len(df), chunksize), desc="Chunks"):
        end = start + chunksize
        df_chunk = df.iloc[start:end]
        with ThreadPoolExecutor(max_workers=20) as executor:
            tasks = [(row['url'], row['focal_class_file'], row['test_class_file'], session) for index, row in df_chunk.iterrows()]
            futures = {executor.submit(download_file_content, task): index for index, task in enumerate(tasks)}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading", leave=False):
                result = future.result()
                index = futures[future] + start  # adjust index based on the start of the chunk
                # Assign the results to the new columns
                df_chunk.at[index, 'focal_class_code'] = result['focal_class_code']
                df_chunk.at[index, 'test_class_code'] = result['test_class_code']
        
        # Write chunk to CSV
        mode = 'a' if start > 0 else 'w'
        header = True if start == 0 else False
        df_chunk.to_csv("data/java_train_dataset_code.csv", mode=mode, header=header, index=False)


# In[10]:


# The new columns initialized
df['focal_class_code'] = None
df['test_class_code'] = None


# In[11]:


# Perform the downloading and updating
download_files_concurrently(df)


# In[12]:


# df.head()


# In[14]:


# print(df['focal_class_code'][100])
df.to_csv("data/java_train_dataset_code.csv", index=False)


# In[ ]:




