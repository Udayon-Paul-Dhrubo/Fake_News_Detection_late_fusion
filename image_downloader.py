import argparse
import pandas as pd
import os
from tqdm import tqdm
import urllib.request
import urllib.error  # Import the error handling module
import numpy as np

parser = argparse.ArgumentParser(description='r/Fakeddit image downloader')
parser.add_argument('type', type=str, help='train, validate, or test')
parser.add_argument('--start_index', type=int, default=0,
                    help='Index to start downloading from (default: 0)')
args = parser.parse_args()

df = pd.read_csv(args.type, sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

# pbar = tqdm(total=len(df))
# Initialize the progress bar with the starting index
pbar = tqdm(total=len(df), initial=args.start_index)

if not os.path.exists("images"):
    os.makedirs("images")

for index, row in df.iterrows():

    if index < args.start_index:  # Adjust this index to the last downloaded image index
        continue  # Skip already downloaded images

    if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
        image_url = row["image_url"]
        try:
            # Use urlopen with a timeout
            # Set timeout to 10 seconds
            with urllib.request.urlopen(image_url, timeout=10) as response:
                content = response.read()

                # Save the content to a file
                with open("images/" + row["id"] + ".jpg", 'wb') as f:
                    f.write(content)

        except urllib.error.HTTPError as e:
            print(f"HTTP Error for {image_url}: {e.code} {e.reason}")
        except urllib.error.URLError as e:
            print(f"URL Error for {image_url}: {e.reason}")
        except Exception as e:
            print(f"Unexpected error for {image_url}: {e}")
    pbar.update(1)

print("done")