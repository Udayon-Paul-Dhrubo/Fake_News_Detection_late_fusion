# Fake News Detection using ```Late Fusion Multimodal Deep Learning```

## Environment Setup
1. ensure you have ```CUDA``` and ```torch``` installed
2. install dependencies
```bash
conda create -n fake_news python=3.8
conda activate fake_news
pip install numpy tqdm pandas urllib3 transformers matplotlib scikit-learn
pip install numpyencoder --trusted-host files.pythonhosted.org
```

## Fakeddit Dataset Download
 - Download the v2.0 dataset from [here](https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm?usp=sharing) 
  
 - The `*.tsv` dataset files have an `image_url` column which contain the image urls. You can use the URLs to download the images. (Note: download image for ```multimodal_only_samples```)

   - For convenience, we have provided a script ```image_downloader.py``` which will download the images for you. Please follow the instructions if you would like to use the attached script.
   - Usage :  ```python image_downloader.py multimodal_only_samples/multimodal_train.tsv --start_index 0```

## Pre-Training
## Bert pre-training

- in ```BERT.py``` file, set up the dataset(```all_samples```) directory according to your local setup
```python
df_train = pd.read_csv('./all_samples/all_train.tsv',encoding='UTF-8',delimiter="\t")
df_val = pd.read_csv('./all_samples/all_validate.tsv',encoding='UTF-8',delimiter="\t")
df_test = pd.read_csv('./all_samples/all_test_public.tsv',encoding='UTF-8',delimiter="\t")

...

DATA_DIR = "../all_samples/"
```
- then run the code to pretrain
```bash
python BERT.py
```

## Resnet pre-training
- in ```resnet > main.py``` file, set up the dataset(```multimodal_only_samples```) directory according to your local setup
```python
csv_dir = "../multimodal_only_samples/"
img_dir = "../multimodal_only_samples/images/"
csv_fname = 'multimodal_train.tsv'
```
- then run the code to pretrain
```bash
cd resnet
python main.py
```


## Late-Fusion Multimodal Model Training
- first run ```late_fusion/convert_saved_model.py``` 
```bash
cd late_fusion
python convert_saved_model.py
```
- then set up the dataset(```multimodal_only_samples```) directory according to your local setup
```python
# Prepare datesets
csv_dir = "../multimodal_only_samples/"
img_dir = "../multimodal_only_samples/images/"
l_datatypes = ['train', 'validate', 'test']
# l_datatypes = ['train']
csv_fnames = {
    'train': 'multimodal_train.tsv',
    'validate': 'multimodal_validate.tsv',
    'test': 'multimodal_test_public.tsv'
}
```

- then run the code to train
```bash
python fusion.py
```
