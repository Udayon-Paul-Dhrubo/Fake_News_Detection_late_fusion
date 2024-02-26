import os
import torch
from PIL import Image, UnidentifiedImageError
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from transformers import BertTokenizer

class FakedditImageDataset(Dataset):
    """The Fakeddit image dataset class"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Read the CSV file
        initial_frame = pd.read_csv(csv_file, delimiter='\t')
        # Filter entries to include only those with existing images
        self.csv_frame = initial_frame[initial_frame['id'].apply(lambda x: os.path.exists(os.path.join(root_dir, x + '.jpg')))]
        # Debugging: Print the number of entries after filtering
        print(f"Number of entries after filtering in {csv_file}: {len(self.csv_frame)}")


    def __len__(self):
        return len(self.csv_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.csv_frame.iloc[idx]['id'] + '.jpg'
        img_path = os.path.join(self.root_dir, img_name)

        try:
          image = Image.open(img_path)

          if image.mode != 'RGB':
              image = image.convert('RGB')
          
          label = self.csv_frame.iloc[idx]['2_way_label']

          if self.transform:
              image = self.transform(image)

          return image, label
        
        except (UnidentifiedImageError, FileNotFoundError) as e:
            print(f"Warning: Skipping problematic image {img_path} due to error: {e}")
            return None  # You can return None or a placeholder image

class FakedditHybridDataset(FakedditImageDataset):
    """The text + image dataset class"""

    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__(csv_file, root_dir, transform)
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get text embedding
        sent = self.csv_frame.iloc[idx]['clean_title']
        bert_encoded_dict = self.bert_tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=120,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        bert_input_id = bert_encoded_dict['input_ids']
        bert_attention_mask = bert_encoded_dict['attention_mask']

        # Image processing
        img_name = self.csv_frame.iloc[idx]['id'] + '.jpg'
        img_path = os.path.join(self.root_dir, img_name)

        try: 
          image = Image.open(img_path).convert('RGB')  # Convert to RGB to ensure consistency

          label = self.csv_frame.iloc[idx]['2_way_label']

          if self.transform:
              image = self.transform(image)
        except (UnidentifiedImageError, FileNotFoundError) as e:
            print(f"Warning: Skipping problematic image {img_path} due to error: {e}")
            # Return a placeholder or None here, if necessary
            return None

        return {
            'bert_input_id': bert_input_id,
            'bert_attention_mask': bert_attention_mask,
            'image': image,
            'label': label
        }

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

# Example usage
if __name__ == "__main__":
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Initialize the dataset
    fake_data = FakedditHybridDataset(csv_file='../multimodal_only_samples/multimodal_test_public.tsv', root_dir= '../multimodal_only_samples/images', transform=data_transforms)
    dataset_size = len(fake_data)
    print(f"The dataset contains {dataset_size} images.")
    # Example of accessing a few samples
    for k in range(1):
        try:
            hybrid = fake_data[k]
            print("Embedding:", hybrid['bert_input_id'])
            print("Mask:", hybrid['bert_attention_mask'])
            print("Image size:", hybrid['image'].size())
            print("Label:", hybrid['label'])
        except Exception as e:
            print(f"Error loading sample {k}: {e}")
