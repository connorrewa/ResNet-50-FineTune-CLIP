import os

def download_coco():
    # Define expected paths
    data_root = "./coco_data"
    os.makedirs(data_root, exist_ok=True)
    
    print("1. Please download the dataset from: https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3")
    print(f"2. Extract the contents into '{os.path.abspath(data_root)}'")
    print("3. Ensure your directory structure looks exactly like this:")
    print(f"""
    {data_root}/
    ├── train2014/              # Folder containing training images
    ├── val2014/                # Folder containing validation images
    ├── annotations/            # Folder containing json files
    │   ├── captions_train2014.json
    │   └── captions_val2014.json
    """)
    
    # Optional: If you have the kaggle CLI installed, you can uncomment this:
    # os.system(f"kaggle datasets download -d jeffaudi/coco-2014-dataset-for-yolov3 -p {data_root} --unzip")

if __name__ == "__main__":
    download_coco()