from pathlib import Path
from PIL import Image


class LeafDataset():
    def __init__(self, df, images, transform, has_labels=True):
        images_dir = Path(images)
        self.paths = [images_dir / name for name in df["image_id"]]
        self.labels = (
            df["label"].to_numpy(dtype="int64", copy=True)
            if "label" in df.columns
            else None
        )    
        self.transform = transform

    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            
        return image, int(self.labels[index])