from torchvision.datasets.folder import ImageFolder, default_loader, IMG_EXTENSIONS 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder

class ImageAndPathDataset(ImageFolder):

    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
    ):
        super(ImageFolder, self).__init__(root=root,
                                          loader=default_loader,
                                          transform=transform,
                                          extensions=IMG_EXTENSIONS,
                                          target_transform=target_transform)

    def __getitem__(self, index):

        categories = [['action', 'animation', 'comedy', 'documentary',
                'drama', 'fantasy', 'horror', 'romance',
                'science fiction', 'thriller'],[0,1,2,3,4,5,6,7,8,9]]

        path, _ = self.samples[index]
        sample = self.loader(path)

        if path.split('/')[-2] in categories[0]:
            label = categories[1][categories[0].index(path.split('/')[-2])]
        else:
            label = 10

        
        if self.target_transform is not None:
            target = self.target_transform(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, 

class ImageAndPathDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(ImageAndPathDataLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        return ImageAndPathDataLoaderIter(self)

    

