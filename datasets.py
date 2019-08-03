import torch
from torch.utils.data import Dataset 
import cv2
import numpy as np 
import pandas as pd

__all__ = ['VideoDataset', 'VideoLabelDataset']

class VideoDataset(Dataset):
    """ Video Dataset for loading video.
        It will output only path of video (neither video file path or video folder path). 
        However, you can load video as torch.Tensor (C x L x H x W).
        See below for an example of how to read video as torch.Tensor.
        Your video dataset can be image frames or video files.

    Args:
        csv_file (str): path fo csv file which store path of video file or video folder.
            the format of csv_file should like:
            
            # example_video_file.csv   (if the videos of dataset is saved as video file)

            path
            ~/path/to/video/file1.mp4
            ~/path/to/video/file2.mp4
            ~/path/to/video/file3.mp4
            ~/path/to/video/file4.mp4

            # example_video_folder.csv   (if the videos of dataset is saved as image frames)
            
            path
            ~/path/to/video/folder1/
            ~/path/to/video/folder2/
            ~/path/to/video/folder3/
            ~/path/to/video/folder4/

    Example:

        if the videos of dataset is saved as video file

        >>> import torch
        >>> from datasets import VideoDataset
        >>> import transforms
        >>> dataset = VideoDataset(
        >>>     "example_video_file.csv",
        >>>     transform = transforms.VideoFilePathToTensor()  # See more options at transforms.py
        >>> )
        >>> data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
        >>> for videos in data_loader:
        >>>     print(videos.size())

        if the video of dataset is saved as frames in video folder
        The tree like: (The names of the images are arranged in ascending order of frames)
        ~/path/to/video/folder1
        ├── frame-001.jpg
        ├── frame-002.jpg
        ├── frame-003.jpg
        └── frame-004.jpg

        >>> import torch
        >>> from datasets import VideoDataset
        >>> import transforms
        >>> dataset = VideoDataset(
        >>>     "example_video_folder.csv",
        >>>     transform = transforms.VideoFolderPathToTensor()  # See more options at transforms.py
        >>> )
        >>> data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
        >>> for videos in data_loader:
        >>>     print(videos.size())
    """
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform 

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """ get a video """
        video = self.dataframe.iloc[index].path
        if self.transform:
            video = self.transform(video)
        return video


class VideoLabelDataset(Dataset):
    """ Dataset Class for Loading Video.
        It will output path and label. However, you can load video as torch.Tensor (C x L x H x W).
        See below for an example of how to read video as torch.Tensor.
        You can load tensor from video file or video folder by using the same way as VideoDataset.

    Args:
        csv_file (str): path fo csv file which store path and label of video file (or video folder).
            the format of csv_file should like:
            
            path, label
            ~/path/to/video/file1.mp4, 0
            ~/path/to/video/file2.mp4, 1
            ~/path/to/video/file3.mp4, 0
            ~/path/to/video/file4.mp4, 2

    Example:
        >>> import torch
        >>> import transforms
        >>> dataset = VideoDataset(
        >>>     "example_video_file_with_label.csv",
        >>>     transform = transforms.VideoFilePathToTensor()  # See more options at transforms.py
        >>> )
        >>> data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
        >>> for videos, labels in data_loader:
        >>>     print(videos.size())
    """
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform 

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """ get a video and its label """
        video = self.dataframe.iloc[index].path
        label = self.dataframe.iloc[index].label 
        if self.transform:
            video = self.transform(video)
        return video, label


if __name__ == '__main__':
    import torchvision
    import PIL 
    import transforms

    # test for VideoDataset
    dataset = VideoDataset(
        './data/example_video_file.csv', 
    )
    path = dataset[0]
    print(path)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for video in test_loader:
        print(video)
    
    # test for VideoLabelDataset
    dataset = VideoLabelDataset(
        './data/example_video_file_with_label.csv', 
        transform=torchvision.transforms.Compose([
            transforms.VideoFilePathToTensor(max_len=50, fps=10, padding_mode='last'),
            transforms.VideoRandomCrop([512, 512]),
            transforms.VideoResize([256, 256]),
        ])   
    )
    video, label = dataset[0]
    print(video.size(), label)
    frame1 = torchvision.transforms.ToPILImage()(video[:, 29, :, :])
    frame2 = torchvision.transforms.ToPILImage()(video[:, 39, :, :])
    frame1.show()
    frame2.show()

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    for videos, labels in test_loader:
        print(videos.size(), label)

