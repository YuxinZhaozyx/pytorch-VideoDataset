# pytorch-VideoDataset
Tools for loading video dataset and transforms on video in pytorch. You can directly load video files without preprocessing.

## Requirements

+ pytorch
+ torchvision
+ numpy
+ python-opencv
+ PIL

## How to use

1. Place the files [datasets.py](./datasets.py) and [transforms.py](./transforms.py) at your project directory.

2. Create csv file to declare where your video data are. The format of your csv file should like:

   ```csv
   path
   ~/path/to/video/file1.mp4
   ~/path/to/video/file2.mp4
   ~/path/to/video/file3.mp4
   ~/path/to/video/file4.mp4
   ```

   if the videos of your dataset are saved as image in folders. The format of your csv file should like:

   ``` 
   path
   ~/path/to/video/folder1/
   ~/path/to/video/folder2/
   ~/path/to/video/folder3/
   ~/path/to/video/folder4/
   ```

3. Prepare video datasets and load video to `torch.Tensor`.

   ```python
   import torch
   import torchvision
   import datasets
   import transforms
   
   dataset = datasets.VideoDataset(
   	"./data/example_video_file.csv",
       transform=torchvision.transforms.Compose([
           transforms.VideoFilePathToTensor(max_len=50, fps=10, padding_mode='last'),
           transforms.VideoRandomCrop([512, 512]),
           transforms.VideoResize([256, 256]),
       ])
   )
   data_loader = torch.utils.data.DataLoader(dataset, batch_size = 2, shuffle = True)
   for videos in data_loader:
       print(videos.size())
   ```

   If the videos of your dataset are saved as image in folders. You can use `VideoFolderPathToTensor` transfoms rather than `VideoFilePathToTensor` .

   ```python
   import torch
   import torchvision
   import datasets
   import transforms
   
   dataset = datasets.VideoDataset(
   	"./data/example_video_folder.csv",
       transform=torchvision.transforms.Compose([
           transforms.VideoFolderPathToTensor(max_len=50, padding_mode='last'),
           transforms.VideoRandomCrop([512, 512]),
           transforms.VideoResize([256, 256]),
       ])
   )
   data_loader = torch.utils.data.DataLoader(dataset, batch_size = 2, shuffle = True)
   for videos in data_loader:
       print(videos.size())
   ```

4. You can use `VideoLabelDataset` to load both video and label.

   ```python
   import torch
   import torchvision
   import datasets
   import transforms
   
   dataset = datasets.VideoLabelDataset(
   	"./data/example_video_file_with_label.csv",
       transform=torchvision.transforms.Compose([
           transforms.VideoFilePathToTensor(max_len=50, fps=10, padding_mode='last'),
           transforms.VideoRandomCrop([512, 512]),
           transforms.VideoResize([256, 256]),
       ])
   )
   data_loader = torch.utils.data.DataLoader(dataset, batch_size = 2, shuffle = True)
   for videos, labels in data_loader:
       print(videos.size(), labels)
   ```

5. You can also customize your dataset. It's easy to create your own `CustomVideoDataset` class and reuse the transforms I provided to transform video path to `torch.Tensor` and do some preprocessing such as `VideoRandomCrop`. 

   

## Docs

### [datasets](./datasets.py)

+ **datasets.VideoDataset**

  Video Dataset for loading video. 

  It will output only path of video (neither video file path or video folder path). However, you can load video as torch.Tensor (C x L x H x W). See below for an example of how to read video as torch.Tensor. Your video dataset can be image frames or video files.

  + **Parameters**

    + **csv_file** (str): path fo csv file which store path of video file or video folder. The format of csv_file should like:

      ```csv
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
      ```

  + **Example**

    if the videos of dataset is saved as video file.

    ```python
    import torch
    from datasets import VideoDataset
    import transforms
    dataset = VideoDataset(
        "example_video_file.csv",
        transform = transforms.VideoFilePathToTensor()  # See more options at transforms.py
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
    for videos in data_loader:
    	print(videos.size())
    ```

    if the video of dataset is saved as frames in video folder. The tree like: (The names of the images are arranged in ascending order of frames)

    ```shell
    ~/path/to/video/folder1
    ├── frame-001.jpg
    ├── frame-002.jpg
    ├── frame-003.jpg
    └── frame-004.jpg
    ```

    ```python
    import torch
    from datasets import VideoDataset
    import transforms
    dataset = VideoDataset(
        "example_video_folder.csv",
        transform = transforms.VideoFolderPathToTensor()  # See more options at transforms.py
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
    for videos in data_loader:
    	print(videos.size())
    ```

+ **datasets.VideoLabelDataset**

  Dataset Class for Loading Video with label.

  It will output path and label. However, you can load video as torch.Tensor (C x L x H x W). See below for an example of how to read video as torch.Tensor.

  You can load tensor from video file or video folder by using the same way as VideoDataset.

  + **Parameters**

    + **csv_file** (str): path fo csv file which store path and label of video file (or video folder). The format of csv_file should like:

      ```csv
      path, label
      ~/path/to/video/file1.mp4, 0
      ~/path/to/video/file2.mp4, 1
      ~/path/to/video/file3.mp4, 0
      ~/path/to/video/file4.mp4, 2
      ```

  + **Example**

    ```python
    import torch
    import transforms
    dataset = VideoDataset(
        "example_video_file_with_label.csv",
        transform = transforms.VideoFilePathToTensor()  # See more options at transforms.py
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
    for videos, labels in data_loader:
        print(videos.size())
    ```

### [transforms](./transforms.py)

All transforms at here can be composed with `torchvision.transforms.Compose()`.

+ **transforms.VideoFilePathToTensor** 

  load video at given file path to torch.Tensor (C x L x H x W, C = 3). 

  + **Parameters**
    + **max_len** (int): Maximum output time depth (L <= max_len). Default is None. If it is set to None, it will output all frames. 
    + **fps** (int): sample frame per seconds. It must lower than or equal the origin video fps. Defaults to None. 
    + **padding_mode** (str): Type of padding. Default to None. Only available when max_len is not None.
      + None: won't padding, video length is variable.
      + 'zero': padding the rest empty frames to zeros.
      + 'last': padding the rest empty frames to the last frame.

+ **transforms.VideoFolderPathToTensor**

  load video at given folder path to torch.Tensor (C x L x H x W).

  + **Parameters**
    + **max_len** (int): Maximum output time depth (L <= max_len). Default is None. If it is set to None, it will output all frames. 
    + **padding_mode** (str): Type of padding. Default to None. Only available when max_len is not None.
      + None: won't padding, video length is variable.
      + 'zero': padding the rest empty frames to zeros.
      + 'last': padding the rest empty frames to the last frame.

  

+ **transforms.VideoResize**

  resize video tensor (C x L x H x W) to (C x L x h x w).

  + **Parameters**
    + **size** (sequence): Desired output size. size is a sequence like (H, W), output size will matched to this.
    + **interpolation** (int, optional): Desired interpolation. Default is `PIL.Image.BILINEAR`

+ **transforms.VideoRandomCrop**

  Crop the given Video Tensor (C x L x H x W) at a random location.

  + **Parameters**
    + **size** (sequence): Desired output size like (h, w).

+ **transforms.VideoCenterCrop**

  Crops the given video tensor (C x L x H x W) at the center.

  + **Parameters**
    + **size** (sequence): Desired output size of the crop like (h, w).

+ **transforms.VideoRandomHorizontalFlip**

  Horizontal flip the given video tensor (C x L x H x W) randomly with a given probability.

  + **Parameters**
    + **p** (float): probability of the video being flipped. Default value is 0.5.

+ **transforms.VideoRandomVerticalFlip**

  Vertical flip the given video tensor (C x L x H x W) randomly with a given probability.

  + **Parameters**
    + **p** (float): probability of the video being flipped. Default value is 0.5.

+ **transforms.VideoGrayscale**

  Convert video (C x L x H x W) to grayscale (C' x L x H x W, C' = 1 or 3)

  + **Parameters**
    + **num_output_channels** (int): (1 or 3) number of channels desired for output video.

