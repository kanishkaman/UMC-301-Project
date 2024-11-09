# Dataset

We have used the iWildCam-2022 Wildlife dataset for out species classification model.
This dataset contains approximately 260,000 camera trap images with species-level annotations derived from the [WCS Camera Traps](https://lila.science/datasets/wcscameratraps) dataset, with additional metadata and supplementary training data not included in the original WCS Camera Traps dataset.

This dataset was used for the [iWildCam 2022 competition](https://github.com/visipedia/iwildcam_comp) as part of the [FGVC9](https://sites.google.com/view/fgvc9) workshop at [CVPR 2022](https://cvpr2022.thecvf.com/). Original results can be found on [Kaggle](https://www.kaggle.com/c/iwildcam2022-fgvc9/overview).  The task for this competition was counting the number of individual animals in a sequence of images.

The dataset is also used in the [WILDS](https://wilds.stanford.edu/) benchmark set, for which the task for this dataset is species classification.  Updated results are available on the [WILDS leaderboard](https://wilds.stanford.edu/leaderboard/#iwildcam).

The dataset includes three data types: camera trap images, iNaturalist images.

### Camera trap images

Camera trap images were selected from the [WCS Camera Traps](https://lila.science/datasets/wcscameratraps) dataset.  The training set contains 201,399 images from 323 locations, and the test set contains 60,029 images from 91 locations. These 414 locations are spread across the globe. A location ID (`location`) is given for each image, and in some special cases where two cameras were set up by ecologists at the same location, we have provided a `sub_location` identifier. Camera traps operate with a motion trigger and, after motion is detected, the camera will take a sequence of photos (from 1 to 10 images depending on the camera). We provide a `seq_id` for each sequence, and the competition task was to count the number of individuals across each test sequence.  Species-level annotations are provided for each training image (see `metadata/iwildcam2022_train_annotations.json`).  The dataset also includes count annotations on 1780 of the 36,292 train sequences (see `metadata/train_sequence_counts.csv`).

# Data Pre-Processing
`main.ipynb` contains different function and methods used to extract the data from the provided annotations file. 

The annotation file `training_annotations.json` has the following structure:
```json
{
  "images" : [image],
  "categories" : [category],
  "annotations" : [annotation]
}

image {
  "id" : str,
  "width" : int,
  "height" : int,
  "file_name" : str,
  "rights_holder" : str,
  "location" : int,
  "sub_location" : int,
  "datetime" : datetime,
  "seq_id" : str,
  "seq_num_frames" : int,
  "frame_num" : int
}

category {
  "id" : int,
  "name" : str
}

annotation {
  "id" : str,
  "image_id" : str,
  "category_id" : int
}
```
