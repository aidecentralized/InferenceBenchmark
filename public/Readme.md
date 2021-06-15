To get started, make sure that python3 and pip are up to date and installed.

To load your own dataset. Please follow the directions for setting up / understanding how to use your own dataset_utils.py

For more experienced developers, please see dataset_utils. You can use any object dataloader as long as __getitem__ returns a dictionary mapping matching those as seen in dataset. Then change the scheduler.py to reference your dataloader object rather than the, "Custom" dataloader object.

For Easy use, please follow the general format for chap to infer and use your dataset without having to change any of the scripts, but just the organization / naming structure of your dataset.

Each dataset will have a, "name" (this will later be used in the "dataset" variable). The dataset should be in a folder with this, "name". All other data related to this dataset will live inside this folder. Ex. "fairface"  

Within this folder there should include the following:

1. A csv file named, "<name of dataset>_label_train.csv". This will be used for training.

2. A csv file named, "<name of dataset>_label_val.csv". This will be used for validation data.

3. A folder named, "train". This will hold the input images used for training.

4. A folder named, "val". This will hold the input images used for validation.

Here's what the contents of each folder / file mentioned above should contain:

1. This file should have the first row representing the columns of the input dataset seperated by commas. This will be used for mapping inputs to their respective labels of interest. The first column should be named, "file". This will represent relative path relating each image to their respective labels. They must by .jpg extensions. (And be jpg files) Values used for labeling can hold any value except special characters, and must be consistent. (No typos, must match capitilation, etc.)

Example name: (with name, "fairface") fairface_label_train.csv

Example contents: (refrencing the fairface dataset) 

file,age,gender,race,service_test
train/9.jpg,10-19,Male,White,True
......

2. This file should have the first row representing the columns of the input dataset seperated by commas. This will be used for mapping inputs to their respective labels of interest. The first column should be named, "file". This will represent relative path relating each image to their respective labels. They must by .jpg extensions. (And be jpg files) Values used for labeling can hold any value except special characters, and must be consistent. (No typos, must match capitilation, etc.)

Example name: (with name, "fairface") fairface_label_val.csv

Example contents: (refrencing the fairface dataset)

file,age,gender,race,service_test
val/3.jpg,30-39,Male,White,True
......

3. This folder must include all referenced files from the csv being used for training. Each file is expected to be a jpeg file ending with a .jpg extension.

For example. Our folder should at least include an jpeg image named, "9.jpg"

4. This folder must include all referenced files from the csv being used for training. Each file is expected to be a jpeg file ending with a .jpg extension.

For example. Our folder should at least include an jpeg image named, "3.jpg"

--------------

Once your dataset is set up, make sure to run the following command (may very per device / cuda capabilities)

    $ sudo pip install -r requirements.txt

or

    $ sudo pip3 install -r requirements.txt

Then to train:

    $ sudo python chap --dataset <name of dataset> --dataset_path <path leading to, but not including name of dataset>


