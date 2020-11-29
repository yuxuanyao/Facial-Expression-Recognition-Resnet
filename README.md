#### Yet another Repo for our implementation of Facial Emotion Recognition using Resnet for APS360

This is to organize all the code that we've written for the model

Some folders have been added to .gitignore due to size constraints. Please create them when using the code.

Using the code:
- Create folders that have been gitignored, make sure the file structure is similar to the tree below
- Run the Data Augmentation Notebook in src to process data
    - Combines the Muxspace and FER datasets, preprocess, and augments images to make sure each class has 12k images exactly.
    - Removes the contempt and disgust classes
- Run the Resnet Model Training - No extraction Notebook to train Resnet

<pre>
The final file structure should look like this:
├───datasets
│   ├───FER_dataset
│   ├───KDEF_and_AKDEF
│   │   ├───KDEF
│   │   ├───KDEFmap
│   │   └───ReadThis
│   ├───CK+48
│   ├───CollectedData
│   └───MuxspaceDataset
│       ├───data
│       ├───images
│       └───test
├───ProcessedData
│   ├───combined
│   ├───Testing
├───src
│   ├───.ipynb_checkpoints
│   └───__pycache__
├───torch_checkpoints
└───training_curves
</pre>
