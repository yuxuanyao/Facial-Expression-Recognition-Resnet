import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
import os
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt
import math
torch.manual_seed(1) # set the random seed

# Modify Muxspace labels to match FER
def modify_mux_labels():
    # Read legend csv as pandas dataframe
    legend = pd.read_csv('../datasets/MuxspaceDataset/data/legend.csv')
    # Change all capitalized emotions into lower case
    legend['emotion'] = legend['emotion'].str.lower()

    # Modify the legend to match FER legend
    # FER Categories: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    legend.loc[legend['emotion'] == 'anger', 'emotion'] =     0
    legend.loc[legend['emotion'] == 'disgust', 'emotion'] =   1
    legend.loc[legend['emotion'] == 'fear', 'emotion'] =      2
    legend.loc[legend['emotion'] == 'happiness', 'emotion'] = 3
    legend.loc[legend['emotion'] == 'sadness', 'emotion'] =   4
    legend.loc[legend['emotion'] == 'surprise', 'emotion'] =  5
    legend.loc[legend['emotion'] == 'neutral', 'emotion'] =   6

    # Export to csv if it doesn't exist already
    if(not path.exists('../datasets/MuxspaceDataset/data/new_legend.csv')):
        legend.to_csv('../datasets/MuxspaceDataset/data/new_legend.csv', index=False) 

# Combine FER and Muxspace dataset into one csv
def combine_mux_fer():
    # Read mux and fer datasets
    mux = pd.read_csv("../datasets/MuxspaceDataset/data/new_legend.csv", usecols=["emotion", "image"])
    fer = pd.read_csv("../datasets/FER_dataset/icml_face_data.csv", usecols=["emotion", " pixels"])

    combined = pd.DataFrame(columns=["emotion", "image"," pixels"])
    combined = combined.append(mux)
    combined = combined.append(fer)

    # Remove Disgust
    combined = combined[combined.emotion != 1]

    return combined

# Reads and resizes an image from the mux dataset to 48 x 48 and makes it grayscale
def read_and_resize_mux_image(img_path):   
    """
    Args:
        img_path: relative path to the mux image
        
    Returns: the resized image as numpy array
    """ 
    # Read and resize image with OpenCV 
    img_pixels = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) # cv2 reads in BGR, need to convert to grayscale
    
    # Resize to 48 x 48
    img_pixels = cv2.resize(img_pixels, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
    img_data = np.asarray(img_pixels)
    return img_data


# Reads and resizes an image from the FER dataset to 48 x 48
def read_and_resize_fer_image(img_pixels): 
    """
    Args:
        img_pixels: string containing the pixels of the FER image
        
    Returns: the resized image as numpy array
    """
    # Parse the pixel string
    img_string = img_pixels.split(' ')
    # Resize to 48 x 48
    img_data = np.asarray(img_string, dtype=np.int32).reshape(48, 48)
    return img_data

normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Augments image and returns all augmented images
def augment(img, num_imgs=3, normalize_tensors=False):
    """
    Args:
        img: resized image
        num_imgs: how many augmented images to return (3, )
    Returns: array of augmented images
    """

    # Normalize to between 0 and 1
    img = img/255   # Since values are in (0, 255)
    # Original Image and Add RGB channels
    img_tensor = torch.from_numpy(np.copy(img)).unsqueeze(0).repeat(3,1,1)
    # Rotate Image and add RGB channels
    img_rotated_left_tensor = torch.from_numpy(scipy.ndimage.rotate(np.copy(img), 5, order=1, reshape=False)).unsqueeze(0).repeat(3,1,1)
    # Flip image and add RGB channels
    img_flipped_tensor = (torch.from_numpy(np.fliplr(np.copy(img)).copy())).repeat(3,1,1)

    # Normalize to PyTorch ResNet Input requirement
    if normalize_tensors:
        img_tensor = normalize(img_tensor)
        img_rotated_left_tensor = normalize(img_rotated_left_tensor)
        img_flipped_tensor = normalize(img_flipped_tensor)

    return [img_tensor, img_rotated_left_tensor, img_flipped_tensor]


def plot_augmented_images(augmented_images):
    # Plotting
    fig = plt.figure()
    for i, image in enumerate(augmented_images):
        # Figure stuff
        fig.add_subplot(1,len(augmented_images),i+1)
        plt.axis('off')
        # Transpose for visualization
        image_transposed = np.asarray((image.transpose(0,1)).transpose(1,2))
        plt.imshow(image_transposed, cmap = plt.cm.gray) # Plot the gray scale image
        
        # Make sure the values of the tensors are between 0 and 1, size should be 3x48x48
        print(np.max(image_transposed))
        print(np.min(image_transposed))
        print(image.shape)

'''
Split dataset into train, validation, and test sets
'''

def split_dataset(combined):
    """
    Args:
        combined: combined dataset
        
    Returns: train, validate, and test sets as pandas dataframes
    """

    # Select rows of each emotion
    emotions = [combined[combined.emotion == 0], combined[combined.emotion == 1], combined[combined.emotion == 2], 
                combined[combined.emotion == 3], combined[combined.emotion == 4], combined[combined.emotion == 5],
                combined[combined.emotion == 6]]

    # Initialize pandas dataframe for training, validation, and testing
    training = pd.DataFrame(columns=["emotion"," pixels"])
    validation = pd.DataFrame(columns=["emotion"," pixels"])
    test = pd.DataFrame(columns=["emotion"," pixels"])

    # Split
    for emotion in emotions:
        sample = emotion.sample(frac=1, random_state=1).reset_index(drop=True)              # Shuffle the emotion df
        training = training.append(sample[:int(0.7*sample.shape[0])])                       # Append the first 70% to training
        valid_and_test = sample[int(0.7*sample.shape[0]):]                                  # Save the last 30% data for testing and validation
        validation = validation.append(valid_and_test[:int(0.5*valid_and_test.shape[0])])   # 15% for validation
        test = test.append(valid_and_test[int(0.5*valid_and_test.shape[0]):])               # 15% for testing

    return training, validation, test


'''
Augments data and flips, saves only augmented images
'''
def augment_and_save(df, subfolder, dataset, cutoff, count, normalize_tensors=False):
    """
    Args:
        df: dataframe
        subfolder: subfolder to save in (i.e. cutoff9100)
        dataset: name of dataset (train, validate, test)
        cutoff: total number of images we want for each class
        count: current count of each emotion, array
        normalize_tensors: whether to normalize tensors with mean and std given by pytorch
    """
    # Path to save processed tensor
    master_path = '../ProcessedData/' + subfolder + '/' + dataset + '/'
    
    # Create directory for master path
    if not os.path.isdir(master_path):
        os.mkdir(master_path)

    num_imgs = len(df.index)
    for i in range(num_imgs):

        img_pixels = df.iloc[i][" pixels"]
        
        # Muxspace dataset (pixels column is NaN which is of type float)
        if type(img_pixels) is float:
            # Read and resize image
            img_path = '../datasets/MuxspaceDataset/images/' + str(df.iloc[i]["image"])           
            img_data = read_and_resize_mux_image(img_path)
            
        # FER Dataset 
        else:
            img_data = read_and_resize_fer_image(img_pixels)
        
       # Augment: Add RGB, Flip, Rotate, Normalize,      Returns 3 tensors
        augmented_images = augment(img_data, normalize_tensors=normalize_tensors)
        emotion = df.iloc[i]["emotion"]
        
        # Folder for the specific emotion
        folder_name = master_path + str(emotion)
        
        # Create directory for emotion if it doesn't already exist
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        # Save if total images for emotion is less than total images
        if(count[emotion] < cutoff):
            # Save augmented images if class != 3 or 6
            if emotion != 3 and emotion != 6:
                torch.save(augmented_images[1], folder_name + '/' + str(i) + '_rotated.tensor')
                count[emotion] += 1

                # Check again
                if(count[emotion] < cutoff):
                    torch.save(augmented_images[2], folder_name + '/' + str(i) + '_flipped.tensor')
                    count[emotion] += 1

    print("Finished saving augmented images to " + master_path)
    print(count)



'''
Saves dataset without augmentation, will never cutoff original images
'''
def save_dataset_as_tensors(df, subfolder, dataset, cutoff, count, normalize_tensors=False):
    """
    Args:
        df: dataframe
        subfolder: subfolder to save in (i.e. cutoff9100)
        dataset: name of dataset (train, validate, test)
        cutoff: total number of images we want for each class
        count: current count of each emotion, array
        normalize_tensors: whether to normalize tensors with mean and std given by pytorch
    """

    # Path to save processed tensor
    master_path = '../ProcessedData/' + subfolder + '/' + dataset + '/'
    
    # Create directory for master path
    if not os.path.isdir(master_path):
        os.mkdir(master_path)

    num_imgs = len(df.index)
    for i in range(num_imgs):

        img_pixels = df.iloc[i][" pixels"]
        
        # Muxspace dataset (pixels column is NaN which is of type float)
        if type(img_pixels) is float:
            # Read and resize image
            img_path = '../datasets/MuxspaceDataset/images/' + str(df.iloc[i]["image"])           
            img_data = read_and_resize_mux_image(img_path)
            
        # FER Dataset 
        else:
            img_data = read_and_resize_fer_image(img_pixels)
        
        # Normalize to between 0 and 1
        img_data = img_data/255   # Since values are in (0, 255)

        # Add RGB channels
        img_tensor = torch.from_numpy(np.copy(img_data)).unsqueeze(0).repeat(3,1,1)

        # Normalize to PyTorch requirements
        if normalize_tensors:
            img_tensor = normalize(img_tensor)
        
        # Folder for the specific emotion
        emotion = df.iloc[i]["emotion"]
        folder_name = master_path + str(emotion)
        
        # Create directory for emotion if it doesn't already exist
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        # Save if total images for emotion is less than total images
        if(count[emotion] < cutoff):
            # Save original image
            torch.save(img_tensor, folder_name + '/' + str(i) + '.tensor')
            count[emotion] += 1

    print("Finished saving to " + master_path)


'''
Only saves the rotated image
'''
def augment_and_rotate(df, subfolder, dataset, cutoff, count, normalize_tensors=False):
    """
    Args:
        df: dataframe
        subfolder: subfolder to save in (i.e. cutoff9100)
        dataset: name of dataset (train, validate, test)
        cutoff: total number of images we want for each class
        count: current count of each emotion, array
        normalize_tensors: whether to normalize tensors with mean and std given by pytorch
    """
    # Path to save processed tensor
    master_path = '../ProcessedData/' + subfolder + '/' + dataset + '/'
    
    # Create directory for master path
    if not os.path.isdir(master_path):
        os.mkdir(master_path)

    num_imgs = len(df.index)
    for i in range(num_imgs):

        img_pixels = df.iloc[i][" pixels"]
        
        # Muxspace dataset (pixels column is NaN which is of type float)
        if type(img_pixels) is float:
            # Read and resize image
            img_path = '../datasets/MuxspaceDataset/images/' + str(df.iloc[i]["image"])           
            img_data = read_and_resize_mux_image(img_path)
            
        # FER Dataset 
        else:
            img_data = read_and_resize_fer_image(img_pixels)
        
        # Normalize to between 0 and 1
        img_data = img_data/255
        # Rotate Image to the opposite way
        img_rotated_left_tensor = torch.from_numpy(scipy.ndimage.rotate(np.copy(img_data), -5, order=1, reshape=False)).unsqueeze(0).repeat(3,1,1)
        # Normalize to PyTorch requirements
        if normalize_tensors:
            img_rotated_left_tensor = normalize(img_rotated_left_tensor)

        # Folder for the specific emotion
        emotion = df.iloc[i]["emotion"]
        folder_name = master_path + str(emotion)
        
        # Create directory for emotion if it doesn't already exist
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        # Save if total images for emotion is less than total images
        if(count[emotion] < cutoff):
            # Save augmented images if class != 3 or 6
            torch.save(img_rotated_left_tensor, folder_name + '/' + str(i) + '_rotated_right.tensor')
            count[emotion] += 1

    print("Finished saving augmented images to " + master_path)

