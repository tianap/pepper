from os.path import isfile, join
from os import listdir
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import sys
from modules.python.Options import ImageSizeOptions
from modules.python.TextColor import TextColor


def get_file_paths_from_directory(directory_path):
    """
    Returns all paths of files given a directory path
    :param directory_path: Path to the directory
    :return: A list of paths of files
    """
    file_paths = [join(directory_path, file) for file in listdir(directory_path) if isfile(join(directory_path, file))
                  and file[-3:] == 'hdf']
    return file_paths


class SequenceDataset(Dataset):
    """
    Arguments:
        A HDF5 file path
    """
    def __init__(self, image_directory, load_labels=False):
        self.transform = transforms.Compose([transforms.ToTensor()])
        file_image_pair = []

        hdf_files = get_file_paths_from_directory(image_directory)

        for hdf5_file_path in hdf_files:
            with h5py.File(hdf5_file_path, 'r') as hdf5_file:
                if 'summaries' in hdf5_file:
                    image_names = list(hdf5_file['summaries'].keys())

                    for image_name in image_names:
                        file_image_pair.append((hdf5_file_path, image_name))
                else:
                    sys.stderr.write(TextColor.YELLOW + "WARN: NO IMAGES FOUND IN FILE: "
                                     + hdf5_file_path + "\n" + TextColor.END)

        self.all_images = file_image_pair
        self.load_labels = load_labels

    def __getitem__(self, index):
        # load the image
        hdf5_filepath, image_name = self.all_images[index]

        with h5py.File(hdf5_filepath, 'r') as hdf5_file:
            image = hdf5_file['summaries'][image_name]['image'][()]
            position = hdf5_file['summaries'][image_name]['position'][()]
            index = hdf5_file['summaries'][image_name]['index'][()]
            contig = hdf5_file['summaries'][image_name]['contig'][()]
            chunk_id = hdf5_file['summaries'][image_name]['chunk_id'][()]
            contig_start = hdf5_file['summaries'][image_name]['region_start'][()]
            contig_end = hdf5_file['summaries'][image_name]['region_end'][()]
            ref_seq = hdf5_file['summaries'][image_name]['ref_seq'][()]
            coverage = hdf5_file['summaries'][image_name]['coverage'][()]
            if self.load_labels:
                label = hdf5_file['summaries'][image_name]['label'][()]
            else:
                label = []

        return contig, contig_start, contig_end, chunk_id, image, position, index, ref_seq, coverage, label

    def __len__(self):
        return len(self.all_images)
