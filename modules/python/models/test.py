import sys
import torch
from tqdm import tqdm
import torchnet.meter as meter
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from modules.python.models.dataloader import SequenceDataset
from modules.python.TextColor import TextColor
from modules.python.Options import ImageSizeOptions, TrainOptions
"""
This script will evaluate a model and return the loss value.

Input:
- A trained model
- A test CSV file to evaluate

Returns:
- Loss value
"""
CLASS_WEIGHTS = [0.3, 1.0, 1.0, 1.0, 1.0]
label_decoder = {0: '*', 1: 'A', 2: 'C', 3: 'G', 4: 'T', 5: '#'}


def test(data_file, batch_size, gpu_mode, transducer_model, num_workers, gru_layers, hidden_size,
         num_classes=ImageSizeOptions.TOTAL_LABELS, print_details=False):
    # transformations = transforms.Compose([transforms.ToTensor()])

    # data loader
    test_data = SequenceDataset(data_file)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=gpu_mode)
    # sys.stderr.write(TextColor.CYAN + 'Test data loaded\n')

    # set the evaluation mode of the model
    transducer_model.eval()

    class_weights = torch.Tensor(CLASS_WEIGHTS)
    # Loss
    criterion = nn.CrossEntropyLoss(class_weights)

    if gpu_mode is True:
        criterion = criterion.cuda()

    # Test the Model
    sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    confusion_matrix_h1 = meter.ConfusionMeter(num_classes)
    confusion_matrix_h2 = meter.ConfusionMeter(num_classes)

    total_loss = 0
    total_images = 0
    accuracy_h1 = 0
    accuracy_h2 = 0

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Accuracy: ', leave=True, ncols=100) as pbar:
            for ii, (images, labels) in enumerate(test_loader):
                labels = labels.type(torch.LongTensor)
                images = images.type(torch.FloatTensor)
                if gpu_mode:
                    # encoder_hidden = encoder_hidden.cuda()
                    images = images.cuda()
                    labels = labels.cuda()

                hidden_h1 = torch.zeros(images.size(0), 2 * TrainOptions.GRU_LAYERS, TrainOptions.HIDDEN_SIZE)
                hidden_h2 = torch.zeros(images.size(0), 2 * TrainOptions.GRU_LAYERS, TrainOptions.HIDDEN_SIZE)

                if gpu_mode:
                    hidden_h1 = hidden_h1.cuda()
                    hidden_h2 = hidden_h2.cuda()

                for i in range(0, ImageSizeOptions.SEQ_LENGTH, TrainOptions.WINDOW_JUMP):
                    if i + TrainOptions.TRAIN_WINDOW > ImageSizeOptions.SEQ_LENGTH:
                        break

                    image_chunk_h  = images[:, 0, i:i+TrainOptions.TRAIN_WINDOW]
                    image_chunk_h1 = images[:, 1, i:i+TrainOptions.TRAIN_WINDOW]
                    image_chunk_h2 = images[:, 2, i:i+TrainOptions.TRAIN_WINDOW]
                    label_chunk_h1 = labels[:, 0, i:i+TrainOptions.TRAIN_WINDOW]
                    label_chunk_h2 = labels[:, 1, i:i+TrainOptions.TRAIN_WINDOW]

                    hap_1_tensor = torch.cat((image_chunk_h1, image_chunk_h2), 2)
                    hap_2_tensor = torch.cat((image_chunk_h2, image_chunk_h1), 2)

                    out_h1, out_h2, hidden_h1, hidden_h2 = \
                        transducer_model(hap_1_tensor, hap_2_tensor, hidden_h1, hidden_h2)

                    h1_loss = criterion(out_h1.contiguous().view(-1, num_classes), label_chunk_h1.contiguous().view(-1))
                    h2_loss = criterion(out_h2.contiguous().view(-1, num_classes), label_chunk_h2.contiguous().view(-1))
                    loss = h1_loss + h2_loss

                    confusion_matrix_h1.add(out_h1.data.contiguous().view(-1, num_classes),
                                            label_chunk_h1.data.contiguous().view(-1))

                    confusion_matrix_h2.add(out_h2.data.contiguous().view(-1, num_classes),
                                            label_chunk_h2.data.contiguous().view(-1))

                    total_loss += loss.item()
                    total_images += images.size(0)

                pbar.update(1)
                cm_value_h1 = confusion_matrix_h1.value()
                cm_value_h2 = confusion_matrix_h2.value()
                denom_h1 = cm_value_h1.sum() if cm_value_h1.sum() > 0 else 1.0
                denom_h2 = cm_value_h2.sum() if cm_value_h2.sum() > 0 else 1.0
                accuracy_h1 = 100.0 * (cm_value_h1[0][0] + cm_value_h1[1][1] + cm_value_h1[2][2]
                                       + cm_value_h1[3][3] + cm_value_h1[4][4]) / denom_h1
                accuracy_h2 = 100.0 * (cm_value_h2[0][0] + cm_value_h2[1][1] + cm_value_h2[2][2]
                                       + cm_value_h2[3][3] + cm_value_h2[4][4]) / denom_h2
                pbar.set_description("Accuracy H1: " + str(round(accuracy_h1, 5)) +
                                     " H2: " + str(round(accuracy_h2, 5)))

    avg_loss = total_loss / total_images if total_images else 0

    sys.stderr.write(TextColor.YELLOW+'\nTest Loss: ' + str(avg_loss) + "\n"+TextColor.END)
    sys.stderr.write("Confusion Matrix H1: \n" + str(confusion_matrix_h1.conf) + "\n" + TextColor.END)
    sys.stderr.write("Confusion Matrix H2: \n" + str(confusion_matrix_h2.conf) + "\n" + TextColor.END)

    return {'loss': avg_loss, 'accuracy_h1': accuracy_h1, 'accuracy_h2': accuracy_h2,
            'confusion_matrix_h1': str(confusion_matrix_h1.conf), 'confusion_matrix_h2': str(confusion_matrix_h2.conf)}
