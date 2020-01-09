import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules.python.models.dataloader_predict import SequenceDataset
from modules.python.TextColor import TextColor
from tqdm import tqdm
import numpy as np
from modules.python.models.ModelHander import ModelHandler
from modules.python.Options import ImageSizeOptions, TrainOptions
from modules.python.DataStorePredict import DataStore


def predict(test_file, output_filename, model_path, batch_size, threads, num_workers, gpu_mode):
    """
    Create a prediction table/dictionary of an images set using a trained model.
    :param test_file: File to predict on
    :param batch_size: Batch size used for prediction
    :param model_path: Path to a trained model
    :param gpu_mode: If true, predictions will be done over GPU
    :param threads: Number of threads to set for pytorch
    :param num_workers: Number of workers to be used by the dataloader
    :return: Prediction dictionary
    """
    prediction_data_file = DataStore(output_filename, mode='w')
    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    torch.set_num_threads(threads)
    sys.stderr.write(TextColor.GREEN + 'INFO: TORCH THREADS SET TO: ' + str(torch.get_num_threads()) + ".\n"
                     + TextColor.END)
    sys.stderr.flush()

    # data loader
    test_data = SequenceDataset(test_file)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    transducer_model, hidden_size, gru_layers, prev_ite = \
        ModelHandler.load_simple_model_for_training(model_path,
                                                    input_channels=ImageSizeOptions.IMAGE_CHANNELS,
                                                    image_features=ImageSizeOptions.IMAGE_HEIGHT,
                                                    seq_len=ImageSizeOptions.SEQ_LENGTH,
                                                    num_classes=ImageSizeOptions.TOTAL_LABELS)
    transducer_model.eval()

    if gpu_mode:
        transducer_model = torch.nn.DataParallel(transducer_model).cuda()
    sys.stderr.write(TextColor.CYAN + 'MODEL LOADED\n')

    with torch.no_grad():
        for contig, contig_start, contig_end, chunk_id, images, position, index in tqdm(test_loader, ncols=50):
            sys.stderr.flush()
            images = images.type(torch.FloatTensor)
            if gpu_mode:
                images = images.cuda()

            hidden_h1 = torch.zeros(images.size(0), 2 * TrainOptions.GRU_LAYERS, TrainOptions.HIDDEN_SIZE)
            hidden_h2 = torch.zeros(images.size(0), 2 * TrainOptions.GRU_LAYERS, TrainOptions.HIDDEN_SIZE)

            if gpu_mode:
                hidden_h1 = hidden_h1.cuda()
                hidden_h2 = hidden_h2.cuda()

            prediction_base_dict_h1 = np.zeros((images.size(0), ImageSizeOptions.SEQ_LENGTH, ImageSizeOptions.TOTAL_LABELS))
            prediction_base_dict_h2 = np.zeros((images.size(0), ImageSizeOptions.SEQ_LENGTH, ImageSizeOptions.TOTAL_LABELS))

            for i in range(0, ImageSizeOptions.SEQ_LENGTH, TrainOptions.WINDOW_JUMP):
                if i + TrainOptions.TRAIN_WINDOW > ImageSizeOptions.SEQ_LENGTH:
                    break
                chunk_start = i
                chunk_end = i + TrainOptions.TRAIN_WINDOW
                # chunk all the data
                image_chunk_h  = images[:, 0, i:i+TrainOptions.TRAIN_WINDOW]
                image_chunk_h1 = images[:, 1, i:i+TrainOptions.TRAIN_WINDOW]
                image_chunk_h2 = images[:, 2, i:i+TrainOptions.TRAIN_WINDOW]

                hap_1_tensor = torch.cat((image_chunk_h1, image_chunk_h2), 2)
                hap_2_tensor = torch.cat((image_chunk_h2, image_chunk_h1), 2)

                # run inference
                out_h1, out_h2, hidden_h1, hidden_h2 = \
                    transducer_model(hap_1_tensor, hap_2_tensor, hidden_h1, hidden_h2)

                # do softmax and get prediction
                m = nn.Softmax(dim=2)
                soft_probs_h1 = m(out_h1)
                output_preds_h1 = soft_probs_h1.cpu()
                base_max_value_h1, predicted_base_label_h1 = torch.max(output_preds_h1, dim=2)

                # do softmax and get prediction
                m = nn.Softmax(dim=2)
                soft_probs_h2 = m(out_h2)
                output_preds_h2 = soft_probs_h2.cpu()
                base_max_value_h2, predicted_base_label_h2 = torch.max(output_preds_h2, dim=2)

                # convert everything to list
                base_max_value_h1 = base_max_value_h1.numpy().tolist()
                base_max_value_h2 = base_max_value_h2.numpy().tolist()
                predicted_base_label_h1 = predicted_base_label_h1.numpy().tolist()
                predicted_base_label_h2 = predicted_base_label_h2.numpy().tolist()

                assert(len(base_max_value_h1) == len(predicted_base_label_h1))
                assert(len(base_max_value_h2) == len(predicted_base_label_h2))

                for ii in range(0, len(predicted_base_label_h1)):
                    chunk_pos = chunk_start
                    for p_base, base in zip(base_max_value_h1[ii], predicted_base_label_h1[ii]):
                        prediction_base_dict_h1[ii][chunk_pos][base] += p_base
                        chunk_pos += 1

                for ii in range(0, len(predicted_base_label_h2)):
                    chunk_pos = chunk_start
                    for p_base, base in zip(base_max_value_h2[ii], predicted_base_label_h2[ii]):
                        prediction_base_dict_h2[ii][chunk_pos][base] += p_base
                        chunk_pos += 1
            predicted_base_labels_h1 = np.argmax(np.array(prediction_base_dict_h1), axis=2)
            predicted_base_labels_h2 = np.argmax(np.array(prediction_base_dict_h2), axis=2)

            for i in range(images.size(0)):
                prediction_data_file.write_prediction(contig[i], contig_start[i], contig_end[i], chunk_id[i],
                                                      position[i], index[i], predicted_base_labels_h1[i],
                                                      predicted_base_labels_h2[i])
