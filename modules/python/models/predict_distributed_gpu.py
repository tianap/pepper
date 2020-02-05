import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from modules.python.models.dataloader_predict import SequenceDataset
from tqdm import tqdm
from modules.python.models.ModelHander import ModelHandler
from modules.python.Options import ImageSizeOptions, TrainOptions
from modules.python.DataStorePredict import DataStore
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'


def predict(input_filepath, file_chunks, output_filepath, model_path, batch_size, num_workers, total_devices, device_id):
    transducer_model, hidden_size, gru_layers, prev_ite = \
        ModelHandler.load_simple_model_for_training(model_path,
                                                    input_channels=ImageSizeOptions.IMAGE_CHANNELS,
                                                    image_features=ImageSizeOptions.IMAGE_HEIGHT,
                                                    seq_len=ImageSizeOptions.SEQ_LENGTH,
                                                    num_classes=ImageSizeOptions.TOTAL_LABELS)
    transducer_model.eval()
    transducer_model = transducer_model.eval()
    # create output file
    output_filename = output_filepath + "pepper_prediction_" + str(device_id) + ".hdf"
    prediction_data_file = DataStore(output_filename, mode='w')

    # data loader
    input_data = SequenceDataset(input_filepath, file_chunks)
    data_loader = DataLoader(input_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    torch.cuda.set_device(device_id)
    transducer_model.to(device_id)
    transducer_model.eval()
    transducer_model = DistributedDataParallel(transducer_model, device_ids=[device_id])

    progress_bar = tqdm(
        total=len(data_loader),
        ncols=100,
        leave=False,
        position=device_id,
        desc="GPU #" + str(device_id),
    )

    with torch.no_grad():
        for contig, contig_start, contig_end, chunk_id, images, position, index, ref_seq in data_loader:
            sys.stderr.flush()
            images = images.type(torch.FloatTensor)
            hidden = torch.zeros(images.size(0), 2 * TrainOptions.GRU_LAYERS, TrainOptions.HIDDEN_SIZE)

            prediction_base_tensor = torch.zeros((images.size(0), images.size(1), ImageSizeOptions.TOTAL_LABELS))

            images = images.to(device_id)
            hidden = hidden.to(device_id)
            prediction_base_tensor = prediction_base_tensor.to(device_id)

            for i in range(0, ImageSizeOptions.SEQ_LENGTH, TrainOptions.WINDOW_JUMP):
                if i + TrainOptions.TRAIN_WINDOW > ImageSizeOptions.SEQ_LENGTH:
                    break
                chunk_start = i
                chunk_end = i + TrainOptions.TRAIN_WINDOW
                # chunk all the data
                image_chunk = images[:, chunk_start:chunk_end]

                # run inference
                output_base, hidden = transducer_model(image_chunk, hidden)

                # now calculate how much padding is on the top and bottom of this chunk so we can do a simple
                # add operation
                top_zeros = chunk_start
                bottom_zeros = ImageSizeOptions.SEQ_LENGTH - chunk_end

                # do softmax and get prediction
                # we run a softmax a padding to make the output tensor compatible for adding
                inference_layers = nn.Sequential(
                    nn.Softmax(dim=2),
                    nn.ZeroPad2d((0, 0, top_zeros, bottom_zeros))
                )
                inference_layers = inference_layers.to(device_id)

                # run the softmax and padding layers
                base_prediction = (inference_layers(output_base) * 10).type(torch.IntTensor).to(device_id)

                # now simply add the tensor to the global counter
                prediction_base_tensor = torch.add(prediction_base_tensor, base_prediction)

                del inference_layers
                torch.cuda.empty_cache()

            prediction_base_tensor = prediction_base_tensor.cpu().numpy().astype(int)

            for i in range(images.size(0)):
                prediction_data_file.write_prediction(contig[i], contig_start[i], contig_end[i], chunk_id[i],
                                                      position[i], index[i], prediction_base_tensor[i], ref_seq[i])
            progress_bar.update(1)

    progress_bar.close()


def cleanup():
    dist.destroy_process_group()


def setup(rank, total_threads, args, all_input_files):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=total_threads)

    filepath, output_filepath, model_path, batch_size, num_workers = args

    # issue with semaphore lock: https://github.com/pytorch/pytorch/issues/2517
    # mp.set_start_method('spawn')

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases. https://github.com/pytorch/pytorch/issues/2517
    # torch.manual_seed(42)
    predict(filepath, all_input_files[rank],  output_filepath, model_path, batch_size, num_workers, total_threads, rank)
    cleanup()


def predict_distributed_gpu(filepath, file_chunks, output_filepath, model_path, batch_size, threads, num_workers):
    """
    Create a prediction table/dictionary of an images set using a trained model.
    :param filepath: Path to image files to predict on
    :param file_chunks: Path to chunked files
    :param batch_size: Batch size used for prediction
    :param model_path: Path to a trained model
    :param output_filepath: Path to output directory
    :param threads: Number of threads to set for pytorch
    :param num_workers: Number of workers to be used by the dataloader
    :return: Prediction dictionary
    """
    args = (filepath, output_filepath, model_path, batch_size, num_workers)
    mp.spawn(setup,
             args=(threads, args, file_chunks),
             nprocs=threads,
             join=True)
