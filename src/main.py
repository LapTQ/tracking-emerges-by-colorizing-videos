from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent

sys.path.append(str(ROOT_DIR))

import src as GLOBAL
from src.utils.dataset import setup_dataset_and_transform
from src.models import model_factory
from src.utils.mics import set_seed, get_device

# ==================================================================================================

LOGGER = GLOBAL.LOGGER
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from queue import Queue
from time import sleep
from threading import Thread
import numpy as np
import cv2
import imgviz


def train():

    config = deepcopy(GLOBAL.CONFIG)
    config_train_dataset = config['dataset']['train']
    config_val_dataset = config['dataset']['val']
    config_transform = config['transform']['train']
    config_model = config['model']
    config_training = config['training']

    n_references = config_train_dataset['kwargs']['n_references']
    assert config_val_dataset['kwargs']['n_references'] == n_references

    config_model['module_name'] = {
        'backbone': config_model['backbone']['module_name'],
        'head': config_model['head']['module_name']
    }
    config_model['kwargs'] = {
        'backbone': config_model['backbone']['kwargs'],
        'head': config_model['head']['kwargs']
    }

    # set model parameters to match the input
    config_model['kwargs']['backbone']['in_channels'] = 1
    config_model['kwargs']['head']['n_references'] = n_references
    config_model['kwargs']['head']['in_channels'] = config_model['backbone']['kwargs']['mid_channels'][-1]

    set_seed()
    _ = setup_dataset_and_transform(
        config_dataset=config_train_dataset,
        config_input_transform=config_transform['input'],
        config_label_transform=config_transform['label']
    )
    train_dataloader = _['dataloader']
    label_transform = _['label_transform']

    _ = setup_dataset_and_transform(
        config_dataset=config_val_dataset,
        config_input_transform=config_transform['input'],
        config_label_transform=config_transform['label']
    )
    val_dataloader = _['dataloader']

    model = model_factory(
        **config_model['module_name']
    )(
        **config_model.get('kwargs', {})
    )
    LOGGER.info('Model:\n{}'.format(model))

    device = get_device(config_training['device'])
    model = model.to(device)
    criterion = eval(config_training['loss'])()
    optimizer = eval(config_training['optimizer']['module_name'])(
        model.parameters(),
        **config_training['optimizer']['kwargs']
    )
    schedulers = [
        eval(scheduler['module_name'])(
            optimizer,
            **scheduler['kwargs']
        ) for scheduler in config_training.get('schedulers', [])
    ]
    epochs = config_training['epochs']
    verbose_step = config_training['verbose_step']

    queue = Queue(maxsize=config_training['show_batch_queue_max_size'])
    stop_show_running_batch = False
    def _show_running_batch(in_queue):
        assert config_transform['label'][-2]['module_name'] == 'Quantize', \
            'Assuming the second last label transform to be Quantize.'
        quantize_transform = label_transform.transforms[-2]
        while not stop_show_running_batch:
            if in_queue.empty():
                sleep(0.1)
                continue

            _ = in_queue.get()
            X = _['X']
            true_color = _['true_color']
            predicted_color = _['predicted_color']  # (B, C, H, W)

            batch_size = X.shape[0]

            X = X[[i for i in range(batch_size) if i % (n_references + 1) == n_references]] # (B, C, H, W)
            X = (X.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')  # (B, H, W, C)
            true_color = true_color.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
            predicted_color = F.one_hot(
                torch.argmax(predicted_color, dim=1),
                num_classes=predicted_color.shape[1]
            ).cpu().numpy()   # (B, H, W, C)

            true_color = quantize_transform.invert_transform_batch(true_color).astype('uint8')
            predicted_color = quantize_transform.invert_transform_batch(predicted_color).astype('uint8')

            tile = [
                    cv2.cvtColor(
                        np.stack(
                            [
                                X[i, :, :, 0],
                                *[cv2.resize(color[i, :, :, _], X.shape[1:-1][::-1]) for _ in range(color.shape[-1])]
                            ],
                            axis=2
                        ),
                        cv2.COLOR_LAB2BGR
                    ) for i in range(len(X)) for color in [true_color, predicted_color]
            ]
            tile = imgviz.tile(
                tile,
                border=(255, 255, 255),
                border_width=5
            )
            cv2.imwrite('temp.jpg', tile)
            # window_title = 'Validation images'
            # cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            # cv2.imshow(window_title, tile)
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     exit(0)
        
        cv2.destroyAllWindows()

    show_val_thread = Thread(target=_show_running_batch, args=(queue,))
    show_val_thread.start()

    for epoch in range(epochs):

        # training
        running_loss = 0.0
        running_correct = 0
        total_train_loss = 0.0
        total_train_correct = 0
        model.train()
        for i, batch in enumerate(train_dataloader):
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)
            batch_size, _, H, W = X.shape
            true_color = Y[[i for i in range(batch_size) if i % (n_references + 1) == n_references]]
            ref_colors = Y[[i for i in range(batch_size) if i % (n_references + 1) != n_references]]
            ref_colors = ref_colors.float()

            optimizer.zero_grad()
            predicted_color = model(X, ref_colors)
            loss = criterion(predicted_color, true_color)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += torch.sum(
                torch.argmax(predicted_color, dim=1) == torch.argmax(true_color, dim=1)
            ).item()
            if i % verbose_step == verbose_step - 1:
                LOGGER.info('[Epoch {}/{}][Batch {}/{}] train loss: {}, train acc: {}'.format(
                    epoch + 1,
                    epochs,
                    i + 1,
                    len(train_dataloader),
                    running_loss / verbose_step,
                    running_correct / (verbose_step * H * W)
                ))
                running_loss = 0.0
                running_correct = 0
            total_train_loss += loss.item()
            total_train_correct += torch.sum(
                torch.argmax(predicted_color, dim=1) == torch.argmax(true_color, dim=1)
            ).item()
            
        total_train_loss /= len(train_dataloader)
        total_train_correct /= len(train_dataloader) * H * W

        # validation
        running_loss = 0.0
        running_correct = 0
        total_val_loss = 0.0
        total_val_correct = 0
        model.eval()
        for i, batch in enumerate(val_dataloader):
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)
            batch_size = X.shape[0]
            true_color = Y[[i for i in range(batch_size) if i % (n_references + 1) == n_references]]
            ref_colors = Y[[i for i in range(batch_size) if i % (n_references + 1) != n_references]]
            ref_colors = ref_colors.float()

            predicted_color = model(X, ref_colors)
            loss = criterion(predicted_color, true_color)

            running_loss += loss.item()
            running_correct += torch.sum(
                torch.argmax(predicted_color, dim=1) == torch.argmax(true_color, dim=1)
            ).item()
            if i % verbose_step == verbose_step - 1:
                LOGGER.info('[Epoch {}/{}][Batch {}/{}] val loss: {}, val acc: {}'.format(
                    epoch + 1,
                    epochs,
                    i + 1,
                    len(val_dataloader),
                    running_loss / verbose_step,
                    running_correct / (verbose_step * H * W)
                ))
                running_loss = 0.0
                running_correct = 0
            total_val_loss += loss.item()
            total_val_correct += torch.sum(
                torch.argmax(predicted_color, dim=1) == torch.argmax(true_color, dim=1)
            ).item()

            if queue.full():
                queue.get()
            queue.put({
                'X': X,
                'true_color': true_color,
                'predicted_color': predicted_color
            })
        
        total_val_loss /= len(val_dataloader)
        total_val_correct /= len(val_dataloader) * H * W

        LOGGER.info('[Epoch {}/{}] train loss: {}, val loss: {}, train acc: {}, val acc: {}, lr: {}'.format(
            epoch + 1,
            epochs,
            total_train_loss,
            total_val_loss,
            total_train_correct,
            total_val_correct,
            optimizer.param_groups[0]['lr']
        ))
        
        for scheduler in schedulers:
            scheduler.step(total_val_loss)
    
    stop_show_running_batch = True


if __name__ == '__main__':
    train()