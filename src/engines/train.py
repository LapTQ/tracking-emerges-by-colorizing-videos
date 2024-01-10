import logging
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import imgviz
import wandb


logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
            self,
            **kwargs
    ):
        self.model = kwargs['model']
        self.train_dataloader = kwargs['train_dataloader']
        self.val_dataloader = kwargs['val_dataloader']
        self.device = kwargs['device']
        self.n_references = kwargs['n_references']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']

    
    def train(
            self,
            **kwargs
    ):
        epochs = kwargs['epochs']
        scheduler = kwargs.get('scheduler', None)
        callbacks = kwargs.get('callbacks', [])
        callback_targets = kwargs.get('callback_targets', [])
        quantize_transform = kwargs['quantize_transform']
        verbose_step = kwargs['verbose_step']

        assert len(callbacks) == len(callback_targets)

        self.model.to(self.device)
        wandb.watch(self.model, log_freq=1, log='all')

        callback_stop = False
        for epoch in range(epochs):
            train_info = self.step(
                mode='train',
                epoch=epoch,
                epochs=epochs,
                verbose_step=verbose_step
            )
            train_loss = train_info['loss']
            train_acc = train_info['acc']

            val_info = self.step(
                mode='val',
                epoch=epoch,
                epochs=epochs,
                quantize_transform=quantize_transform,
                verbose_step=verbose_step
            )
            val_loss = val_info['loss']
            val_acc = val_info['acc']

            logger.info('[Epoch {}/{}] train loss: {}, val loss: {}, train acc: {}, val acc: {}, lr: {}'.format(
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                self.optimizer.param_groups[0]['lr']
            ))

            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            if scheduler is not None:
                scheduler.step(train_loss)

            self.model.save_checkpoint()

            for callback, target in zip(callbacks, callback_targets):
                assert target in ['train_loss', 'val_loss', 'train_acc', 'val_acc']
                if not callback.step(
                    value=train_loss if target == 'train_loss' \
                    else val_loss if target == 'val_loss' \
                    else train_acc if target == 'train_acc' \
                    else val_acc
                ):
                    callback_stop = True
                    break

            if callback_stop:
                break
        
        wandb.finish()

    
    def step(
            self,
            **kwargs
    ):
        mode = kwargs['mode']
        epoch = kwargs['epoch']
        epochs = kwargs['epochs']
        quantize_transform = kwargs.get('quantize_transform', None)
        verbose_step = kwargs['verbose_step']

        if mode == 'train':
            dataloader = self.train_dataloader
        elif mode == 'val':
            dataloader = self.val_dataloader
            assert quantize_transform is not None
        else:
            raise ValueError('Invalid mode: {}'.format(mode))


        running_loss = 0.0
        running_acc = 0
        epoch_loss = 0.0
        epoch_acc = 0

        if mode == 'train':
            self.model.train()
        elif mode == 'val':
            self.model.eval()

        for b_idx, batch in enumerate(dataloader):
            X, Y = batch
            X = X.to(self.device)
            Y = Y.to(self.device)
            batch_size, _, H, W = Y.shape
            true_color = Y[[i for i in range(batch_size) if i % (self.n_references + 1) == self.n_references]]
            ref_colors = Y[[i for i in range(batch_size) if i % (self.n_references + 1) != self.n_references]]
            ref_colors = ref_colors.float()

            if mode == 'train':
                self.optimizer.zero_grad()
            
            predicted_color = self.model(X, ref_colors)
            loss = self.criterion(predicted_color, true_color)

            if mode == 'train':
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            n_corrects = torch.sum(
                torch.argmax(predicted_color, dim=1) == torch.argmax(true_color, dim=1)
            ).item()
            running_acc += n_corrects
            if b_idx % verbose_step == verbose_step - 1:
                logger.info('[Epoch {}/{}][Batch {}/{}] {} loss: {}, {} acc: {}'.format(
                    epoch + 1,
                    epochs,
                    b_idx + 1,
                    len(dataloader),
                    mode,
                    running_loss / verbose_step,
                    mode,
                    round(running_acc / (verbose_step * H * W) / (batch_size // (self.n_references + 1)) * 100, 1)
                ))
                running_loss = 0.0
                running_acc = 0
            epoch_loss += loss.item()
            epoch_acc += n_corrects
            
            if mode == 'train':
                wandb.log({"loss": loss})

            if mode == 'val':
                self._show_running_batch(
                    epoch=epoch,
                    batch_id=b_idx,
                    X=X,
                    true_color=true_color,
                    predicted_color=predicted_color,
                    quantize_transform=quantize_transform
                )
            
        epoch_loss /= len(dataloader)
        epoch_acc = round(epoch_acc / (len(dataloader) * H * W * batch_size // (self.n_references + 1)) * 100, 1)

        return {
            'loss': epoch_loss,
            'acc': epoch_acc
        }
    

    def _show_running_batch(
            self,
            **kwargs
    ):
        epoch = kwargs['epoch']
        batch_id = kwargs['batch_id']
        X = kwargs['X']
        true_color = kwargs['true_color']
        predicted_color = kwargs['predicted_color'] # (B, C, H, W)
        quantize_transform = kwargs['quantize_transform']

        table = wandb.Table(
            columns=['epoch', 'batch', 'true_color', 'predicted_color'],
        )

        batch_size = X.shape[0]

        X = X[[i for i in range(batch_size) if i % (self.n_references + 1) == self.n_references]] # (B, C, H, W)
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

        for i in range(len(tile) // 2):
            table.add_data(
                epoch,
                batch_id,
                wandb.Image(tile[2*i]),
                wandb.Image(tile[2*i+1])
            )
            wandb.log({"Predictions_table": table}, commit=True)

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



        
        
        
