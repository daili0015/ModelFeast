import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from collections import OrderedDict

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, 
                 train_logger=None, tensorboard_image = False, steps_update=1):
        """
        steps_update: how many steps to update parameters, use it when memory is not enough 
        """    
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.steps_update = steps_update
        self.steps_to_verb = len(self.data_loader)//self.verbose_per_epoch
        self.steps_to_verb = 1 if self.steps_to_verb<=0 else self.steps_to_verb

        self.tensorboard_image = tensorboard_image # whether save img when train

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(str(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train() # turn to train mode
    
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.data_loader):
            
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            if  self.steps_update==1 or batch_idx%self.steps_update==0:
                self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.steps_to_verb == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                if self.tensorboard_image:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = OrderedDict()
        log['train_loss'] = total_loss / len(self.data_loader)
        log['metrics'] = (total_metrics / len(self.data_loader)).tolist()
        # log = {'train_loss': total_loss / len(self.data_loader),
        #     'metrics': (total_metrics / len(self.data_loader)).tolist()}

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            for key, value in val_log.items():
                log[key] = value

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                if self.tensorboard_image:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        log = OrderedDict()
        log['val_loss'] = total_val_loss / len(self.valid_data_loader)
        log['val_metrics'] = (total_val_metrics / len(self.valid_data_loader)).tolist()

        return log
