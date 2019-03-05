import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from collections import OrderedDict
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, 
                 train_logger=None, tensorboard_image = False):
        """
        steps_update: how many steps to update parameters, use it when memory is not enough 
        """    
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.steps_update = self.config["trainer"]['steps_update']
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
        self.optimizer.zero_grad() # for gradient accumulation

        for batch_idx, (data, target) in enumerate(self.data_loader):
            
            data, target = data.to(self.device), target.to(self.device)
            isUpdateOptim = self.steps_update==1 or (batch_idx+1)%self.steps_update==0

            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()

            if isUpdateOptim:
                self.optimizer.step()
                self.optimizer.zero_grad()

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

        print("Already train a epoch")
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

        # print("cal_f1_score val")
        # self.cal_f1_score(dataset='val')
        # print("cal_f1_score train")
        # self.cal_f1_score(dataset='train')

        return log

    def cal_f1_score(self, dataset='train'):
        
        if dataset=='train':
            data_loader = self.data_loader
        else:
            data_loader = self.valid_data_loader

        self.model.eval()
        y_preds = y_trues = np.array([])
        batches = len(data_loader)
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss(output, target).data.cpu().numpy()
                total_loss += loss

                y_pred = output.data.cpu().max(1)[1].numpy()
                y_true = target.data.cpu().numpy()

                y_preds = np.append(y_preds, y_pred)
                y_trues = np.append(y_trues, y_true)
                if batch_idx%50==0: print("progress {}/{} ".format(batch_idx, batches))
        total_loss = total_loss/len(data_loader)
        res = f1_score(y_trues, y_preds, average='macro')
        acc = accuracy_score(y_trues, y_preds)
        print("{} dataset, f1 is {}, acc is {} loss is {}".format(dataset, res, \
            acc, total_loss))
        return res



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
