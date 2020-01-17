from typing import Tuple

import torch
from torch.utils import data

from pos_and_ner.trainers import ModelTrainer
from pos_and_ner.models import BaseModel
from pos_and_ner.configs import TrainingConfig
from pos_and_ner.predictors import BasePredictor


class SNLITrainer(ModelTrainer):

    def __init__(self, model: BaseModel, train_config: TrainingConfig, predictor: BasePredictor, loss_function):
        super().__init__(model, train_config, predictor, loss_function)

    def train(self, model_name: str, train_dataset: data.Dataset, dev_dataset: data.Dataset) -> None:
        # training hyper parameters and configuration
        train_batch_size = self.train_config["batch_size"]
        dev_batch_size = train_batch_size * 60
        num_workers = self.train_config["num_workers"]
        num_epochs = self.train_config["num_epochs"]
        learning_rate = self.train_config["learning_rate"]
        print_batch_step = self.train_config["print_step"]
        device = torch.device(self.train_config["device"])

        # model, loss and optimizer
        model = self.model
        model = model.to(device)
        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        # create data loaders
        train_config_dict = {"batch_size": train_batch_size, "num_workers": num_workers}
        dev_config_dict = {"batch_size": dev_batch_size, "num_workers": num_workers}
        training_loader = data.DataLoader(train_dataset, **train_config_dict)
        dev_loader = data.DataLoader(dev_dataset, **dev_config_dict)

        # Start training
        print("Starting to train...")
        start_epoch = self.current_epoch
        best_dev_accuracy = -1000
        for epoch in range(start_epoch, num_epochs):

            model.train(mode=True)
            epoch_num = epoch + 1

            for batch_idx, sample in enumerate(training_loader, 1):

                x_1, x_2, y = sample
                x_1, x_2, y = x_1.to(device), x_2.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = model(x_1, x_2)
                loss = self.loss_function(outputs, y)

                loss.backward()
                optimizer.step()

                # print inter epoch statistics
                if batch_idx % print_batch_step == 0:
                    train_loss, train_accuracy = self.predict_accuracy(model, device, training_loader)
                    dev_loss, dev_accuracy = self.predict_accuracy(model, device, dev_loader)
                    print("Train Epoch: {} [{}/{} ({:.0f}%)]\t Train Loss: {:.6f}, Train Accuracy: {:.6f}, Dev Loss {:.6f}, Dev Accuracy: {:.6f}".format(
                        epoch_num,
                        batch_idx * train_batch_size,
                        len(train_dataset) + 1,
                        100. * batch_idx / len(training_loader),
                        train_loss, train_accuracy,
                        dev_loss, dev_accuracy
                    ))

                    # move model back to training mode
                    model.train(mode=True)

            # end of epoch - compute loss on dev set
            train_loss, train_accuracy = self.predict_accuracy(model, device, training_loader)
            dev_loss, dev_accuracy = self.predict_accuracy(model, device, dev_loader)
            print("Epoch {} Loss on Train set is:\t{:.6f}, Accuracy on Train set is:\t{:.6f}".format(epoch_num, train_loss, train_accuracy))
            print("Epoch {} Loss on Dev set is:\t{:.6f}, Accuracy on Dev set is:\t{:.6f}".format(epoch_num, dev_loss, dev_accuracy))

            # save checkpoint of the model if needed
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                print("Epoch {} Saving best model so far with accuracy of {:.6f} on Dev set".format(epoch_num, best_dev_accuracy))
                self.save_checkpoint(model_name)

    def predict_accuracy(self, model: torch.nn.Module, device: torch.device, loader: data.DataLoader) -> Tuple[float, float]:
        dataset_loss = 0
        model.eval()
        total_predictions = 0
        num_correct_predictions = 0
        with torch.no_grad():

            for batch_idx, sample in enumerate(loader):
                x_1, x_2, y = sample
                x_1, x_2, y = x_1.to(device), x_2.to(device), y.to(device)
                outputs = model(x_1, x_2)

                # compute the loss of the batch
                loss = self.loss_function(outputs, y)
                dataset_loss += loss.item() * len(y)  # sum of losses, so multiply with batch size

                # compute number of correct predictions for the batch
                num_correct_batch, total_predictions_batch = self.predictor.infer_model_outputs_with_gold_labels(outputs, y)
                num_correct_predictions += num_correct_batch
                total_predictions += total_predictions_batch

        average_dataset_loss = dataset_loss / total_predictions
        dataset_accuracy = num_correct_predictions / total_predictions

        return average_dataset_loss, dataset_accuracy
