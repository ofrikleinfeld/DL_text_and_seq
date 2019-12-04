from pathlib import Path
from datetime import date
import torch
from torch.utils import data
from models import BaseModel
from configs import TrainingConfig


class ModelTrainer(object):
    def __init__(self, model: BaseModel, train_config: TrainingConfig, loss_function):
        self.model = model
        self.train_config = train_config
        self.loss_function = loss_function
        self.current_epoch = 0

    def save_checkpoint(self, epoch: int, epoch_loss: float, best_model: bool = False) -> None:
        model_name = type(self.model).__name__
        checkpoint_base_dir = self.train_config.checkpoints_path
        current_date = date.today().strftime("%d-%m-%y")

        # construct checkpoint folder name and create it
        checkpoint_path = Path(checkpoint_base_dir) / model_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # construct checkpoint file name
        if best_model:
            checkpoint_file = checkpoint_path / f"{current_date}_best_model.pth"
        else:
            checkpoint_file = checkpoint_path / f"{current_date}_epoch_{epoch}_loss_{epoch_loss:.2f}.pth"

        checkpoint_data = {
            "serialized_model": self.model.serialize_model(),
            "train_config": self.train_config.to_dict(),
            "epoch": epoch
        }

        # save data to disk
        torch.save(checkpoint_data, checkpoint_file)

    @classmethod
    def load_checkpoint_for_training(cls, path_to_pth_file: str, uninitialized_model_class, loss_function):
        checkpoint_data = torch.load(path_to_pth_file)

        # extract checkpoint data components
        serialized_model: dict = checkpoint_data["serialized_model"]
        train_config_dict: dict = checkpoint_data["train_config"]
        current_epoch: int = checkpoint_data["epoch"]

        model: BaseModel = uninitialized_model_class.deserialize_model(serialized_model)
        train_config: TrainingConfig = TrainingConfig(train_config_dict)

        # create a new model trainer object
        trainer = cls(model, train_config, loss_function)
        trainer.current_epoch = current_epoch

        return trainer

    @staticmethod
    def load_trained_model(path_to_pth_file: str,  uninitialized_model_class) -> BaseModel:
        checkpoint_data = torch.load(path_to_pth_file)

        # extract serialized_model and create a model object
        serialized_model: dict = checkpoint_data["serialized_model"]
        model: BaseModel = uninitialized_model_class.deserialize_model(serialized_model)

        return model

    def train(self, train_dataset: data.Dataset, dev_dataset: data.Dataset):
        # training hyper parameters and configuration
        batch_size = self.train_config.batch_size
        num_workers = self.train_config.num_workers
        num_epochs = self.train_config.num_epochs
        learning_rate = self.train_config.learning_rate
        print_batch_step = self.train_config.print_step
        checkpoint_step = self.train_config.checkpoints_step
        device = torch.device(self.train_config.device)

        # model, loss and optimizer
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # create data loaders
        train_config_dict = {"batch_size": batch_size, "num_workers": num_workers}
        training_loader = data.DataLoader(train_dataset, **train_config_dict)
        dev_loader = data.DataLoader(dev_dataset, **train_config_dict)

        # Start training
        model = model.to(device)
        start_epoch = self.current_epoch
        best_dev_accuracy = 1000
        for epoch in range(start_epoch, num_epochs):

            model.train(mode=True)
            epoch_num = epoch + 1

            for batch_idx, sample in enumerate(training_loader):

                x, y = sample
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = self.loss_function(output, y)

                loss.backward()
                optimizer.step()

                # print inter epoch statistics
                if batch_idx % print_batch_step == 0:
                    print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch_num, batch_idx * batch_size, len(train_dataset),
                        100. * batch_idx / len(training_loader), loss.item()))

            # compute average loss for epoch on dev set
            # if with_dev:
            #     with torch.no_grad():
            #         model.train(mode=False)  # equivalent to model.eval()
            #         epoch_loss = 0
            #
            #         for i, sample in enumerate(dev_loader, 1):
            #
            #             x, y = sample
            #             x, y = x.to(device), y.to(device)
            #             output = model(x)
            #             loss = self.loss_function(output, y)
            #             epoch_loss += loss.item() * batch_size
            #
            #         dev_average_epoch_loss = epoch_loss / len(dev_dataset)
            #         print(f"Epoch {epoch_num}: average loss on development set is {dev_average_epoch_loss}")

            # # save checkpoint of the model if needed
            # average_epoch_loss = dev_average_epoch_loss if with_dev else train_average_epoch_loss
            # if epoch % checkpoint_step == 0:
            #     self.save_checkpoint(epoch, average_epoch_loss, best_model=False)
            #
            # if dev_average_epoch_loss < best_epoch_loss:
            #     best_epoch_loss = average_epoch_loss
            #     self.save_checkpoint(epoch, best_epoch_loss, best_model=True)