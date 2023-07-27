# import argparse
# parser = argparse.ArgumentParser()

# parser.add_argument('--name', type = str , required = True)
# parser.add_argument('--lr', type = float , required = False, default=0.1)
# parser.add_argument('--batch_size' ,type = int , required = False, default = 128)
# parser.add_argument('--wd', type = float , required = False, default = 1e-4)
# parser.add_argument('--sparsity', type = float , required = True)
# parser.add_argument('--epochs', type = int, required = False , default = 1000)
# parser.add_argument('--architecture', type = str , required = False, default = 'resnet20')
# parser.add_argument('--method', type = str, required = False, default = 'rigl')
# parser.add_argument('--momentum', type =  float, required = False, default = 0.9)
# parser.add_argument('--device_id', type = str, required = False, default = "1")
# args = parser.parse_args()
import wandb
wandb.login(key = "23a6d19ba7e2b661adf79adccf3ff4fddabaf0e2")
# hparams = {"name" : args.name ,"lr" : args.lr , "batch_size" : args.batch_size ,"epochs" : args.epochs, "architecture" : args.architecture , "device_id" : args.device_id, "weight_decay" : args.wd, "method" : args.method, "momentum" : args.momentum, "sparsity" : args.sparsity}
# wandb.init(project='RigL', entity='ucalgary', name = 'test')
sweep_configuration = {
    'method': 'grid',
    'name': 'rigL_sweeps_test',
    'metric': {
        'goal': 'maximize', 
        'name': 'eval_acc'
        },
    'parameters': {
        'sparsity': {'values': [0.5, 0.6, 0.7,0.8,0.9]},
        'seed': {'values': [0,1,2,3,4]},
        
     }
}
sweep_id = wandb.sweep(sweep_configuration, entity = 'ucalgary' , project = 'RigL')
print(sweep_id)
import os
# os.environ['CUDA_VISIBLE_DEVICES'] ="1"
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from absl import logging
import flax
import pandas as pd
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import optax
import tensorflow as tf
from tqdm import tqdm
# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.experimental.set_visible_devices([], "GPU")


logging.set_verbosity(logging.INFO)
import jaxpruner
import ml_collections


DATASET_PATH = "./data"
train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2))
DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2))
print("Data mean", DATA_MEANS)
print("Data std", DATA_STD)

def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - DATA_MEANS) / DATA_STD
    return img

# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


test_transform = image_to_numpy
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                      image_to_numpy
                                     ])
train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
_, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))


test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)
train_loader = data.DataLoader(train_set,
                               batch_size=128,
                               shuffle=True,
                               drop_last=True,
                               collate_fn=numpy_collate,
                               num_workers=8,
                               persistent_workers=True)
val_loader   = data.DataLoader(val_set,
                               batch_size=128*4,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate,
                               num_workers=4,
                               persistent_workers=True)
test_loader  = data.DataLoader(test_set,
                               batch_size=128*4,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate,
                               num_workers=4,
                               persistent_workers=True)

from resnet import ResNet20 as res20
from typing import Any
from collections import defaultdict
from flax.training import train_state, checkpoints
import time


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any



# config.learning_rate = 0.1
# config.momentum = 0.9
# config.batch_size = 256
# config.num_epochs = hparams['epochs']# 1 epoch is 468 steps for bs=128
# config = ml_collections.ConfigDict()
# config.sparsity_config = ml_collections.ConfigDict()
# config.sparsity_config.algorithm = 'rigl'
# config.sparsity_config.update_freq = 10
# config.sparsity_config.update_end_step = 1000
# config.sparsity_config.update_start_step = 200
# config.sparsity_config.sparsity = 0.70
# config.sparsity_config.dist_type = 'erk'


class TrainerModule:

    def __init__(self,
                 model_name : str,
                 model_class : nn.Module,
                 model_hparams : dict,
                 optimizer_name : str,
                 optimizer_hparams : dict,
                 exmp_imgs : Any,
                 seed=42):
        """
        Module for summarizing all training functionalities for classification on CIFAR10.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_hparams - Hyperparameters of the optimizer, including learning rate as 'lr'
            exmp_imgs - Example imgs, used as input to initialize the model
            seed - Seed to use in the model initialization
        """
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = res20()
        # Prepare logging
        self.log_dir = os.path.join('./', self.model_name)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_imgs)

    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def calculate_loss(params, batch_stats, batch, train):
            imgs, labels = batch
            # Run model. During training, we need to update the BatchNorm statistics.
            outs = self.model.apply({'params': params, 'batch_stats': batch_stats},
                                    imgs,
                                    train=train,
                                    mutable=['batch_stats'] if train else False)
            logits, new_model_state = outs if train else (outs, None)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            preds = logits.argmax(axis = -1)
            # preds =[]
            return loss, (acc, preds, new_model_state)
        # Training function
        def train_step(state, batch):
            loss_fn = lambda params: calculate_loss(params, state.batch_stats, batch, train=True)
            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc, _, new_model_state = ret[0], *ret[1]
            # Update parameters and batch statistics
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
            return state, loss, acc
        # Eval function
        def eval_step(state, batch):
            # Return the accuracy for a single batch
            loss, (acc, preds, _) = calculate_loss(state.params, state.batch_stats, batch, train=False)
            return loss,acc,preds
        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self, exmp_imgs):
        # Initialize model
        init_rng = jax.random.PRNGKey(self.seed)
        variables = self.model.init(init_rng, exmp_imgs, train=True)
        self.init_params, self.init_batch_stats = variables['params'], variables['batch_stats']
        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        # Initialize learning rate schedule and optimizer
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{opt_class}"'
        # We decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.optimizer_hparams.pop('lr'),
            boundaries_and_scales=
                {int(num_steps_per_epoch*num_epochs*0.6): 0.1,
                 int(num_steps_per_epoch*num_epochs*0.85): 0.1}
        )
        # Clip gradients at max value, and evt. apply weight decay
        #transf = [optax.clip(1.0)]
        transf =[]
        sparsity_updater = jaxpruner.create_updater_from_config(self.config.sparsity_config)
        if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop('weight_decay')))
        optimizer = optax.chain(
            *transf,
            sparsity_updater.wrap_optax(opt_class(lr_schedule, **self.optimizer_hparams))
        )
        # Initialize training state
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=self.init_params if self.state is None else self.state.params,
                                       batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
                                       tx=optimizer)
        
        return sparsity_updater

    def train_model(self, train_loader, val_loader, seed, config, num_epochs=200):
        self.config = config
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        sparsity_updater = self.init_optimizer(num_epochs, len(train_loader))
        # Track best eval accuracy
        best_eval = 0.0
        ep_pred = []
        for epoch_idx in tqdm(range(1, num_epochs+1)):
            new_params = sparsity_updater.pre_forward_update(
                    self.state.params, self.state.opt_state[1])
            self.train_epoch(train_loader, sparsity_updater= sparsity_updater, epoch=epoch_idx)
            new_params = sparsity_updater.pre_forward_update(
                self.state.params, self.state.opt_state[1])
            self.state = self.state.replace(params=new_params)
            if epoch_idx % 1 == 0:
                eval_acc,preds = self.eval_model(val_loader)
                                # train_acc,preds, loss_train = self.eval_model(train_loader)
                # print(preds.shape)
                ep_pred.append(preds.reshape((preds.shape[0],1)))
                # metrics = {"val loss" : loss_val, "val accuracy" : 100*eval_acc}
                wandb.log({"val accuracy" : 100*eval_acc})
                # self.logger.add_scalar('val/acc', eval_acc, global_step=epoch_idx)
                if epoch_idx>0.5*num_epochs:
                    if eval_acc >= best_eval:
                        best_eval = eval_acc
                        print("current_best " , best_eval)
                        self.save_model(step=epoch_idx)
                    # self.logger.flush()
        preds = np.concatenate(ep_pred,axis = 1)
        return preds

    def train_epoch(self, train_loader, epoch, sparsity_updater):
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(list)
        is_ste = isinstance(sparsity_updater, (jaxpruner.SteMagnitudePruning,
                                         jaxpruner.SteRandomPruning))
        pre_op = jax.jit(sparsity_updater.pre_forward_update)
        i = 0
        for batch in train_loader:
            i+=1
            # new_params = pre_op(self.state.params, self.state.opt_state[1])
            # self.state = self.state.replace(params=new_params)
            self.state, loss, acc = self.train_step(self.state, batch)
            post_params = sparsity_updater.post_gradient_update(
                self.state.params, self.state.opt_state[1])
            self.state = self.state.replace(params=post_params)
            # if i % 100 == 0:
            #     if is_ste:
            #         print(jaxpruner.summarize_sparsity(
            #         new_params, only_total_sparsity=True))
            #     else:
            #         print("Non ste ",jaxpruner.summarize_sparsity(
            #         self.state.params, only_total_sparsity=True))
        #     metrics['loss'].append(loss)
        #     metrics['acc'].append(acc)
        # # for key in metrics:
        #     avg_val = np.stack(jax.device_get(metrics[key])).mean()
        #     self.logger.add_scalar('train/'+key, avg_val, global_step=epoch)

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg loss
        correct_class, count = 0, 0
        preds_list = []
        
    
        for batch in data_loader:
            
            loss,acc,preds = self.eval_step(self.state, batch)
            wandb.log({'val_loss' : loss})
            # print(preds.shape)
            preds_list.append(preds)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        
        preds = np.concatenate(preds_list,axis = 0)
        return eval_acc, preds

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.state.params,
                                            'batch_stats': self.state.batch_stats},
                                    step=step,
                                   overwrite=True)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join('./', f'{self.model_name}.ckpt'), target=None)
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=state_dict['params'],
                                       batch_stats=state_dict['batch_stats'],
                                       tx=self.state.tx if self.state else optax.sgd(0.1)   # Default optimizer
                                      )

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join('./', f'{self.model_name}.ckpt'))
    


def train_classifier(*args, sparsity, seed, num_epochs=200, **kwargs):
    config = ml_collections.ConfigDict()
    config.sparsity_config = ml_collections.ConfigDict()
    config.sparsity_config.algorithm = 'rigl'
    config.sparsity_config.update_freq = 100
    config.sparsity_config.update_end_step = 58650
    config.sparsity_config.update_start_step = 1000
    config.sparsity_config.sparsity = sparsity
    config.sparsity_config.dist_type = 'erk'
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(*args, **kwargs)
    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        preds  = trainer.train_model(train_loader, val_loader, config=config, seed=seed, num_epochs=num_epochs)
        trainer.load_model()
        preds =  pd.DataFrame(preds)
        name = 'resnet'
        preds.to_csv(f'{name}_{sparsity}_{seed}.csv', index = False)
    else:
        trainer.load_model(pretrained=True)
    # Test trained model
    val_acc,preds = trainer.eval_model(val_loader)
    test_acc,preds = trainer.eval_model(test_loader)
    return trainer, {'val': val_acc, 'test': test_acc}


# resnet_trainer, resnet_results = train_classifier(model_name="ResNet",
#                                                   model_class=None,
#                                                   model_hparams={"num_classes": 10,
#                                                                  "c_hidden": (16, 32, 64),
#                                                                  "num_blocks": (3, 3, 3),
#                                                                  "act_fn": nn.relu,
#                                                                  "block_class": None},
#                                                   optimizer_name="SGD",
#                                                   optimizer_hparams={"lr": 0.1,
#                                                                      "momentum":0.9,
#                                                                      "weight_decay": 1e-4},
#                                                   exmp_imgs=jax.device_put(
#                                                       next(iter(train_loader))[0]),
#                                                   sparsity=0.5,
#                                                   seed =  42,
#                                                   num_epochs=200)

def main_trainer():
    wandb.init (name = 'RigL_grid', entity='ucalgary')
    resnet_trainer, resnet_results = train_classifier(model_name="ResNet",
                                                  model_class=None,
                                                  model_hparams={"num_classes": 10,
                                                                 "c_hidden": (16, 32, 64),
                                                                 "num_blocks": (3, 3, 3),
                                                                 "act_fn": nn.relu,
                                                                 "block_class": None},
                                                  optimizer_name="SGD",
                                                  optimizer_hparams={"lr": 0.1,
                                                                     "momentum":0.9,
                                                                     "weight_decay": 1e-4},
                                                  exmp_imgs=jax.device_put(
                                                      next(iter(train_loader))[0]),
                                                  sparsity=wandb.config.sparsity,
                                                  seed = wandb.config.seed,
                                                  num_epochs=200)
    
# wandb.init (name = 'RigL_grid', entity='ucalgary')


# resnet_trainer, resnet_results = train_classifier(model_name="ResNet",
#                                                 model_class=None,
#                                                 model_hparams={"num_classes": 10,
#                                                                 "c_hidden": (16, 32, 64),
#                                                                 "num_blocks": (3, 3, 3),
#                                                                 "act_fn": nn.relu,
#                                                                 "block_class": None},
#                                                 optimizer_name="SGD",
#                                                 optimizer_hparams={"lr": 0.1,
#                                                                     "momentum":0.9,
#                                                                     "weight_decay": 1e-4},
#                                                 exmp_imgs=jax.device_put(
#                                                     next(iter(train_loader))[0]),
#                                                 sparsity=0.5,
#                                                 seed = 42,
#                                                 num_epochs=3)


wandb.agent(sweep_id= 'cagkh4ya', function=main_trainer)
# main_trainer()
