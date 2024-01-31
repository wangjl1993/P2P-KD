'''
TaICML incremental learning
Copyright (c) Jathushan Rajasegaran, 2019
'''
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Sampler
from torchvision import datasets, transforms


class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, shuffle):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if(self.shuffle):
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class IncrementalDataset:

    def __init__(
        self,
        dataset_name,
        root,
        order,
        workers=16,
        batch_size=128,
        class_per_task=10,
        memory_size=0,
        mu=1
    ):
        self._current_task = 0
        
        self.memory_size = memory_size
        self.mu = mu
        self.class_per_task = class_per_task
        self.batch_size = batch_size
        self.workers = workers
        self.sample_per_task_testing = {}
        self._setup_data(
            dataset_name,
            root,
            order,
            class_per_task=class_per_task,
        )

    @property
    def n_tasks(self):
        return len(self.increments)
    
    def get_same_index(self, target, label, memory=None):
        # 获取当前任务的训练数据(通过该任务数据的target获得)的indice  
        # 训练数据 = 当前任务数据 + memory
        label_indices = []
        label_targets = []

        for i in range(len(target)):
            if int(target[i]) in label:
                label_indices.append(i)
                label_targets.append(target[i])
        for_memory = (label_indices.copy(),label_targets.copy())
        
        if memory is not None:
            memory_indices, memory_targets = memory
            memory_indices2 = np.tile(memory_indices, (self.mu,))
            all_indices = np.concatenate([memory_indices2,label_indices]).astype(np.int64)
        else:
            all_indices = label_indices
            
        return all_indices, for_memory     
        # all_indices 该任务训练数据下标 当前任务数据+memory;
        # for_memory (当前任务数据下标, 标签)
    
    def get_same_index_test_chunk(self, target, label):
        label_indices = []
        label_targets = []
        
        np_target = np.array(target, dtype="int32")
        np_indices = np.array(list(range(len(target))), dtype="int32")

        for t in range(len(label)//self.class_per_task):
            task_idx = []
            for class_id in label[t*self.class_per_task: (t+1)*self.class_per_task]:
                idx = np.where(np_target==class_id)[0]
                task_idx.extend(list(idx.ravel()))
            task_idx = np.array(task_idx, dtype="int32")
            task_idx.ravel()
            random.shuffle(task_idx)

            label_indices.extend(list(np_indices[task_idx]))
            label_targets.extend(list(np_target[task_idx]))
            if(t not in self.sample_per_task_testing.keys()):
                self.sample_per_task_testing[t] = len(task_idx)
        label_indices = np.array(label_indices, dtype="int32")
        label_indices.ravel()
        return list(label_indices), label_targets
    

    def new_task(self, memory=None):
        
        print(self._current_task)
        print(self.increments)
        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])

        
        train_indices, for_memory = self.get_same_index(self.train_dataset.targets, list(range(min_class, max_class)), memory=memory)
        test_indices, _ = self.get_same_index_test_chunk(self.test_dataset.targets, list(range(max_class)))

        train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=False,num_workers=self.workers, sampler=SubsetRandomSampler(train_indices, True))
        test_data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=False,num_workers=self.workers, sampler=SubsetRandomSampler(test_indices, False))

        
        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": len(train_indices),
            "n_test_data": len(test_indices)
        }

        self._current_task += 1

        return task_info, train_data_loader, test_data_loader, for_memory
    
    
    def _setup_data(self, dataset_name, root, order, class_per_task=10):
        self.increments = []
        
        dataset = _get_dataset(dataset_name)

        train_path = Path(root) / "train"
        test_path = Path(root) / "test"
        train_dataset = dataset.base_dataset(root=str(train_path), transform=dataset.train_transforms)
        test_dataset = dataset.base_dataset(root=str(test_path), transform=dataset.test_transforms)


        for i,t in enumerate(train_dataset.targets):
                train_dataset.targets[i] = order[t]
        for i,t in enumerate(test_dataset.targets):
            test_dataset.targets[i] = order[t]

        self.increments = [class_per_task for _ in range(len(order) // class_per_task)]

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


    
    def get_memory(self, memory, for_memory, sess, seed=1):
        random.seed(seed)
        memory_per_task = self.memory_size // ((sess+1)*self.class_per_task)
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        mu = 1
        
        #update old memory
        if memory is not None:
            data_memory, targets_memory = memory
            data_memory = np.array(data_memory, dtype="int32")
            targets_memory = np.array(targets_memory, dtype="int32")
            for class_idx in range(self.class_per_task*(sess)):
                tmp_index = np.where(targets_memory==class_idx)[0]
                idx = np.random.choice(tmp_index, size=min(len(tmp_index), memory_per_task))
                self._data_memory = np.concatenate([self._data_memory, np.tile(data_memory[idx], (mu,))   ])
                self._targets_memory = np.concatenate([self._targets_memory, np.tile(targets_memory[idx], (mu,))    ])
                
                
        #add new classes to the memory
        new_indices, new_targets = for_memory

        new_indices = np.array(new_indices, dtype="int32")
        new_targets = np.array(new_targets, dtype="int32")
        for class_idx in range(self.class_per_task*(sess),self.class_per_task*(1+sess)):
            tmp_index = np.where(new_targets==class_idx)[0]
            idx = np.random.choice(tmp_index,size=min(len(tmp_index),memory_per_task))
            self._data_memory = np.concatenate([self._data_memory, np.tile(new_indices[idx],(mu,))   ])
            self._targets_memory = np.concatenate([self._targets_memory, np.tile(new_targets[idx],(mu,))    ])
            
        print(len(self._data_memory))
        return list(self._data_memory.astype("int32")), list(self._targets_memory.astype("int32"))


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name in ['mycifar30']:
        return MyCifar30
    elif dataset_name == 'animal_imagenet':
        return AnimalImagenet
    elif dataset_name in ['digit5', 'digit4']:
        return Digit5
    else:
        raise NotImplementedError(f"illegal dataset {dataset_name}.")


class DataHandler:
    base_dataset = datasets.ImageFolder


class AnimalImagenet(DataHandler):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

class MyCifar30(DataHandler):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.5071, 0.4867, 0.4408],
        #     std=[0.2675, 0.2565, 0.2761]
        # )
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.5071, 0.4867, 0.4408],
        #     std=[0.2675, 0.2565, 0.2761]
        # )
    ])

class Digit5(DataHandler):
    train_transforms = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
    ])