import logging
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random

# class FedMLPNGDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, mode='train', transform=None):
#         self.mode = mode
#         self.transform = transform
#
#         if self.mode == 'train':
#             self.dataset = datasets.ImageFolder(root=root_dir + '/train',
#                                                 transform=self.transform)
#         elif self.mode == 'test':
#             self.dataset = datasets.ImageFolder(root=root_dir + '/test',
#                                                 transform=self.transform)
#
#     def __getitem__(self, index):
#         return self.dataset[index]
#
#     def __len__(self):
#         return len(self.dataset)


def load_data_global(data_dir, batch_size):
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'

    # Define transforms for image preprocessing
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

    # Create DataLoader for train set
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Create DataLoader for test set
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader

def distribute_data(train_loader, test_loader, num_clients):
    train_data_local_dict = {}
    test_data_local_dict = {}
    data_local_num_dict = {}

    # count the number of training and testing examples
    train_count = len(train_loader.dataset)
    test_count = len(test_loader.dataset)

    # calculate the minimum number of examples each client should have
    min_train_per_client = train_count // num_clients
    min_test_per_client = test_count // num_clients

    # calculate any remaining examples after dividing
    remainder_train = train_count % num_clients
    remainder_test = test_count % num_clients

    # create a list of indices for both datasets
    train_indices = list(range(train_count))
    test_indices = list(range(test_count))

    # shuffle the indices randomly
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    # initialize counters for each dataset
    train_counter = 0
    test_counter = 0

    # loop over each client
    for i in range(num_clients):
        # assign the minimum number of training examples to each client
        local_train_indices = train_indices[train_counter:train_counter+min_train_per_client]
        train_counter += min_train_per_client

        # assign the minimum number of testing examples to each client
        local_test_indices = test_indices[test_counter:test_counter+min_test_per_client]
        test_counter += min_test_per_client

        # if there are any remaining training examples, assign them to clients starting from the first one
        if remainder_train > 0:
            local_train_indices.append(train_indices[train_counter])
            train_counter += 1
            remainder_train -= 1

        # if there are any remaining testing examples, assign them to clients starting from the first one
        if remainder_test > 0:
            local_test_indices.append(test_indices[test_counter])
            test_counter += 1
            remainder_test -= 1

        # save the indices for each client in a dictionary
        train_data_local_dict[i] = local_train_indices
        test_data_local_dict[i] = local_test_indices

        # save the number of examples for each client in a dictionary
        data_local_num_dict[i] = {
            'train': len(local_train_indices),
            'test': len(local_test_indices)
        }

    return data_local_num_dict, train_data_local_dict, test_data_local_dict

def load_partition_fed_experiment(args):
    if args.download:
        from .data_downloader import main as download_main
        download_main(args.data_cache_dir)

    train_data_num = 0
    test_data_num = 0
    train_data_global = None
    test_data_global = None
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    nc = args.output_dim
    #nc=8
    #batch_size=32
    train_data_global, test_data_global = load_data_global(args.data_cache_dir, args.batch_size)
    train_data_num = len(train_data_global.dataset)
    test_data_num = len(test_data_global.datset)
    if args.worker_num == 2:
        data_local_num_dict, train_data_local_dict, test_data_local_dict = distribute_data(
            train_data_global, test_data_global, args.worker_num)
    else:
        raise NotImplementedError("Not support other client_number for now!")

    # logging.info(f"train_data_num: {train_data_num}")
    # logging.info(f"test_data_num: {test_data_num}")
    # logging.info(f"train_data_global: {train_data_global}")
    # logging.info(f"test_data_global: {test_data_global}")
    # logging.info(f"data_local_num_dict: {data_local_num_dict}")
    # logging.info(f"train_data_local_dict: {train_data_local_dict}")
    # logging.info(f"test_data_local_dict: {test_data_local_dict}")
    # logging.info(f"nc: {nc}")

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        nc,
    )

