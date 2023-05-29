from fl_experiment import load_partition_fed_experiment

def load_data(args):
    dataset_name = args.dataset.lower()
    if dataset_name in ["fl-experiment", "fl_experiment"]:
        dataset = load_partition_fed_experiment(args)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")
    return dataset
