from dataloaders.DatasetSelfCaptured import SelfCapturedDataset

def load_dataset(args):
    if args.dataset_name == "selfCaptured":
        dataset = SelfCapturedDataset(args)
    else:
        raise Exception("Dataset not found.")
    return dataset