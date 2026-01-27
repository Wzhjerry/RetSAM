from importlib import import_module


def load_dataset(args, train=True):
    print("=> Creating dataset {}, fold {}.".format(args.dataset, args.fold))
    m = import_module("datasets." + args.dataset.lower())
    
    # Try to get collate_fn if the dataset module provides it
    result = m.load_dataset(args, train)
    
    if train:
        # training mode
        train_dataset = val_dataset = test_dataset = collate_fn = None
        if len(result) == 4:
            train_dataset, val_dataset, test_dataset, collate_fn = result
        elif len(result) == 3:
            # If third element is a Dataset, treat it as test_dataset; otherwise collate_fn
            if hasattr(result[2], "__len__") and not callable(result[2]):
                train_dataset, val_dataset, test_dataset = result
            else:
                train_dataset, val_dataset, collate_fn = result
        else:
            train_dataset, val_dataset = result

        # Return test_dataset only if provided (ignored by trainer otherwise)
        if test_dataset is not None and collate_fn is not None:
            return train_dataset, val_dataset, test_dataset, collate_fn
        if test_dataset is not None:
            return train_dataset, val_dataset, test_dataset
        if collate_fn is not None:
            return train_dataset, val_dataset, collate_fn
        return train_dataset, val_dataset
    else:
        # testing mode
        if len(result) == 2:  # dataset returns collate_fn
            test_dataset, collate_fn = result
            return test_dataset, collate_fn
        else:  # legacy datasets without collate_fn
            test_dataset = result
            return test_dataset
