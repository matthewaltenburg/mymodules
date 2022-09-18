def mean_std(data_file_path):
    """Returns the mean and standard deviation of the training data"""
    import torch
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    import os

    try:
        std = torch.load('data/std.pt')
        mean = torch.load('data/mean.pt')
    except:
        print('No files found building them please wait ...')

        if not os.path.exists('data'):
            os.makedirs('data')

        transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])

        loader = training_dataloader = DataLoader(
            datasets.ImageFolder(data_file_path, transform = transformer),
            shuffle = True,
        )

        batch_mean = torch.zeros(3)
        batch_mean_sqrd = torch.zeros(3)

        for batch_data, _ in loader:
            batch_mean += batch_data.mean(dim=(0, 2, 3)) # E[batch_i] 
            batch_mean_sqrd += (batch_data ** 2).mean(dim=(0, 2, 3)) #  E[batch_i**2]

        mean = batch_mean / len(loader)
        torch.save(mean, 'data/mean.pt')

        var = (batch_mean_sqrd / len(loader)) - (mean ** 2)
        std = var ** 0.5
        torch.save(std, 'data/std.pt')
    
    print(f'mean & std: {mean}, {std}')    
    
    return mean, std


def help_print(imput=None):
    
    if imput is None:
        print("Outline\n\
'hyper' = hyperparam_help,\n\
'trans' = transform,\n\
'device' = cuda or cpu,\n\
    ")
    
    if imput == "hyper":
        print("hyper = {\n\
    'learning_rate': .0001, \n\
    'weight_decay': 0.0, \n\
    'epochs': 1, \n\
    'batch_size': 128,\n\
    'num_workers': 4, \n\
    }")

    if imput == "device":
        print("device = 'cuda' if torch.cuda.is_available() else 'cpu'")

