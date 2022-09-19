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
        print("""\nPytorch Outline\n\n1. Inputs\n2. Variables and Hyperparameters = 'hyper', 'device'\n3. Data Prep\n    Transforming and augmenting images = 'trans'\n    Normalization = mean_std()\n4. Getting Data\n    datasets = 'sets'\n    dataloader = 'loader'\n5. Model\n""")
        
    elif imput == "hyper":
        print("""\nhyper = {\n    'learning_rate': .0001, \n    'weight_decay': 0.0, \n    'epochs': 1, \n    'batch_size': 128,\n    'num_workers': 4, \n    }\n""")

    elif imput == "device":
        print("device = 'cuda' if torch.cuda.is_available() else 'cpu'")
    
    elif imput == "trans":
        print("""\ntransformer = transforms.Compose([\n    transforms.Resize(256),\n    transforms.CenterCrop(224),\n    transforms.ToTensor(),\n    transforms.Normalize(mean, std)\n])\n""")
        
    elif imput == "loader":
        print("""\ntraining_dataloader = DataLoader(\n    training_dataset,\n    batch_size=hyper['batch_size'],\n    num_workers=hyper['num_workers'],\n    shuffle=True\n)\nValidataion_dataloader = DataLoader(\n    training_dataset,\n    batch_size=hyper['batch_size'],\n    num_workers=hyper['num_workers'],\n    shuffle=False\n)\n""")
    
    elif imput == "set":
        print("""\ntraining_dataset = datasets.ImageFolder('cat-dog-panda/training', transform=transformer)\nvalidation_dataset = datasets.ImageFolder('cat-dog-panda/validation', transform=transformer)\n""")
    # elif imput == "":
    # elif imput == "":
    # elif imput == "":
    # elif imput == "":
    else:
        print("Not an option")

