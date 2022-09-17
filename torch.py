def mean_std(data_file_path):
    """Returns the mean and standard deviation of the training data"""
    import torch
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader

    try:
        std = torch.load('std.pt')
        mean = torch.load('mean.pt')
    except:
        print('No files found building them please wait ...')
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
        torch.save(mean, 'mean.pt')

        var = (batch_mean_sqrd / len(loader)) - (mean ** 2)
        std = var ** 0.5
        torch.save(std, 'std.pt')
    
    print(f'mean & std: {mean}, {std}')    
    
    return mean, std