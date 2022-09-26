def data_balance(dataloader):
    """prints bar graph of all classes"""
    import numpy as np
    import matplotlib.pyplot as plt

    classes = []
    for batch_idx, data in enumerate(dataloader, 0):
        x, y = data 
        classes.extend(y.tolist())

    #Calculating the unique classes and the respective counts and plotting them
    unique, counts = np.unique(classes, return_counts=True)
    names = list(dataloader.dataset.class_to_idx.keys())
    plt.bar(names, counts)
    plt.xlabel("Target Classes")
    plt.ylabel("Number of training instances")

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

def imshow_grid(training_dataloader):
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np
    images, labels = iter(training_dataloader).next()
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print(', '.join(f'{training_dataloader.dataset.classes[labels[j]]}' for j in range(training_dataloader.batch_size)))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    return images, labels


def help_print(imput=None):
    """ one line print website https://jagt.github.io/python-single-line-convert/ """
    
    if imput is None:
        print("""\nPytorch Outline\n\n1. Inputs\n2. Variables and Hyperparameters = 'hyper', 'device'\n3. Data Prep\n    Transforming and augmenting images = 'trans'\n    Normalization = mean_std()\n4. Getting Data\n    datasets = 'sets'\n    dataloader = 'loader'\n5. Training and validation loop = 'loop'\n""")

    elif imput == "loop":
        print("""\n#Defining the model hyper parameters\ncriterion = torch.nn.CrossEntropyLoss()\noptimizer = torch.optim.Adam(model.parameters(), lr=hyper['learning_rate'], weight_decay=hyper['weight_decay'])\n  \n#Training process begins\ntrain_loss_list = []\nfor epoch in range(hyper['epochs']):\n    print(f'Epoch {epoch+1}/{(hyper["epochs"])}:', end = ' ')\n    train_loss = 0\n      \n    #Iterating over the training dataset in batches\n    model.train()\n    for i, (images, labels) in enumerate(training_dataloader):\n          \n        #Extracting images and target labels for the batch being iterated\n        images = images.to(device)\n        labels = labels.to(device)\n  \n        #Calculating the model output and the cross entropy loss\n        outputs = model(images)\n        loss = criterion(outputs, labels)\n  \n        #Updating weights according to calculated loss\n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()\n        train_loss += loss.item()\n      \n    #Printing loss for each epoch\n    train_loss_list.append(train_loss/len(training_dataloader))\n    print(f"Training loss = {train_loss_list[-1]}")\n    \n    test_acc=0\n    model.eval()\n  \n    with torch.no_grad():\n        #Iterating over the training dataset in batches\n        for i, (images, labels) in enumerate(validation_dataloader):\n\n            images = images.to(device)\n            y_true = labels.to(device)\n\n            #Calculating outputs for the batch being iterated\n            outputs = model(images)\n\n            #Calculated prediction labels from models\n            _, y_pred = torch.max(outputs.data, 1)\n\n            #Comparing predicted and true labels\n            test_acc += (y_pred == y_true).sum().item()\n\n        print(f"Test set accuracy = {100 * test_acc / len(validation_dataset)} %")\n      \n#Plotting loss for all epochs\nplt.plot(range(1,hyper["epochs"]+1), train_loss_list)\nplt.xlabel("Number of epochs")\nplt.ylabel("Training loss")\n""")
    
    elif imput == "all":
        print("""\nimport torch\nimport torchvision\nfrom torchvision import datasets, transforms\nfrom torch.utils.data import DataLoader\n\nhyper = {\n    'learning_rate': .0001, \n    'weight_decay': 0.0, \n    'epochs': 1, \n    'batch_size': 128,\n    'num_workers': 4, \n    }\ndevice = 'cuda' if torch.cuda.is_available() else 'cpu'\n\ndef mean_std(data_file_path):\n    \"\"\"Returns the mean and standard deviation of the training data\"\"\"\n    try:\n        std = torch.load('data/std.pt')\n        mean = torch.load('data/mean.pt')\n    except:\n        print('No files found building them please wait ...')\n\n        if not os.path.exists('data'):\n            os.makedirs('data')\n\n        transformer = transforms.Compose([\n        transforms.Resize(256),\n        transforms.CenterCrop(224),\n        transforms.ToTensor()\n        ])\n\n        loader = training_dataloader = DataLoader(\n            datasets.ImageFolder(data_file_path, transform = transformer),\n            shuffle = True,\n        )\n\n        batch_mean = torch.zeros(3)\n        batch_mean_sqrd = torch.zeros(3)\n\n        for batch_data, _ in loader:\n            batch_mean += batch_data.mean(dim=(0, 2, 3)) # E[batch_i] \n            batch_mean_sqrd += (batch_data ** 2).mean(dim=(0, 2, 3)) #  E[batch_i**2]\n\n        mean = batch_mean / len(loader)\n        torch.save(mean, 'data/mean.pt')\n\n        var = (batch_mean_sqrd / len(loader)) - (mean ** 2)\n        std = var ** 0.5\n        torch.save(std, 'data/std.pt')\n    \n    print(f'mean & std: {mean}, {std}')    \n    \n    return mean, std\n\nmean, std = mean_std('cat-dog-panda/training')\ntransformer = transforms.Compose([\n    transforms.Resize(256),\n    transforms.CenterCrop(224),\n    transforms.ToTensor(),\n    transforms.Normalize(mean, std)\n])\n\ntraining_dataset = datasets.ImageFolder('cat-dog-panda/training', transform=transformer)\nvalidation_dataset = datasets.ImageFolder('cat-dog-panda/validation', transform=transformer)\n\ntraining_dataloader = DataLoader(\n    training_dataset,\n    shuffle=True,\n    num_workers=hyper['num_workers'],\n    batch_size=hyper['batch_size']\n)\nvalidation_dataloader = DataLoader(\n    validation_dataset,\n    shuffle=False,\n    num_workers=hyper['num_workers'],\n    batch_size=hyper['batch_size']\n)\n\nclasses = validation_dataset.class_to_idx\nclasses\n""")
    
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
    
    elif imput == "model":
        print("""\nclass CNN(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.model = torch.nn.Sequential(\n             #Input = 3 x 224 x 224, Output =  x 224 x 224\n            torch.nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1), \n            torch.nn.ReLU(),\n            #Input = 64 x 224 x 224, Output = 64 x 112 x 112\n            torch.nn.MaxPool2d(kernel_size=2),\n            \n            torch.nn.Flatten(),\n            torch.nn.Linear(64*112*112, 512),\n            torch.nn.ReLU(),\n            torch.nn.Linear(512, 3)\n        )\n  \n    def forward(self, x):\n        return self.model(x)\n    \nmodel = CNN().to(device)\n""")
    # elif imput == "":
    # elif imput == "":
    else:
        print("Not an option")

