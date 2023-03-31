from libraries import *
from dataset import train_df, valid_df, ShoeDataset
#from config import CFG
#from efficientnet_pytorch import EfficientNet








# Load configuration file
# Load the YAML file
with open('/notebooks/algorithm/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


    """Set random seed for reproducibility."""

    torch.manual_seed(config['config']['seed_value'])
    torch.cuda.manual_seed_all(config['config']['seed_value'])
    np.random.seed(config['config']['seed_value'])
    random.seed(config['config']['seed_value'])

    # Set the variables
    epochs = config['config']['epochs']
    batch_size = config['config']['batch_size']
    learning_rate = config['config']['learning_rate']
    optimizer = config['config']['optimizer']
    seed_value = config['config']['seed_value']
    image_size = config['config']['image_size']
    classes = config['config']['classes']
    wd = config['config']['wd']
    test_size = config['config']['test_size']
    architecture = config['config']['architecture']
    dataset = config['config']['dataset']   



    # Set device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} device.")

    # Initialize Weights and Biases
    wandb.init(project=config['project'], entity=config['entity'], config=config, job_type='Train')



    # Define data transforms
    data_transform = {
        "train":transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomChoice([
                        transforms.Pad(padding=10),
                        transforms.CenterCrop(480),
                        transforms.RandomRotation(20),
                        transforms.CenterCrop((576,432)),
                        transforms.ColorJitter(
                            brightness=0.1,
                            contrast=0.1, 
                            saturation=0.1,
                            hue=0.1
                        )
                    ]),
                transforms.Resize((config['config']['image_size'], config['config']['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        "val": transforms.Compose([
                transforms.ToPILImage(),   
                transforms.Resize((config['config']['image_size'], config['config']['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])
    }

    # Load datasets
    train_dataset = ShoeDataset(df=train_df, transform=data_transform["train"])
    val_dataset = ShoeDataset(df=valid_df, transform=data_transform["val"])

    # Set number of workers
    nw = min([os.cpu_count(), config['config']['batch_size'] if config['config']['batch_size'] > 1 else 0, 8])

    # Define data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['config']['batch_size'],
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config['config']['batch_size'],
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    #model = models.resnet34(pretrained=True) efficientnet_v2_m
    #model = EfficientNet.from_pretrained('efficientnet-b3')
    #model.fc =  nn.Sequential(
    #                nn.Dropout(0.2),
    #                nn.ReLU(),
    #                nn.Linear(1000, 512),
    #                nn.Linear(124, args.num_classes))


    #model = timm.create_model(CONFIG.MODEL_NAME, pretrained=True)  
    #model.fc = nn.Sequential(
    #        nn.Dropout(0.2),
    #        nn.ReLU(),
    #        nn.Linear(64, args.num_classes)
    #    )

    #model.to(device)   
    #model = model.to(device)


    SHOENET = models.convnext_base(pretrained=True)
    SHOENET.head = nn.Sequential(
                     nn.Linear(64, config['config']['classes']))

    model = SHOENET.to(device)            

    #model = models.resnet34(pretrained=True)
    #model.fc = nn.Sequential(
    #    nn.Dropout(0.1),
    #    nn.Linear(model.fc.in_features, args.num_classes))
    # model = model.to(device)

    # Define model
    #SHOENET = timm.create_model(config['config']['architecture'], pretrained=True)
    #SHOENET.fc = nn.Sequential(
    #    nn.Dropout(0.1),
    #    nn.Linear(124, config['config']['classes'])
    #    )


    #model = SHOENET.to(device)

    # Define optimizer and learning rate scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['config']['learning_rate'], momentum=config['config']['momentum'])



def train_one_epoch(model, criterion, optimizer, train_loader, epoch):

    total_train_loss = 0
    total_train_correct = 0
    total_train_samples = 0

    model.train()

    scaler = amp.GradScaler()

    with tqdm(train_loader, unit='batch', leave=False) as pbar:
        pbar.set_description(f'training')
        for images, idxs in pbar:
            images = images.to(device, non_blocking=True)
            idxs = idxs.to(device, non_blocking=True)

            # Forward pass
            with amp.autocast():
                output = model(images)

                # Compute loss
                loss = criterion(output, idxs)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Compute accuracy
            with amp.autocast(enabled=False):
                preds = torch.argmax(output, dim=1)
                total_train_correct += torch.sum(preds == idxs)
                total_train_samples += idxs.size(0)

            total_train_loss += loss.item()

            pbar.set_postfix(Epoch=epoch, Train_Loss=total_train_loss / (pbar.n + 1))

    train_loss = total_train_loss / len(train_loader.dataset)
    train_acc = total_train_correct / total_train_samples

    #print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')

    return train_loss, train_acc




def evaluate(model, data_loader, device, epoch):
    criterion = torch.nn.CrossEntropyLoss()
    total_val_loss = 0
    total_val_correct = 0
    total_val_samples = 0

    model.eval()
    with torch.no_grad(), tqdm(data_loader, unit='batch', leave=False) as pbar:
        pbar.set_description(f'Validating')
        for images, idxs in pbar:
            images = images.to(device, non_blocking=True)
            idxs = idxs.to(device, non_blocking=True)

            # Forward pass
            output = model(images)
            loss = criterion(output, idxs)
            total_val_loss += loss.item()

            # Compute accuracy
            preds = torch.argmax(output, dim=1)
            total_val_correct += torch.sum(preds == idxs)
            total_val_samples += idxs.size(0)

            pbar.set_postfix(Epoch=epoch, Val_Loss=total_val_loss / (pbar.n + 1))

    val_loss = total_val_loss / len(data_loader.dataset)
    val_acc = total_val_correct / total_val_samples

    #print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

    return val_loss, val_acc




# Define the path where the checkpoint will be saved
#checkpoint_path = "/notebooks/ONNX/checkpoint/best_model.pth"
#checkpoint_path = '/path/to/checkpoint.pth'
checkpoint_path = ""
#checkpoint_pth = "/notebooks/ONNX/oonx/"
    
# Check if the checkpoint file exists
if os.path.exists(checkpoint_path) == "":
    # Load the checkpoint if it exists
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f'Resuming training from epoch {start_epoch}')
else:
    # Start training from scratch if checkpoint does not exist
    start_epoch = 1
    best_val_loss = float('inf')
    print('Starting training from scratch')       


    
def write_checkpoint(model, optimizer, epoch, best_val_loss):
    state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'best_val_loss': best_val_loss}
    filename = '/notebooks/ONNX/bst_mo'
    #torch.save(state, filename + f'CP_epoch{epoch + 1}.pth'
    torch.save(state, filename + f'del_.pth')
   



def train_model():
    # Train and evaluate model
    best_loss = 1000000000

    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in range(config['config']['epochs']):
        train_loss, train_acc = train_one_epoch(model=model, 
                                     criterion=criterion, 
                                     optimizer=optimizer, 
                                     train_loader=train_loader,
                                     epoch=epoch)

         # validate
        val_loss, val_acc = evaluate(model=model,
                             data_loader=val_loader,
                             device=device,
                             epoch=epoch)
        # Log the metrics
        wandb.log({"epoch": epoch})
        wandb.log({"Train Loss": train_loss})
        wandb.log({"Train Accuracy": train_acc})
        wandb.log({"Valid Loss": val_loss})
        wandb.log({"Valid Accuracy": val_acc})

        # Save the model if validation loss is the best seen so far
        if val_loss < best_loss:
            best_loss = val_loss
            write_checkpoint(model, optimizer, epoch, best_loss)
            
            

        # Print training and validation losses for the epoch
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Valid Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')



        
        



                                                
if __name__ == "__main__":
    wandb.login()
    train_model()
