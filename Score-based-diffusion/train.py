# 1) VARYING THE NUMBER OF SAMPLING STEPS
# 2) VARYING THE UNCONDITIONAL TRAINING PROBABILITY
# 3) VARYING THE CLASSIFIER-FREE GUIDANCE STRENGTH

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from utility_funs import *
from utility_classes import *
from select_model import *
from prepare_dataset import myDataset
from loss_function import loss_fn

###################### Main ######################
# Name the experiment
experiment_id = "1e-2_newdataset_nonnegative_OneDUnet"
model_name = experiment_id.split('_')[-1]
print('Now running job: {}'.format(experiment_id))

# Work directory
wd = './experiments/{}'.format(experiment_id)
# Enable SummaryWriter
writer = SummaryWriter('{}/run'.format(wd))
# Show device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define training setup
# Load settings from previous run
# with open('config.json', 'r') as f:
#     config = json.load(f)

# About data
x_dim = 5
y_dim = 61
embed_dim = 256

# About training
n_epochs = 1000
batch_size = 1024
lr = 1e-3
num_cpu = 16
lr_scheduler_factor = 0.1
lr_scheduler_patience = 10
dropout_prob = 0.1

# About SDE
sigma =  25.0 #@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=device)

## Prepare datasets
train_set = myDataset('../Yun_simulation_data/train_y.csv', '../Yun_simulation_data/train_x.csv')
validation_set = myDataset('../Yun_simulation_data/val_y.csv', '../Yun_simulation_data/val_x.csv')
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_cpu)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_cpu)

# Initialize model, optimizer
score_model = model_initializer(model_name, x_dim, y_dim, marginal_prob_std_fn, embed_dim).to(device)
# model.load_state_dict(torch.load('./model/loss_29.447.pth')) # Keep training old models

# If multiple GPUs are available, use DataParallel
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs")
#     model = nn.DataParallel(model)

# Define optimizer and loss function
optimizer = Adam(score_model.parameters(), lr=lr)

# Define the learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_factor, patience=lr_scheduler_patience)

###################### Train ######################
previous_loss = 1e8

try:
    for epoch in range(n_epochs):
        score_model.train()
        avg_loss = 0.
        num_items = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            x = torch.log(x) ### log transform x  ###
            x = x.unsqueeze(1).float().to(device)
            y = y.float().to(device)
            loss = loss_fn(score_model, x, y, dropout_prob, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        train_loss = avg_loss / num_items

        score_model.eval()
        with torch.no_grad():
            avg_loss = 0.
            num_items = 0

            for i, (x, y) in enumerate(validation_loader, start = 1):
                x = torch.log(x) ### log transform x ###
                x = x.unsqueeze(1).float().to(device)
                y = y.float().to(device)
                loss = loss_fn(score_model, x, y, dropout_prob, marginal_prob_std_fn)

                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]

            validation_loss = avg_loss / num_items

        print('Epoch {}, Train Loss: {:.8f}, Validation Loss: {:.8f}'.format(epoch + 1, train_loss, validation_loss))

        # Step the scheduler
        scheduler.step(train_loss)
        print('Learning rate:', scheduler.optimizer.param_groups[0]['lr'])

        # Log the losses and learning rate
        # writer.add_scalar('Loss/train', train_loss, epoch)
        # writer.add_scalar('Loss/val', validation_loss, epoch)
        # writer.add_scalar('Learning_Rate', scheduler.optimizer.param_groups[0]['lr'], epoch)

        if train_loss < previous_loss and epoch > 1:
            previous_loss = train_loss
            model_file_name = '{}/epoch_{}_trainloss_{:.8f}_valloss_{:.8f}'.format(wd, epoch, train_loss, validation_loss)
            torch.save(score_model.state_dict(), '{}.pth'.format(model_file_name))
            save_config(model_file_name, model_name, x_dim, y_dim, embed_dim, lr, dropout_prob, epoch, lr_scheduler_factor, lr_scheduler_patience)

except KeyboardInterrupt:
# Save model
    model_file_name = '{}/epoch_{}_trainloss_{:.8f}_valloss_{:.8f}'.format(wd, epoch, train_loss, validation_loss)
    torch.save(score_model.state_dict(), '{}.pth'.format(model_file_name))
    save_config(model_file_name, model_name, x_dim, y_dim, embed_dim, lr, dropout_prob, epoch, lr_scheduler_factor, lr_scheduler_patience)

model_file_name = '{}/epoch_{}_trainloss_{:.8f}_valloss_{:.8f}'.format(wd, epoch, train_loss, validation_loss)
torch.save(score_model.state_dict(), '{}.pth'.format(model_file_name))
save_config(model_file_name, model_name, x_dim, y_dim, embed_dim, lr, dropout_prob, epoch, lr_scheduler_factor, lr_scheduler_patience)

