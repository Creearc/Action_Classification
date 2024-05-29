import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics


ANNOTATIONS_PATH = 'datasets/action_classification/'
DATASET_TRAIN = '{}train_data_1_in.data'.format(ANNOTATIONS_PATH)
DATASET_VAL = '{}val_data_1_in.data'.format(ANNOTATIONS_PATH)
actions = ['crouch', 'walk', 'jump', 'stand', 'death', 'reload', 'fire']

num_keypoints = 12
num_coords = 3
num_channels = 1
num_classes = 7
examples = 108

EPOCHES = 1000
BATCH = 1024


class ActionClassifyNet(nn.Module):
    def __init__(self, in_channels=2, grid_height=5, grid_width=10):
        super(ActionClassifyNet, self).__init__()

        # Conv2d layer expects input shape (batch_size, num_channels, grid_height, grid_width)
        # For our example, num_channels = 2, because we have x and y as separate channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # Derive the size of the flattened features after convolution and pooling (if any)
        conv_output_size = 64 * (grid_width - 1) * (grid_height - 1)  # This depends on how you structure your convolutions and any pooling layers.

        self.fc1 = nn.Linear(conv_output_size, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Initialize weights and biases
        self._initialize_weights()

    # Weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1) # Flatten the tensor while keeping batch size the same
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def calculate_f1(preds, labels):
    # Convert to numpy arrays
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    # Calculate the F1 score
    f1 = f1_score(labels, preds.argmax(axis=1), average='weighted')
    return f1


def read_data(file_path):
    with open(file_path, 'rb') as file: 
        data = pickle.load(file)

    keypoints_data = torch.tensor(data[0], dtype=torch.float32)
    labels_data = torch.tensor(data[1], dtype=torch.int64)
    return keypoints_data, labels_data


# Instantiate the model
model = ActionClassifyNet(in_channels=num_channels,
                          grid_height=num_coords,
                          grid_width=num_keypoints)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Adding a learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                 factor=0.1,
                                                 threshold=0.0001,
                                                 cooldown=5,
                                                 verbose=True)


# Create a dataset and dataloader
keypoints_data, labels = read_data(DATASET_TRAIN)
print(keypoints_data.shape, labels.shape)
dataset = TensorDataset(keypoints_data, labels)
train_data = DataLoader(dataset, batch_size=BATCH, shuffle=True)

# Create a dataset and dataloader
keypoints_data, labels = read_data(DATASET_VAL)
dataset = TensorDataset(keypoints_data, labels)
val_data = DataLoader(dataset, batch_size=BATCH, shuffle=True)

result_val = [[], []]
graph_data = dict()
for field in ['loss_train', 'F1_train', 'loss_val', 'F1_val']:
    graph_data[field] = []

# Training loop
for epoch in range(EPOCHES):  # number of epochs
    model.train()  # Set the model to training mode
    train_epoch_loss = 0
    train_f1_scores = []  # List to store F1 scores for each batch
    
    for i, data in enumerate(train_data, 0):
        inputs, labels = data
        batch_size = labels.shape[0]
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Calculate F1 score for this batch and append to list
        batch_f1 = calculate_f1(outputs, labels)
        train_f1_scores.append(batch_f1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()        

    # Step the learning rate scheduler
    scheduler.step(loss)

    val_epoch_loss = 0
    val_f1_scores = []  # List to store F1 scores for each batch
    
    with torch.no_grad():    
        model.eval()
        for i, data in enumerate(val_data, 0):
            inputs, labels = data
        
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate F1 score for this batch and append to list
            batch_f1 = calculate_f1(outputs, labels)
            val_f1_scores.append(batch_f1)

            val_epoch_loss += loss.item()

            if epoch == EPOCHES - 1:
                result_val[0] += outputs.argmax(axis=1)
                result_val[1] += labels

    print(f'{epoch + 1}')    
    train_epoch_loss = train_epoch_loss / len(train_data)
    train_f1_scores = sum(train_f1_scores) / len(train_f1_scores)  # Average F1 score for the epoch
    print(f'Train | Average Loss: {train_epoch_loss:.3f}, Average F1: {train_f1_scores:.3f}')
    val_epoch_loss = val_epoch_loss / len(val_data)
    val_f1_scores = sum(val_f1_scores) / len(val_f1_scores)  # Average F1 score for the epoch
    print(f'Val  | Average Loss: {val_epoch_loss:.3f}, Average F1: {val_f1_scores:.3f}')

    graph_data['loss_train'].append(train_epoch_loss)
    graph_data['F1_train'].append(train_f1_scores)
    graph_data['loss_val'].append(val_epoch_loss)
    graph_data['F1_val'].append(val_f1_scores)

print('Done!')

plt.figure(figsize=(5,3)) 
plt.plot([i for i in range(1, EPOCHES + 1)], graph_data['F1_train'], label="f1_train")
plt.plot([i for i in range(1, EPOCHES + 1)], graph_data['F1_val'], label="f1_val")

        
plt.xlabel('Эпохи обучения') 
plt.ylabel('Точность') 

plt.legend() 
plt.savefig('graphs.jpg', bbox_inches='tight')
plt.clf()

confusion_matrix = metrics.confusion_matrix(result_val[0], result_val[1])

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = actions)

cm_display.plot()
plt.savefig('matrix.jpg', bbox_inches='tight')
plt.clf()
