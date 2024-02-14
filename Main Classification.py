import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from data_formatting import DataProcessor
from data_loader import DataLoader
from Net import BrakeNet
from FeatureExtractionROI import FeatureExtractionROI
import torchvision.transforms as transforms
from sklearn.metrics import classification_report


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

fer = FeatureExtractionROI()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((227, 227))
])


def read_image_to_tensor(file_path: str):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converti da BGR a RGB
    img = cv2.resize(img, (227, 227))  # Resize the image to the same size
    img = torch.from_numpy(img)  # Converti l'array NumPy in un tensore PyTorch
    img = img.permute(2, 0, 1)  # Permuta le dimensioni da (H, W, C) a (C, H, W)
    return img


def read_img(data_list_element):
    data_list_element.picture = read_image_to_tensor(data_list_element.location)
    return data_list_element


# Load dataset
data_loader = DataLoader(root_directory="rear_signal_dataset_filtered")
data_loader.list_of_all_pictures = sorted(data_loader.list_of_all_pictures)
data_loader.filtered_list = sorted(data_loader.filtered_list)
data_processor = DataProcessor(data_loader.filtered_list)
data_list = data_processor.get_frame_list()

# Possiamo scegliere quali classi utilizzare
key_val_list = ['OLO', 'BLO', 'OOO', 'BOO', 'OOR', 'BOR']

data_list = [el for el in data_list if el.data_class in key_val_list]

train_sequences, val_sequences, test_sequences, data_list = data_processor.convert_to_train_test(
    data_list,
    val_ratio=0.2,
    max_sequence_size=16,
)

data_list[:] = map(read_img, tqdm(data_list))
train_loader = data_processor.sequence_map_to_torch_loader(train_sequences, 16, letter_idx=0)
val_loader = data_processor.sequence_map_to_torch_loader(val_sequences, 16, letter_idx=0)
test_loader = data_processor.sequence_map_to_torch_loader(test_sequences, 16, letter_idx=0)

label_dict = {'OOO': 0, 'BOO': 0, 'OLO': 0, 'BLO': 0, 'OOR': 0, 'BOR': 0}
for elem in data_list:
    label_dict[elem.data_class] += 1
print(label_dict)


def train(train_loader, optimizer, device, model, criterion, turn=False):
    avg_loss = 0
    correct = 0
    total = 0

    # Iterate through batches
    for i, data in enumerate(tqdm(train_loader)):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if turn == 'L':
            inputs, _ = fer.extract_roi_sequence(inputs)
            inputs = torch.stack(inputs)
        elif turn == 'R':
            _, inputs = fer.extract_roi_sequence(inputs)
            inputs = torch.stack(inputs)

        elif turn == 'both':
            left, right = fer.extract_roi_sequence(inputs)
            left = torch.stack(left)
            right = torch.stack(right)
            inputs = torch.cat((left, right), 0)
            labels = labels.reshape(labels.size()[0] * 2, 2)
        else:
            inputs = torch.stack(inputs)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Move data to target device
        inputs = inputs.float().to(device)
        labels = labels.to(device)

        outputs = torch.empty(len(inputs), 2).to(device)
        for j in range(len(inputs)):
            outputs[j] = model(inputs[j])

        # Forward + backward + optimize
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Keep track of loss and accuracy
        avg_loss += float(loss)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(labels, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return avg_loss / len(train_loader), 100 * correct / total


def test(data_loader, device, model, criterion, turn=False):
    avg_loss = 0
    correct = 0
    total = 0

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # Iterate through batches
        for i, data in enumerate(tqdm(data_loader)):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if turn == 'L':
                inputs, _ = fer.extract_roi_sequence(inputs)
                inputs = torch.stack(inputs)

            elif turn == 'R':
                _, inputs = fer.extract_roi_sequence(inputs)
                inputs = torch.stack(inputs)

            elif turn == 'both':
                left, right = fer.extract_roi_sequence(inputs)
                left = torch.stack(left)
                right = torch.stack(right)
                inputs = torch.cat((left, right), 0)
                labels = labels.reshape(labels.size()[0] * 2, 2)
            else:
                inputs = torch.stack(inputs)

            # Move data to target device
            labels = labels.to(device)
            inputs = inputs.float().to(device)

            outputs = torch.empty(len(inputs), 2).to(device)
            for j in range(len(inputs)):
                outputs[j] = model(inputs[j])

            # Forward pass
            loss = criterion(outputs, labels)

            # Keep track of loss and accuracy
            avg_loss += float(loss)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return avg_loss / len(data_loader), 100 * correct / total


model = BrakeNet()
model.to(device)

LR = 0.00001
weight_decay = 0.0001
patience = 5
EPOCHS = 30
use_val = True
save_weights = True
early_stopping = True

# Define the loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()  # nn.MSELoss()  # mean squared error

# Initialize early stopping variables
val_acc_best = 0
patience_cnt = 0

for epoch in range(EPOCHS):
    # Train on data
    train_loss, train_acc = train(train_loader, optimizer, device, model, criterion, turn=None)
    val_loss, val_acc = test(val_loader, device, model, criterion, turn=None)

    model_type = 'Both Turn'

    log_dict = {'Train_Loss_{}'.format(model_type): train_loss,
                'Val_Loss_{}'.format(model_type): val_loss,
                'Train_Accuracy_{}'.format(model_type): train_acc,
                'Val_Accuracy_{}'.format(model_type): val_acc}
    print(log_dict)

    if save_weights:
        torch.save(model.state_dict(), f'weights_{epoch}.pt')

    if early_stopping:
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            patience_cnt = 0
        else:
            if patience_cnt == 0:
                best_epoch = epoch
            patience_cnt += 1
            if patience_cnt == patience:
                break

print('\nTraining Finished.')

if early_stopping:
    checkpoint = torch.load(f'weights_{best_epoch}.pt', map_location=device)
    model.load_state_dict(checkpoint)

torch.save(model.state_dict(), 'drive/MyDrive/Progetto CV/Brake_Net')

model.load_state_dict(torch.load('drive/MyDrive/Progetto CV/Brake_Net'))

torch.cuda.empty_cache()

test_loss, test_acc = test(test_loader, device, model, criterion, turn=None)
model_type = 'Brake'

log_dict = {'Test_Loss_{}'.format(model_type): test_loss,
            'Test_Accuracy_{}'.format(model_type): test_acc}
print(log_dict)

print('\nTest Finished.')


def final_test(data_loader, device, model_B, model_L, model_R):
    preds = []
    truth = []

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # Iterate through batches
        for i, data in enumerate(tqdm(data_loader)):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            left, right = fer.extract_roi_sequence(inputs)
            left = torch.stack(left)
            right = torch.stack(right)
            inputs = torch.stack(inputs)

            # Move data to target device
            labels = labels.to(device)
            inputs = inputs.float().to(device)
            left = left.float().to(device)
            right = right.float().to(device)

            outputs_B = torch.empty(len(inputs), 2).to(device)
            outputs_L = torch.empty(len(left), 2).to(device)
            outputs_R = torch.empty(len(right), 2).to(device)

            for j in range(len(inputs)):
                outputs_B[j] = model_B(inputs[j])
                outputs_L[j] = model_L(left[j])
                outputs_R[j] = model_R(right[j])

            for t in labels:
                if t[0][0] == 1:
                    b = 'O'
                else:
                    b = 'B'
                if t[1][0] == 1:
                    l = 'O'
                else:
                    l = 'L'
                if t[2][0] == 1:
                    r = 'O'
                else:
                    r = 'R'
                label = b + l + r
                truth.append(label)

            # Keep track of loss and accuracy
            _, predicted_B = torch.max(outputs_B.data, 1)
            _, predicted_L = torch.max(outputs_L.data, 1)
            _, predicted_R = torch.max(outputs_R.data, 1)

            for b, l, r in zip(predicted_B, predicted_L, predicted_R):
                b = b.cpu().tolist()
                l = l.cpu().tolist()
                r = r.cpu().tolist()
                preds.append([b, l, r])

    return preds, truth


# Load dataset
data_loader = DataLoader(root_directory="../rear_signal_dataset_filtered")
data_loader.list_of_all_pictures = sorted(data_loader.list_of_all_pictures)
data_loader.filtered_list = sorted(data_loader.filtered_list)
data_processor = DataProcessor(data_loader.filtered_list)
data_list = data_processor.get_frame_list()

key_val_list = ['OOO', 'BOO', 'OLO', 'BLO', 'OOR', 'BOR']
# key_val_list = ['BOR']

data_list = [el for el in data_list if el.data_class in key_val_list]

train_sequences, val_sequences, test_sequences, data_list = data_processor.convert_to_train_test(
    data_list,
    n_train_sequences=0,
    n_test_sequences=16,
    val_ratio=0,
    max_sequence_size=16,
)

data_list[:] = map(read_img, tqdm(data_list))
test_loader = data_processor.sequence_map_to_torch_loader(test_sequences, 16, letter_idx='testing')

model_B = BrakeNet()
model_B = model_B.to(device)
# model_B.load_state_dict(torch.load('drive/MyDrive/Progetto CV/Brake_Net'))

model_L = BrakeNet()
model_L = model_L.to(device)
# model_L.load_state_dict(torch.load('drive/MyDrive/Progetto CV/Left_Turn_big'))

model_R = BrakeNet()
model_R = model_R.to(device)
# model_R.load_state_dict(torch.load('drive/MyDrive/Progetto CV/Left_Turn_big'))

truth_dict = {'OOO': 0, 'BOO': 1, 'OLO': 2, 'BLO': 3, 'OOR': 4, 'BOR': 5, 'Other': 6}
preds_list = []
truth_list = []

for i in range(len(test_sequences)):
    truth_list.append(truth_dict[data_list[0].data_class])

preds, truth = final_test(test_loader, device, model_B, model_L, model_R)

for e in preds:
    if e == [0, 0, 0]:
        preds_list.append(0)
    elif e == [1, 0, 0]:
        preds_list.append(1)
    elif e == [0, 1, 0]:
        preds_list.append(2)
    elif e == [1, 1, 0]:
        preds_list.append(3)
    elif e == [0, 0, 1]:
        preds_list.append(4)
    elif e == [1, 0, 1]:
        preds_list.append(5)
    else:
        preds_list.append(6)


report = classification_report(truth_list, preds_list)
print(report)
