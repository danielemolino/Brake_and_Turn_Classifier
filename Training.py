"""

Al fine di girare il codice, è necessario creare sul drive una cartella chiamata Progetto CV, al cui interno vanno inseriti:
- Il Dataset Completo in formato zip (si è scelta questa strada perchè rende molto più rapido il caricamento delle immagini)
- Una cartella Codici con dentro tutti gli script che contengono le funzioni e le classi da importare
- I pesi dei modelli già addestrati

Iniziamo dapprima importando tutte le librerie necessarie, è necessario installare pyahocorasick poichè non è installata di default su Colab.
"""

import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from data_formatting import DataProcessor
from data_loader import DataLoader
from Net import BrakeNet
from FeatureExtractionROI import FeatureExtractionROI

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Creiamo un istanza della classe FeatureExtractionROI che ci servirà per estrarre le feature riguardo il funzionamento delle frecce
fer = FeatureExtractionROI()


# Funzione con tutti i passaggi per caricare correttamente un immagine
def read_image_to_tensor(file_path: str):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converti da BGR a RGB
    img = cv2.resize(img, (227, 227))  # Resize dell'immagine alle dimensioni necessarie per il modello
    img = torch.from_numpy(img)  # Converti l'array NumPy in un tensore PyTorch
    img = img.permute(2, 0, 1)  # Permuta le dimensioni da (H, W, C) a (C, H, W)
    return img


def read_img(data_list_element):
    data_list_element.picture = read_image_to_tensor(data_list_element.location)
    return data_list_element


# Load dataset
data_loader = DataLoader(root_directory="../rear_signal_dataset_filtered")
data_loader.list_of_all_pictures = sorted(data_loader.list_of_all_pictures)
data_loader.filtered_list = sorted(data_loader.filtered_list)
data_processor = DataProcessor(data_loader.filtered_list)
data_list = data_processor.get_frame_list()

# Selezioniamo le classi di cui vogliamo estrarre i dati
key_val_list = ['OLO', 'BLO', 'OOO', 'BOO', 'OOR', 'BOR']

# Selezioniamo solo i campioni delle classi che abbiamo scelto
data_list = [el for el in data_list if el.data_class in key_val_list]

train_sequences, val_sequences, test_sequences, data_list = data_processor.convert_to_train_test(
    data_list,
    val_ratio=0.2,  # il 20% del Training Set viene utilizzato come Validation
    max_sequence_size=16,  # Consideriamo sequenze da 16 frame
)

# Con letter_idx possiamo scegliere che task intraprendere:
# 0 - Classificazione Freno
# 1 - Classificazione Freccia Sinistra
# 2 - Classificazione Freccia Destra
# both - Classificazione Freccia Destra e Sinistra
# testing - Classificazione finale

data_list[:] = map(read_img, tqdm(data_list))
train_loader = data_processor.sequence_map_to_torch_loader(train_sequences, 16, letter_idx=0)
val_loader = data_processor.sequence_map_to_torch_loader(val_sequences, 16, letter_idx=0)
test_loader = data_processor.sequence_map_to_torch_loader(test_sequences, 16, letter_idx=0)

# Printiamo quante immagini stiamo caricando per ogni classe

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
        # Se l'attributo turn assume qualche valore, vuol dire che dobbiamo attuare la feature extraction per le ROI sinistre e destre
        if turn == 'L':
            inputs, _ = fer.extract_roi_sequence(inputs)
            inputs = torch.stack(inputs)
        elif turn == 'R':
            _, inputs = fer.extract_roi_sequence(inputs)
            inputs = torch.stack(inputs)

        # In particolare se assume il valore both, andiamo a estrarre sia le roi sinistre che quelle destre, che entreranno come input indipendenti al modello
        elif turn == 'both':
            left, right = fer.extract_roi_sequence(inputs)
            left = torch.stack(left)
            right = torch.stack(right)
            inputs = torch.cat((left, right), 0)
            labels = labels.reshape(labels.size()[0] * 2, 2)
        else:
            inputs = torch.stack(inputs)

        optimizer.zero_grad()

        # Spostiamo i dati sul device corretto
        inputs = inputs.float().to(device)
        labels = labels.to(device)

        # Iteriamo sulla batch
        outputs = torch.empty(len(inputs), 2).to(device)
        for j in range(len(inputs)):
            outputs[j] = model(inputs[j])

        # Forward + backward + optimize
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Teniamo traccia di Loss e Accuracy
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

            labels = labels.to(device)
            inputs = inputs.float().to(device)

            outputs = torch.empty(len(inputs), 2).to(device)
            for j in range(len(inputs)):
                outputs[j] = model(inputs[j])

            loss = criterion(outputs, labels)

            # Keep track of loss and accuracy
            avg_loss += float(loss)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return avg_loss / len(data_loader), 100 * correct / total


# Definiamo i parametri per la fase di Training
model = BrakeNet()
model.to(device)

LR = 0.000001
weight_decay = 0.0001
patience = 5
EPOCHS = 30
early_stopping = True  # Se utilizzare o meno early stopping
save_weights = True  # Se salvare i pesi a ogni epoca

# Define the loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()  # nn.MSELoss()  # mean squared error

# Initialize early stopping variables
val_acc_best = 0
patience_cnt = 0

for epoch in range(EPOCHS):
    # Trainiamo sui dati di training, e testiamo su quelli di validation
    # Va passato il parametri turn per il tipo di addestramento:
    # - Di default è None quindi non passa le batch alla funzione per l'estrazione delle ROI
    # - Se L/R estrae solo le roi left o le roi right
    # - Se Both le estrae entrambe, dandole come input indipendenti al modello
    train_loss, train_acc = train(train_loader, optimizer, device, model, criterion, turn='both')
    val_loss, val_acc = test(val_loader, device, model, criterion, turn='both')

    # Nome del modello
    model_type = 'Left Turn'

    log_dict = {'Train_Loss_{}'.format(model_type): train_loss,
                'Val_Loss_{}'.format(model_type): val_loss,
                'Train_Accuracy_{}'.format(model_type): train_acc,
                'Val_Accuracy_{}'.format(model_type): val_acc}
    print(log_dict)

    if save_weights:
        torch.save(model.state_dict(), f'weights_{epoch}.pt')

    if early_stopping:
        # controlliamo se abbiamo superato il valore di pazienza o meno
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

# Se abbiamo utilizzato l'Early Stopping, recuperiamo i pesi dell'epoca ottimale

if early_stopping:
    checkpoint = torch.load(f'weights_{best_epoch}.pt', map_location=device)
    model.load_state_dict(checkpoint)

# Se vogliamo possiamo salvare i pesi del modello
torch.save(model.state_dict(), f'{model_type}')

# Adesso testiamo sul test set
test_loss, test_acc = test(test_loader, device, model, criterion, turn=None)
model_type = 'Brake'

log_dict = {'Test_Loss_{}'.format(model_type): test_loss,
            'Test_Accuracy_{}'.format(model_type): test_acc}
print(log_dict)

print('\nTest Finished.')
