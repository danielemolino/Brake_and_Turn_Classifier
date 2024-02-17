"""

Al fine di girare il codice, è necessario creare sul drive una cartella chiamata Progetto CV, al cui interno vanno inseriti:
- Il Dataset Completo in formato zip (si è scelta questa strada perchè rende molto più rapido il caricamento delle immagini)
- Una cartella Codici con dentro tutti gli script che contengono le funzioni e le classi da importare
- I pesi dei modelli già addestrati

Iniziamo dapprima importando tutte le librerie necessarie, è necessario installare pyahocorasick poichè non è installata di default su Colab.
"""

from tqdm import tqdm
import torch
from data_formatting import DataProcessor
from data_loader import DataLoader
from Net import BrakeNet
from FeatureExtractionROI import FeatureExtractionROI
from sklearn.metrics import classification_report
from Training import read_img


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# Questa parte del codice implementa il testing dei modelli addestrati, attuando la classificazione finale
# Prende in ingresso entrambi i modelli, quello per la classificazione del freno e quello per la classificazione del segnale delle frecce
def final_test(data_loader, device, model_B, model_Turn):
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
                # Passiamo tutti gli input ai modelli
                outputs_B[j] = model_B(inputs[j])
                outputs_L[j] = model_Turn(left[j])
                outputs_R[j] = model_Turn(right[j])

            # Generiamoci le label complessive
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

            # Prendiamo la predizione per ciascun campione
            _, predicted_B = torch.max(outputs_B.data, 1)
            _, predicted_L = torch.max(outputs_L.data, 1)
            _, predicted_R = torch.max(outputs_R.data, 1)

            # costruiamo la label predetta
            for b, l, r in zip(predicted_B, predicted_L, predicted_R):
                b = b.cpu().tolist()
                l = l.cpu().tolist()
                r = r.cpu().tolist()
                preds.append([b, l, r])

    return preds, truth


fer = FeatureExtractionROI()

# Load dataset
data_loader = DataLoader(root_directory="../rear_signal_dataset_filtered")
data_loader.list_of_all_pictures = sorted(data_loader.list_of_all_pictures)
data_loader.filtered_list = sorted(data_loader.filtered_list)
data_processor = DataProcessor(data_loader.filtered_list)
data_list = data_processor.get_frame_list()

key_val_list = ['OOO', 'BOO', 'OLO', 'BLO', 'OOR', 'BOR']

data_list = [el for el in data_list if el.data_class in key_val_list]

train_sequences, val_sequences, test_sequences, data_list = data_processor.convert_to_train_test(
    data_list,
    # ci servono solo le sequenze di test
    n_train_sequences=0,
    val_ratio=0,
    max_sequence_size=16,
)

data_list[:] = map(read_img, tqdm(data_list))
test_loader = data_processor.sequence_map_to_torch_loader(test_sequences, 16, letter_idx='testing')

model_B = BrakeNet()
model_B = model_B.to(device)
model_B.load_state_dict(torch.load('drive/MyDrive/Progetto CV/Brake_Weights'))

model_Turn = BrakeNet()
model_Turn = model_Turn.to(device)
model_Turn.load_state_dict(torch.load('drive/MyDrive/Progetto CV/Turn_Weights'))


truth_dict = {'OOO': 0, 'BOO': 1, 'OLO': 2, 'BLO': 3, 'OOR': 4, 'BOR': 5, 'Other': 6}
preds_list = []
truth_list = []

preds, truth = final_test(test_loader, device, model_B, model_Turn)

# Trasformiamo predizioni e label da stringhe del tipo OOO, BOO... a interi, cosi da poter essere utilizzati per
# il classification report di scikit-learn
# avremo:
# - 0 = OOO
# - 1 = BOO
# - 2 = OLO
# - 3 = BLO
# - 4 = OOR
# - 5 = BOR
# - 6 = Eventuale classificazione in classi non facenti parti delle sei a cui siamo interessati, come BLR od OLR
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

for t in truth:
    truth_list.append(truth_dict[t])

report = classification_report(truth_list, preds_list)
print(report)
