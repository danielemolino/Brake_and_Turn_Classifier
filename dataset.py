import numpy as np
import torch
from tqdm import tqdm


# Classe per gestire le sequenze d'immagini
class SequenceDataSet:
    def __init__(self, sequences, letter_idx):
        # Indice della lettera per identificare la classe
        self.letter_idx = letter_idx

        # Creazione delle sequenze con le immagini e la classe corrispondente
        self.sequences = [
            {
                "sequence": torch.stack([e.picture for e in entry_list]),
                "class": self.class_to_tensor(entry_list[0].data_class),
            }
            for name, entry_list in tqdm(sequences.items())
        ]
        # Conversione delle sequenze in tuple
        self.sequence_tuples = [tuple(seq.values()) for seq in self.sequences]

    def __len__(self):
        # Restituisce la lunghezza del dataset
        return len(self.sequence_tuples)

    def __getitem__(self, idx):
        # Restituisce l'elemento all'indice specificato
        return self.sequence_tuples[idx]

    @staticmethod
    def collate_fn(data):
        # Funzione per raggruppare i dati
        return [el[0] for el in data], torch.stack([el[1] for el in data])

    def class_to_tensor(self, class_name):
        # Conversione della classe in un tensore
        # testing serve quando si fa inferenza sul modello già addestrato
        if self.letter_idx == 'testing':
            if class_name[0] == "O":
                brake = torch.Tensor([1, 0])
            else:
                brake = torch.Tensor([0, 1])
            if class_name[1] == "O":
                left = torch.Tensor([1, 0])
            else:
                left = torch.Tensor([0, 1])
            if class_name[2] == "O":
                right = torch.Tensor([1, 0])
            else:
                right = torch.Tensor([0, 1])
            return torch.stack([brake, left, right])

        if self.letter_idx == 'both':
          if class_name[1] == "O":
              left = torch.Tensor([1, 0])
          else:
              left = torch.Tensor([0, 1])
          if class_name[2] == "O":
              right = torch.Tensor([1, 0])
          else:
              right = torch.Tensor([0, 1])
          return torch.stack([left, right])

        if class_name[self.letter_idx] == "O":
            return torch.Tensor([1, 0])
        else:
            return torch.Tensor([0, 1])


# Classe per creare un DataLoader personalizzato
class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        # Inizializzazione del DataLoader con il dataset, la dimensione del batch e l'opzione di mescolamento
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = []
        self.iter_i = 0

    def __iter__(self):
        # Mescolamento degli indici se richiesto
        if self.shuffle:
            self.idx = np.random.permutation(len(self.dataset)).astype(int)
        else:
            self.idx = np.arange(len(self.dataset), dtype=int)
        self.iter_i = 0
        return self

    def __len__(self):
        # Restituisce la lunghezza del DataLoader
        return len(self.dataset) // self.batch_size + 1

    def __next__(self):
        # Restituisce il prossimo batch di dati
        if self.iter_i == len(self.dataset):
            raise StopIteration
        if self.iter_i + self.batch_size > len(self.dataset):
            indices = self.idx[self.iter_i:]
            self.iter_i = len(self.dataset)
        else:
            indices = self.idx[self.iter_i:self.iter_i + self.batch_size]
            self.iter_i += self.batch_size
        return self.dataset.collate_fn([self.dataset[i] for i in indices])