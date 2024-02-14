import random
import torch
import ahocorasick
from dataclasses import dataclass
from dataset import SequenceDataSet, CustomDataLoader

# Only the six classes considered for the task
classes = ['OOO', 'BOO', 'OLO', 'BLO', 'OOR', 'BOR']


class DataProcessor:
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def get_frame_list(self):
        # implementiamo l'algoritmo di Aho-Corasick per facilitare la ricerca della classe nel nome del file
        automaton = ahocorasick.Automaton()
        for name in classes:
            automaton.add_word(name, name)
        automaton.make_automaton()

        data_dict_list = []

        for element in self.raw_data:
            append_entry(data_dict_list, automaton, element)

        return data_dict_list

    @staticmethod
    def convert_to_train_test(data_dict_list, n_train_sequences=None, n_test_sequences=None, val_ratio=None, shuffle=True, max_sequence_size=None):
        # prendiamo i nomi di tutte le sequenze
        sequence_names = list(set(entry.name for entry in data_dict_list))

        all_sequences = {}
        for name in sequence_names:
            all_sequences[name] = sorted([e for e in data_dict_list if name == e.name], key=lambda e: e.name)

            if max_sequence_size:
                # prendiamo solo sequenze di lunghezza pari a max_sequence_size, se una folder non ha abbastanza frames
                # la eliminiamo, se ne ha di più le dividiamo in più sequenze
                sequence_length = len(all_sequences[name])
                if sequence_length < max_sequence_size:
                    del all_sequences[name]
                else:
                    extra = sequence_length % max_sequence_size
                    all_sequences[name] = all_sequences[name][:-extra]
                    num_sequences = len(all_sequences[name]) // max_sequence_size
                    for i in range(num_sequences):
                        # Create a new sequence name
                        new_name = f"{name}+{i+1}"
                        # Split the sequence and add it to all_sequences
                        all_sequences[new_name] = all_sequences[name][i * max_sequence_size:(i + 1) * max_sequence_size]
                    # erase the original sequence
                    del all_sequences[name]

        sequence_names = list(set(all_sequences.keys()))

        # dividiamo le sequenze in train e test a seconda della presenza della parola "test" nel nome
        train_sequence_names = [name for name in sequence_names if "test" not in name]
        test_sequence_names = [name for name in sequence_names if "test" in name]

        # se richiesto, mescoliamo le sequenze
        if shuffle:
            random.shuffle(test_sequence_names)
            random.shuffle(train_sequence_names)
        if n_train_sequences is not None:
            train_sequence_names = train_sequence_names[:n_train_sequences]
        if n_test_sequences is not None:
            test_sequence_names = test_sequence_names[:n_test_sequences]

        # Split train set into train - val
        train_val_split = int(len(train_sequence_names) * val_ratio)
        val_sequence_names = train_sequence_names[:train_val_split]
        train_sequence_names = train_sequence_names[train_val_split:]

        train_sequences = {k: v for k, v in all_sequences.items() if k in train_sequence_names}
        val_sequences = {k: v for k, v in all_sequences.items() if k in val_sequence_names}
        test_sequences = {k: v for k, v in all_sequences.items() if k in test_sequence_names}

        if n_test_sequences is not None or n_train_sequences is not None or max_sequence_size is not None:
            # Update the list of all frame entries
            data_dict_list = []
            for name, frame_list in train_sequences.items():
                data_dict_list += frame_list
            for name, frame_list in val_sequences.items():
                data_dict_list += frame_list
            for name, frame_list in test_sequences.items():
                data_dict_list += frame_list
        return train_sequences, val_sequences, test_sequences, data_dict_list

    @staticmethod
    # convertiamo le sequenze in un formato adatto per il DataLoader
    def sequence_map_to_torch_loader(sequences, batch_size, letter_idx):
        data_set = SequenceDataSet(sequences, letter_idx)
        data_loader = CustomDataLoader(data_set, batch_size=batch_size)
        return data_loader


def append_entry(list_to_append, automaton_instance, raw_data_point):
    new_entry = Frame(name=raw_data_point[1],
                      location=raw_data_point[0],
                      data_class=findit_with_ahocorasick_name(automaton_instance, raw_data_point[0]),
                      )

    list_to_append.append(new_entry)


def findit_with_ahocorasick_name(automaton, element):
    try:
        class_id = next(automaton.iter(element))[1]
        return class_id
    except StopIteration:
        return None


@dataclass
class Frame:
    name: str
    location: str
    data_class: str
    picture: torch.Tensor = None
