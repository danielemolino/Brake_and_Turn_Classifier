import ahocorasick
import os


# Definizione della classe DataLoader
class DataLoader:
    def __init__(self, root_directory):
        # Inizializzazione del DataLoader con la directory radice
        self.root_directory = root_directory
        # Ottenimento della lista di tutte le immagini
        self.list_of_all_pictures = get_list_of_pictures(self.root_directory)
        # Filtraggio della lista d'immagini
        self.filtered_list = filter_list(self.root_directory, self.list_of_all_pictures)

    def __call__(self, *args, **kwargs):
        # Restituzione della lista filtrata quando l'oggetto viene chiamato
        return self.filtered_list


# Funzione per filtrare la lista d'immagini
def filter_list(directory, list_of_pictures):
    # Lettura di tutte le cartelle nella directory radice
    files = os.listdir(directory)
    automaton = ahocorasick.Automaton()
    for name in files:
        automaton.add_word(name, name)

    # Creazione dell'automa
    automaton.make_automaton()
    # Ricerca di ogni elemento della lista d'immagini nell'automa
    output = [findit_with_ahocorasick(automaton, element)
              for element in list_of_pictures]
    # Rimozione degli elementi None dalla lista
    output = list(filter(None, output))
    return output


# Funzione per cercare un elemento nell'automa
def findit_with_ahocorasick(automaton, element):
    try:
        # Tentativo di ricerca dell'elemento nell'automa
        result = next(automaton.iter(element))
        index = result[0]
        entry = result[1]
        return element, entry
    except StopIteration:
        # Se l'elemento non è presente nell'automa, viene restituito None
        return None


# Funzione per ottenere la lista di tutte le immagini in una directory
def get_list_of_pictures(directory):
    picture_list = []
    # Scansione di tutti i file nella directory e nelle sue sottodirectory
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            # Se il file è un'immagine .png, viene aggiunto alla lista
            if name.endswith('.png'):
                picture_list.append(os.path.join(root, name))
    return picture_list
