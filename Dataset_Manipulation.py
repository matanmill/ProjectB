import utils
import Paths
import json
import pandas as pd
from collections import defaultdict
import argparse

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--label_vocabulary_path", type=str, default=r'./datafiles/vocabulary_wo_music.csv',
                    help="path for decoding the labels from provided vocabulary")
parser.add_argument("--train_path", type=str, default='./datafiles/fsd50k_tr_full_no_music.json',
                    help="path for training set")
parser.add_argument("--test_path", type=str, default='./datafiles/fsd50k_eval_full_no_music.json',
                    help="path for test set")
parser.add_argument("--val_path", type=str, default='./datafiles/fsd50k_val_full_no_music.json',
                    help="path for validation set")
parser.add_argument("--audioset_onthology_path", type=str, default='./datafiles/audioset_ontology.json',
                    help="path for audioset onthology")
parser.add_argument("--output_path", type=str, default='./datafiles/FSD50K_onthology.json',
                    help="path for audioset onthology")
parser.add_argument("--small_onthology_path", type=str, default='./datafiles/small_FSD50K_onthology.json',
                    help="path for audioset onthology")
parser.add_argument("--vocab_small", type=str, default='./datafiles/small_FSD50K_vocabulary.csv',
                    help="path for audioset onthology")
args = parser.parse_args()


def label_count_from_json(json_path, vocabulary_path):
    # Load the vocabulary and label dictionary
    label_dict = create_label_dictionary(vocabulary_path)

    # Load the training data
    with open(json_path, 'r') as f:
        training_json = json.load(f)

    # Count the labels
    label_counts = defaultdict(int)
    for row in training_json['data']:
        labels = row['labels'].split(',')
        for label in labels:
            label_counts[label] += 1

    return dict(label_counts)


def create_label_dictionary(vocabulary_path):
    vocab = pd.read_csv(vocabulary_path, header=None)
    l_dict = {}
    for index in range(len(vocab[0])):
        l_dict[vocab[2][index]] = vocab[1][index]
    return l_dict


def find_location(lst, target_key):
    try:
        location = next(i for i, (key, _) in enumerate(lst) if key == target_key)
        return location
    except StopIteration:
        return -1  # Key not found


def create_fsd50k_ontology(vocabulary_path, audioset_ontology_path, output_path, training_path, validation_path, test_path,
                           small_vocabulary_path=None, return_onthology=True, small=False):
    # Load FSD50K vocabulary
    fsd50k_vocabulary = create_label_dictionary(vocabulary_path)
    if small:
        wanted_vocabulary = create_label_dictionary(small_vocabulary_path)
    else:
        wanted_vocabulary = fsd50k_vocabulary

    # Load AudioSet ontology
    with open(audioset_ontology_path, 'r') as audio_set_ontology_file:
        audio_set_ontology = json.load(audio_set_ontology_file)

    training_dict = label_count_from_json(training_path, vocabulary_path)
    validation_dict = label_count_from_json(validation_path, vocabulary_path)
    test_dict = label_count_from_json(test_path, vocabulary_path)

    # sorted lists
    training_list = list(sorted(training_dict.items(), key=lambda item: item[1], reverse=True))

    # Create FSD50K ontology
    fsd50k_ontology = []

    for audio_set_category in audio_set_ontology:
        category_id = audio_set_category["id"]
        if category_id in wanted_vocabulary:
            fsd_category = {
                "id": category_id,
                "name": wanted_vocabulary[category_id],
                "training_count": training_dict[category_id],
                "validation_count": validation_dict[category_id],
                "test_count": test_dict[category_id],
                "location in frequency of occurence (training)": find_location(training_list, category_id) + 1,
                "child_ids": [],
                "child_names": []
            }
            child_ids = audio_set_category["child_ids"]
            for child_id in child_ids:
                if child_id in fsd50k_vocabulary:
                    fsd_category["child_ids"].append(child_id)
                    fsd_category["child_names"].append(fsd50k_vocabulary[child_id])

            fsd50k_ontology.append(fsd_category)
            fsd50k_ontology = sorted(fsd50k_ontology, key=lambda x: len(x['child_ids']), reverse=True)
            fsd50k_ontology = sort_ontology_by_frequency(fsd50k_ontology)

    # Save FSD50K ontology as JSON
    with open(output_path, 'w') as output_file:
        json.dump(fsd50k_ontology, output_file, indent=2)

    if return_onthology:
        return fsd50k_ontology


def sort_ontology_by_frequency(ontology):
    sorted_ontology = sorted(ontology, key=lambda x: x["location in frequency of occurence (training)"])
    return sorted_ontology


def find_main_labels(ontholgy):
    all_child_ids = set()
    main_labels = []

    # Collect all child_ids
    for item in ontholgy:
        all_child_ids.update(item['child_ids'])

    # Check if child_ids exist in other labels
    for item in ontholgy:
        # if any(child_id in all_child_ids for child_id in item['child_ids']):
        # main_labels.append(item['name'])

        if not item["id"] in all_child_ids:
            main_labels.append(item['name'])

    return main_labels


def create_small_label_dictionary(vocabulary_path, top_labels, csv_path):
    vocab = pd.read_csv(vocabulary_path, header=None)
    l_dict = {}
    for index in range(len(vocab[0])):
        if vocab[1][index] in top_labels:
            l_dict[vocab[1][index]] = vocab[2][index]
    pd.DataFrame.from_dict(l_dict, orient='index').to_csv(csv_path, header=False)
    return l_dict


def add_numbering_to_csv(input_csv_path, output_csv_path):
    # Read the input CSV file without header
    df = pd.read_csv(input_csv_path, header=None)

    # Add a numbering column starting from zero
    df.insert(0, 'Number', range(len(df)))

    # Write the DataFrame to the output CSV file without headers
    df.to_csv(output_csv_path, header=False, index=False)


fsdk50_onthology = create_fsd50k_ontology(vocabulary_path=args.label_vocabulary_path, audioset_ontology_path=args.audioset_onthology_path, output_path=args.output_path,
                                          training_path=args.train_path, validation_path=args.val_path, test_path=args.test_path)

main_label = find_main_labels(fsdk50_onthology)
create_small_label_dictionary(args.label_vocabulary_path, main_label, args.vocab_small)
add_numbering_to_csv(args.vocab_small, args.vocab_small)

new_small_onthology = create_fsd50k_ontology(vocabulary_path=args.label_vocabulary_path, audioset_ontology_path=args.audioset_onthology_path,
                                             output_path=args.small_onthology_path, training_path=args.train_path, validation_path=args.val_path,
                                             test_path=args.test_path, small_vocabulary_path=args.vocab_small, small=True)

print("Finished preparing the FSD50K onthology")
