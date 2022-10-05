#!/usr/bin/env python3
import argparse
import collections
import pickle
import sys
import typing


def parse_line(line: str):
    fields = line.split()

    card_id = fields[0]
    fields = fields[1:]
    ind_of_L = fields.index('L')

    entry_type = fields[:ind_of_L]
    content = ' '.join(fields[ind_of_L+1:])
    subfields = {sf[0]: sf[1:] for sf in content.split('$$') if sf}

    return card_id, entry_type, subfields


def drop_suffix_if_present(string: str, suffix: str) -> str:
    if string.endswith(suffix):
        return string[:-len(suffix)]
    else:
        return string


def accepted_content(entry_type, subfields):
    if entry_type[0] == '100':  # author
        return [(entry_type[0] + sf_type, subfields.get(sf_type, '')) for sf_type in ['a']]
    elif entry_type[0] == '260':  # Place of publication, name of publisher, year of publication
        return [(entry_type[0] + sf_type, subfields.get(sf_type, '')) for sf_type in ['a', 'b', 'c']]
    elif entry_type[0] == '300':  # Pages
        return [(entry_type[0] + sf_type, subfields.get(sf_type, '')) for sf_type in ['a']]
    elif entry_type[0].startswith('245'):  # title, subtitle, author
        if 'c' in subfields:
            if 'b' in subfields:
                subfields['b'] = drop_suffix_if_present(subfields['b'], ' /')
            elif 'a' in subfields:
                subfields['a'] = drop_suffix_if_present(subfields['a'], ' /')
        return [(entry_type[0] + sf_type, subfields.get(sf_type, '')) for sf_type in ['a', 'b', 'c']]
    elif entry_type[0] == '765':  # title in original language
        return [(entry_type[0] + sf_type, subfields.get(sf_type, '')) for sf_type in ['a']]
    elif entry_type[0] == '910':  # ID that often appears on the card
        return [(entry_type[0] + sf_type, subfields.get(sf_type, '')) for sf_type in ['b']]
    else:
        return []


def filter_empty(all_data):
    cards_to_delete = []
    for card_id in all_data:
        types_to_delete = []
        for full_type in all_data[card_id]:
            if all_data[card_id][full_type] == '':
                types_to_delete.append(full_type)

        for t in types_to_delete:
            del all_data[card_id][t]

        if not all_data[card_id]:
            cards_to_delete.append(card_id)

    for card_id in cards_to_delete:
        del all_data[card_id]


def main(args):
    all_data: typing.Dict[str, typing.Dict[str, str]] = collections.defaultdict(dict)

    nb_failed_lines = 0
    i = 0
    while True:
        i += 1
        try:
            line = sys.stdin.readline()
            if line == '':
                break

            card_id, entry_type, subfields = parse_line(line)
        except Exception as e:
            nb_failed_lines += 1
            sys.stderr.write(f'Failed to parse line no {i+1}: {e}\n')
            continue

        filtered_content = accepted_content(entry_type, subfields)

        for full_type, content in filtered_content:
            all_data[card_id][full_type] = content
    sys.stderr.write(f'Failed to parse total of {nb_failed_lines} lines\n')

    filter_empty(all_data)

    with open(args.out_pickle, 'wb') as f:
        pickle.dump(all_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_pickle')
    args = parser.parse_args()

    main(args)
