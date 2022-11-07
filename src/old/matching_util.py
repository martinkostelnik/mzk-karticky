import numpy as np


def levenshtein_distance(source, target):
    sub_cost = 1
    ins_cost = 1
    del_cost = 1

    source = [ord(c) for c in source]
    target = [ord(c) for c in target]

    if len(target) > len(source):
        target, source = source, target

    target = np.array(target)
    dist = np.ones((1 + len(target) + 1)) * float('inf')
    dist[0] = 0
    for s in source:
        dist[1:-1] = np.minimum(dist[1:-1] + del_cost, dist[:-2] + (target != s) * sub_cost)

        for ii in range(len(dist) - 2):
            if dist[ii + 1] > dist[ii] + ins_cost:
                dist[ii + 1] = dist[ii] + ins_cost

        dist[-1] = np.minimum(dist[-1], dist[-2])

    return dist[-1]


keys_of_interest = [
    '24510',  # title & author and other
    '250',  # edition
    '260',  # publisher
    '1001',  # author
]


def records_match_score(lines, db_entry):
    min_len = 4

    # relevant_values = [db_entry[k] for k in keys_of_interest if k in db_entry]
    relevant_values = [val for val in db_entry.values() if len(val) >= min_len]
    if not relevant_values:
        print('no relevant values in entry')
        print(db_entry)
        return float('-inf')

    matches = []

    for line in lines:
        line = line[:-1]

        if len(line) < min_len:
            continue

        dists = [levenshtein_distance(line, val) for val in relevant_values]
        ref_lens = [min(len(line), len(val)) for val in relevant_values]
        cers = [d/rl for d, rl in zip(dists, ref_lens)]

        # len_ratios = [max(len(line)/len(val), len(val)/len(line)) for val in relevant_values]
        #
        # adj_cers = [cer * f for cer, f in zip(cers, len_ratios)]
        #
        # print(min(adj_cers), line, '=?=', relevant_values[np.argmin(adj_cers)])
        # if min(adj_cers) < 1:
        #     matches.append((line, relevant_values[np.argmin(adj_cers)]))

        if min(cers) < 0.3:
            matches.append((line, relevant_values[np.argmin(cers)]))

    # if '24510' in db_entry:
    #     all_content = ' '.join(line[:1] for line in lines)
    #     the_val = db_entry['24510']
    #     title, author = the_val.split(';')[0].rsplit('/', maxsplit=1)
    #
    #     title_dist = levenshtein_distance(title, all_content)
    #     author_dist = levenshtein_distance(author, all_content)
    #
    #     title_cer = title_dist / len(title)
    #     author_cer = author_dist / len(author)
    #
    #     return len(matches), (title_cer + author_cer) / 2
    # else:
    #     return len(matches), None

    return len(matches), None
