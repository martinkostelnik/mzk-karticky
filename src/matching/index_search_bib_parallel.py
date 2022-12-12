import os
import time

from multiprocessing import Pool
from functools import partial

import src.matching.index_search_bib as seq


def main():
    args = seq.parse_arguments()

    print("Reading inference data ...")
    with open(args.inference_path, "r") as f:
        inference_lines = f.readlines()
    print("Inference data read.")

    t0 = time.time()
    processing_function = partial(seq.process_file, args=args)
    print("Partial function created")

    pool = Pool(processes=4)
    print("Pool created")

    print("Starting search ...")
    result = []
    cards_searched = 0

    try:
        for matching, alignment in pool.imap_unordered(processing_function, inference_lines):
            cards_searched += 1

            if matching and alignment:
                result.append((matching, alignment))
    except KeyboardInterrupt:
        pass

    t1 = time.time()
    dur = t1 - t0
    print(f'Took {dur:.1f} seconds to search {cards_searched} records. {dur / cards_searched:.2f}s')

    print("Saving results ...")
    seq.save_results(result, args.out_path)
    print(f"Results saved to {args.out_path}")


if __name__ == '__main__':
    exit(main())
