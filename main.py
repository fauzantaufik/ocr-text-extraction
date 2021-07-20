from flask import Flask
import os
from extraction import Extractor
import argparse
import time


def bool_as_str(val):
    return val.lower() in ["1", "true", "t"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--implement",
        type=str,
        nargs="?",
        default=os.environ.get('IMPLEMENT', 'extract'),
        const=True,
        help="Kind of implementation of the app.",
    )

    parser.add_argument(
        "--page-start",
        type=str,
        default=os.environ.get('PAGE_START', None),
        help="Page start.",
    )

    parser.add_argument(
        "--page-end",
        type=str,
        default=os.environ.get('PAGE_END', None),
        help="Page end.",
    )

    parser.add_argument(
        "--skip-if-error",
        type=str,
        default=os.environ.get('SKIP_IF_ERROR', 'false'),
        help="Page end.",
    )

    args, _ = parser.parse_known_args()

    # get all params
    implement = args.implement
    page_start = int(args.page_start)
    page_end = int(args.page_end)
    skip_if_error = bool_as_str(args.skip_if_error)


    extractor = Extractor()
    pdf_file = os.path.join(extractor.base_dir, 'csi1.pdf')
    if implement == 'extract':
        extractor.extract(pdf_file, page_start, page_end, skip_if_error)
    elif implement == 'combine':
        extractor.combine_text(page_start, page_end, skip_if_error=skip_if_error)
    elif implement == 'test':
        app = Flask(__name__)
        app.run()