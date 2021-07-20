import sys
from extraction import Extractor

if __name__ == '__main__':
    extractor = Extractor()
    start_page = 12
    end_page = 14
    skip_if_error = True
    extractor.combine_text(start_page, end_page, skip_if_error=skip_if_error)
