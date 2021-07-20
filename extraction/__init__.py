import os
from dotenv import load_dotenv
import logging
import traceback
import sys

logging.basicConfig(level=logging.DEBUG, filename='extraction.log')

# environment variable file
dotenv_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), 
                '.env')
            )
            
load_dotenv(dotenv_path)

from extraction.line import line_detection
from extraction.pytesseract_detection import pytesseract_detection
from extraction.pytesseract_ocr import pytesseract_ocr
from nltk.tokenize import sent_tokenize
import pandas as pd

from pdf2image import convert_from_bytes
import pytesseract
from tqdm import tqdm
import cv2
import nltk


class Extractor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.setup_ocr()

    
    def setup_ocr(self):
        self.size_ = 7000
        self.file_page_format = 'page_{0}.{1}'
        self.tesseract_dir = os.path.abspath(os.path.join(self.base_dir, 'TESSERACT-OCR'))
        self.extract_dir = os.path.abspath(os.path.join(self.base_dir, 'extracted_data'))
        self.mkdir_if_not_exist(self.extract_dir)
        self.all_page_filename = os.path.abspath(os.path.join(self.extract_dir, 'all_page.txt')) # path to save all text in a file
        OS = os.environ.get('OS', 'WINDOWS')
        if OS == 'WINDOWS':
            self.poppler_path = os.path.abspath(os.path.join(self.base_dir, 'poppler-0.90.1', 'bin'))
            pytesseract.pytesseract.tesseract_cmd = os.path.join(self.tesseract_dir, 'tesseract.exe')
        elif OS == 'LINUX':
            self.poppler_path = os.environ.get('POPPLER_DIRECTORY')
            pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_DIRECTORY')


    @staticmethod
    def mkdir_if_not_exist(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)


    def get_file_path(self, num_page, ext):
        path = os.path.join(self.extract_dir, str(num_page))
        file_path = os.path.join(path, self.file_page_format.format(num_page, ext))
        return file_path, path


    def save_page_pdf(self, image, num_page):
        file_path, path = self.get_file_path(num_page, 'png')
        self.mkdir_if_not_exist(path)
        image.save(file_path, "PNG", quality=95, optimize=True, progressive=True)
    

    def read_page_pdf(self, num_page):
        file_path, _ = self.get_file_path(num_page, 'png')
        img_ori = cv2.imread(file_path)
        img_gray = cv2.imread(file_path, 0)
        return img_ori, img_gray


    def page_extractor(self, pdf_file, num_page):
        image = convert_from_bytes(open(pdf_file, "rb").read(), size=self.size_, poppler_path=self.poppler_path, first_page=num_page, last_page=num_page)[0]
        self.save_page_pdf(image, num_page)
        img_ori, img_gray = self.read_page_pdf(num_page)
        height_, width_ = img_gray.shape

        table, nontable = line_detection(img_ori, img_gray)
        undetected_bbox = pytesseract_detection(img_ori)
        df_combined_final = pytesseract_ocr(img_ori, img_gray, height_, width_, table, nontable, undetected_bbox)
        return df_combined_final

    
    def remove_footer(self, df):
        indx = list(df[df['top']>6600].index)
        df = df.drop(indx)
        return df

    
    def isTitle(self, text):
        n_sentence = len(sent_tokenize(text))
        return (len(text) < 50 or n_sentence < 2) and ',' not in text


    def process_text(self, text):
        text = text.replace('-, ', '')
        return text


    def write_text(self, f, df, col):
        for _, value in df.iterrows():
            text = value[col]
            text = self.process_text(text)
            if not self.isTitle(text) :
                f.write(f'{text}\n\n')


    def get_columns(self, df):
        left_column = df[df['left']<2000]
        right_column = df[df['left']>2000]
        return left_column, right_column


    def save_text(self, df, num_page):
        filename, _ = self.get_file_path(num_page, 'txt')
        df_left, df_right = self.get_columns(df)
        with open(filename, 'w') as f:
            self.write_text(f, df_left.sort_values('top'), 'lines_combine')
            self.write_text(f, df_right.sort_values('top'), 'lines_combine')

    
    def save_csv(self, df, num_page):
        filename, _ = self.get_file_path(num_page, 'csv')
        df.to_csv(filename, index=False)

    
    def combine_text(self, start_page, end_page, skip_if_error=False):
        with open(self.all_page_filename, 'a') as outfile:
            for num_page in tqdm(range(start_page, end_page+1)) :
                try :
                    filename, _ =  self.get_file_path(num_page, 'txt')
                    with open(filename) as infile:
                        for line in infile:
                            line = self.process_text(line)
                            outfile.write(line)
                except Exception as e :
                    logging.error(traceback.format_exc())
                    logging.error(str(e))
                    if not skip_if_error :
                        sys.exit()


    def extract(self, pdf_file, page_start, page_end, skip_if_error=False):
        all_images = convert_from_bytes(open(pdf_file, "rb").read(), size=10, poppler_path=self.poppler_path)
        n_pages = len(all_images)
        logging.info(f'Extract {n_pages} images')
        for num_page in tqdm(range(page_start, page_end+1)):
            try :
                df_text = self.page_extractor(pdf_file, num_page)
                self.save_csv(df_text, num_page)
                df_text = df_text.reset_index().drop('index', axis=1)
                df_text = self.remove_footer(df_text)
                self.save_text(df_text, num_page)
                logging.info('Successfully extract page {}'.format(num_page))
            except Exception as e :
                logging.error(traceback.format_exc())
                logging.info('Fail to extract page {}'.format(num_page))
                if not skip_if_error:
                    sys.exit()