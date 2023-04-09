import cv2

from src.enum.preprocessing_stage import PreprocessingStage
from src.service.preprocessor.preprocessor_base import Preprocessor
from PIL import Image, ImageFilter
import numpy as np


class ImageFilterer(Preprocessor):
  @staticmethod
  def get_filterer():
    return ImageFilter.MedianFilter(size=3)

  def get_preprocessing_stage(self) ->PreprocessingStage:
    return PreprocessingStage.filtration

  def process(self, ip_img_from, op_img_at) -> None:
    print(f'IMAGE-FILTERER:: request to process img:  {ip_img_from}')
    print(f'IMAGE-FILTERER:: loading img:  {ip_img_from}')
    img = Image.open(ip_img_from)
    print(f'IMAGE-FILTERER:: successfully loaded img:  {ip_img_from}')
    filtered_img = img.filter(ImageFilterer.get_filterer())
    print(f'IMAGE-FILTERER:: successfully generated output img')
    is_saved = cv2.imwrite(op_img_at, np.array(filtered_img))
    if is_saved:
      print(
          f'IMAGE-FILTERER:: successfully saved output img at:  {op_img_at}')
    else:
      print(
          f'IMAGE-FILTERER:: failed to save output img at:  {op_img_at}')
      raise Exception("Failed to save output image")
