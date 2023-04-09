from src.enum.preprocessing_stage import PreprocessingStage
from src.service.preprocessor.preprocessor_base import Preprocessor
import cv2
import numpy as np


class ImageEnhancer(Preprocessor):
  @staticmethod
  def get_lookup_table():
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    return table

  def get_preprocessing_stage(self) ->PreprocessingStage:
    return PreprocessingStage.enhancement

  def process(self, ip_img_from, op_img_at) -> None:
    print(f'IMAGE-ENHANCER:: request to process img:  {ip_img_from}')
    print(f'IMAGE-ENHANCER:: loading img:  {ip_img_from}')
    img = cv2.imread(ip_img_from)
    print(f'IMAGE-ENHANCER:: successfully loaded img:  {ip_img_from}')
    img = cv2.LUT(img, ImageEnhancer.get_lookup_table())
    print(f'IMAGE-ENHANCER:: successfully generated output img')
    is_saved=cv2.imwrite(op_img_at, img)
    if is_saved:
      print(
        f'IMAGE-ENHANCER:: successfully saved output img at:  {op_img_at}')
    else:
      print(
          f'IMAGE-ENHANCER:: failed to save output img at:  {op_img_at}')
      raise Exception("Failed to save output image")

