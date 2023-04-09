import os

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from src.constant.app_constants import AppConstants


class FileSystemUtils:

    @staticmethod
    def join(directory: str, filename: str) -> str:
        return os.path.join(directory, filename)

    @staticmethod
    def get_filename(file: FileStorage, with_extension=True)->str:
        filename = secure_filename(file.filename)
        if not with_extension:
            filename = FileSystemUtils.get_filename_without_extension(filename)
        return filename

    @staticmethod
    def get_file_extension(filepath: str)->str:
        return os.path.splitext(filepath)[1]

    @staticmethod
    def get_filename_without_extension(filepath: str)->str:
        return os.path.splitext(filepath)[1]

    @staticmethod
    def get_file_directory(filepath: str)->str:
        return os.path.dirname(filepath)

    @staticmethod
    def save_temp_file(file: FileStorage) -> str:
        file_save_path = FileSystemUtils.join(AppConstants.temp_file_upload_directory,
                                              FileSystemUtils.get_filename(file))
        file.save(file_save_path)
        return file_save_path

    @staticmethod
    def append_filename_with(filename, text_to_append: str)->str:
        file_directory=FileSystemUtils.get_file_directory(filename)
        filename_without_extension=FileSystemUtils.get_filename_without_extension(filename)
        file_extension=FileSystemUtils.get_file_extension(filename)
        return FileSystemUtils.join(file_directory,f'{filename_without_extension}{text_to_append}{file_extension}')
