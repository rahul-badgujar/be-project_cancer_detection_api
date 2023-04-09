import os

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from src.constant.app_constants import AppConstants


class FileSystemUtils:

    @staticmethod
    def join(directory: str, filename: str) -> str:
        return os.path.join(directory, filename)

    @staticmethod
    def get_filename_of_file(file: FileStorage, with_extension=True) -> str:
        filename = os.path.basename(file.filename)
        if not with_extension:
            filename = FileSystemUtils.get_filename_without_extension(filename)
        return filename

    @staticmethod
    def get_filename_from_path(filepath: str, with_extension=True) -> str:
        filename = os.path.basename(filepath)
        if not with_extension:
            filename = os.path.splitext(filename)[0]
        return filename

    @staticmethod
    def get_file_extension(filepath: str) -> str:
        return os.path.splitext(filepath)[1]

    @staticmethod
    def get_file_directory(filepath: str) -> str:
        return os.path.dirname(filepath)

    @staticmethod
    def save_temp_file(file: FileStorage) -> str:
        file_save_path = FileSystemUtils.join(AppConstants.temp_intermediate_files_directory,
                                              FileSystemUtils.get_filename_of_file(file))
        file.save(file_save_path)
        return file_save_path

    @staticmethod
    def get_path_to_store_intermediate_file(filename: str) -> str:
        return FileSystemUtils.join(AppConstants.temp_intermediate_files_directory, filename)

    @staticmethod
    def save_uploaded_file(file: FileStorage) -> str:
        file_save_path = FileSystemUtils.join(AppConstants.temp_file_upload_directory,
                                              FileSystemUtils.get_filename_of_file(file))
        file.save(file_save_path)
        return file_save_path

    @staticmethod
    def append_filename_with(filename: str, text_to_append: str) -> str:
        file_directory = FileSystemUtils.get_file_directory(filename)
        filename_without_extension = FileSystemUtils.get_filename_from_path(filename, with_extension=False)
        file_extension = FileSystemUtils.get_file_extension(filename)
        return FileSystemUtils.join(file_directory, f'{filename_without_extension}{text_to_append}{file_extension}')

    @staticmethod
    def get_os_path_separator() -> str:
        return os.path.sep

    @staticmethod
    def get_filepath_with_up_dir(filepath: str, up_dirs: list) -> str:
        filename = FileSystemUtils.get_filename_from_path(filepath)
        filedir = FileSystemUtils.get_file_directory(filepath)
        file_up_dir=FileSystemUtils.get_os_path_separator().join(up_dirs)
        return os.path.join(filedir, file_up_dir,filename)


if __name__ == '__main__':
    print(FileSystemUtils.get_filepath_with_up_dir('/a/b/z.txt', ["c","d"]))
