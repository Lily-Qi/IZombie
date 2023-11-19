import os
import datetime


def get_timestamp():
    return datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")


def create_folder_if_not_exist(folder_name):
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def format_num(n):
    if n > 1_000_000:
        if n % 1_000_000 == 0:
            return f"{int(n / 1_000_000)}m"
        return f"{n / 1_000_000}m"
    if n > 1_000:
        if n % 1_000 == 0:
            return f"{int(n / 1_000)}k"
        return f"{n / 1_000}k"
    return str(n)
