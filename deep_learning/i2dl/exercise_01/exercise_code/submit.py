import os
from time import sleep
from zipfile import ZipFile


RELEVANT_FOLDERS = ['models', 'exercise_code']


def zipdir(path, ziph):
    """ Recursively adds a folder and all its subfolders to a zipfile
    :param path: path of input folder to be added to zipfile
    :param ziph: a ZipFile object
    """
    # print(path)
    # print(os.walk(path))
    for root, dirs, files in os.walk(path):
        for file in files:
            # print(file)
            ziph.write(os.path.join(root, file))


def submit_exercise(
    zip_output_filename='submission',
    data_path='.',
    relevant_folders=RELEVANT_FOLDERS
):
    """ Creates a curated zip out of submission related files
    :param zip_output_filename: output filename of zip without extension
    :param data_path: path where we look for required files
    :param relevant_folder: folders which we consider for zipping besides
    jupyter notebooks
    """
    # Notebook filenames
    notebooks_filenames = [x for x in os.listdir(data_path)
                           if x.endswith('.ipynb')]
    # Existing relevant folders
    relevant_folders = [x for x in os.listdir(data_path)
                        if x in relevant_folders]
    print('relevant folders: {}\nnotebooks files: {}'.format(
        relevant_folders, notebooks_filenames))

    # Check output filename
    if not zip_output_filename.endswith('.zip'):
        zip_output_filename += '.zip'

    # Create output directory if the student removed it
    folder_path = os.path.dirname(zip_output_filename)
    if folder_path != '':
        os.makedirs(folder_path, exist_ok=True)

    with ZipFile(zip_output_filename, 'w') as myzip:
        # Add relevant folders
        for folder in relevant_folders:
            print('Adding folder {}'.format(folder))
            if len(os.listdir(folder)) == 0 and folder == RELEVANT_FOLDERS[0]:
                sleep(2)
                if len(os.listdir(folder)) == 0:
                    msg = f"ERROR: The folder '{folder}' is EMPTY! Make sure that the relevant cells ran properly \
                        and the relevant files were saved and then run the cell again."
                    raise Exception(" ".join(msg.split()))
            
            myzip.write(folder)
            zipdir(folder, myzip)
        # Add notebooks
        for fn in notebooks_filenames:
            print('Adding notebook {}'.format(fn))
            myzip.write(fn)

    print('Zipping successful! Zip is stored under: {}'.format(
        os.path.abspath(zip_output_filename)
    ))


"""
Here:
1. The function `zipdir` is defined to recursively add a folder and all its subfolders to a zipfile.
2. The function `submit_exercise` is defined to create a curated zip out of submission related files.
3. The function checks for existing relevant folders and notebook filenames in the specified data path.
4. It creates an output directory if the student removed it.
5. The function creates a zip file and adds the relevant folders and notebooks to it.
6. It prints a success message with the path of the created zip file.
7. The function also checks if the 'models' folder is empty and raises an exception if it is, after a 2-second wait.
8. The function uses the `os` and `zipfile` modules to handle file and directory operations.
9. The function uses the `sleep` function from the `time` module to introduce a delay before checking if the 'models' folder is empty.
10. The function uses the `os.path` module to handle file paths and directories.
11. The function uses the `os.walk` method to recursively traverse the directory tree and add files to the zip file.
12. The function uses the `os.makedirs` method to create directories if they do not exist.
13. The function uses the `os.listdir` method to list the contents of a directory.
14. The function uses the `os.path.abspath` method to get the absolute path of the zip file.
"""