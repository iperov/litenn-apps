
import litenn_apps.core

import litenn_apps.recognition.face.S3FD

import litenn_apps.test_data.photo

def predownload_all():
    """
    Predownload all files of all modules.
    """
    
    litenn_apps.recognition.face.S3FD.download_files()
    