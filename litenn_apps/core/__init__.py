import urllib.request
import hashlib

def download_file(url, filepath, progress_bar=True):
    f = None
    try:
        f = open(filepath, 'wb')
        
        u = urllib.request.urlopen(url)

        length = int( u.getheader('content-length') )
        file_size_dl = 0
        while True:
            buffer = u.read(8192)
            if not buffer:
                break
            
            f.write(buffer)

            file_size_dl += len(buffer)
        
            if progress_bar:
                print(f'Downloading {file_size_dl} / {length}', end='\r')
            
    except:
        print(f'Unable to download {url}')
        raise
    
    if f is not None:
        f.close()
        

def sha256(filepath):
    h = hashlib.sha256()

    with open(filepath, 'rb') as file:
        while True:
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()
        