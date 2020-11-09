import urllib.request
import hashlib
import cv2

try:
    import google.colab
    # We are in colab
    
    from google.colab.patches import cv2_imshow as cv2_imshow_colab
    
    def cv2_imshow(wnd, img):
        cv2_imshow_colab(img)
        
    def cv2_destroyWindow(wnd):
        pass
except:
    cv2_imshow = cv2.imshow
    cv2_destroyWindow = cv2.destroyWindow


def download_file(url, filepath, progress_bar=True):
    f = None
    
    if progress_bar:
        print(f'Downloading {url}')
        
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
        