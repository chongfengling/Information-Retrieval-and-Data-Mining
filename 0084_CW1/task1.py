import numpy as np

def terms_counter():
    pass

if __name__=='__main__':
    file_path = 'passage-collection.txt'

    with open(file_path, 'r') as f:
        data = f.readlines()

    print(str(data))