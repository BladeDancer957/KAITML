import pickle

def to_pickle(obj, fname): #存储为.pkl文件
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(fname): #读取.pkl文件
    with open(fname, "rb") as f:
        return pickle.load(f)
