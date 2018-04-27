from glob import glob
import os

def path_getter(path):
    """ Return a file with the format of label and path """
    dirs = os.path.join(path, "*")
    dirs = glob(dirs)
    paths = [glob("{}/*.jpg".format(dir)) for dir in dirs]
    x, y = [], []
    for directory in paths:
        for path in directory:
            print path
            x.append(path)
            y.append(os.path.dirname(path).split(
                    '/')[-1].split('.')[0])
    with open("{}/DatasetFile.txt".format(os.path.dirname(dirs[0])),'w')as f:
        for i in xrange(len(x)):
            f.write("{} {}\n".format(y[i], x[i]))
    return x, y
