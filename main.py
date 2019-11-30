import hdf5tools
import models
from trainCNN import TrainCVN, GetData
from rmdl import Image_Classification as RMDL
name = "CVN"

#data = hdf5tools.produce_labeled_h5s("Data", samplecosmics=True)
path = 'Data/labeled_downsampled/*.h5'

#model = models.CVNShortSimple()

#TrainCVN(model,path,name,predict = True)

RMDL(path=path, shape=(80, 100, 1), batch_size=128,  epochs=[5, 5, 5], layers_DNN = 5, nodes_DNN = 100, nodes_RNN=2,
     layers_CNN=5, nodes_CNN=16, random_state=42, dropout=0.05)