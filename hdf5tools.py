import h5py
import datetime
import time
import glob
import os
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def get_pmaps(h5file):
    flatmap = h5file.get('rec.training.cvnmaps').get('cvnmap')
    pmaps = np.reshape(flatmap, (-1, 2, 100, 80, 1))
    pmaps = np.transpose(pmaps, (0, 1, 3, 2, 4))
    return pmaps

# Returns a Numpy array of the 5 class labels where:
# numu = 0, nue = 1, nutau = 2, nc = 3, cosmic = 4
def get_labels(h5file):
    pdgs = h5file.get('rec.mc.nu').get('pdg').value
    iscc = h5file.get('rec.mc.nu').get('iscc').value

    mcrun = h5file.get('rec.mc.nu').get('run').value
    mcsubrun = h5file.get('rec.mc.nu').get('subrun').value
    mcevt = h5file.get('rec.mc.nu').get('evt').value
    mcsubevt = h5file.get('rec.mc.nu').get('subevt').value

    prun = h5file.get('rec.training.cvnmaps').get('run').value
    psubrun = h5file.get('rec.training.cvnmaps').get('subrun').value
    pevt = h5file.get('rec.training.cvnmaps').get('evt').value
    psubevt = h5file.get('rec.training.cvnmaps').get('subevt').value

    mcd = { 'run' : np.char.mod('%d', mcrun[:,0]),
            'subrun' : np.char.mod('%d', mcsubrun[:,0]),
            'evt' : np.char.mod('%d', mcevt[:,0]),
            'subevt' : np.char.mod('%d', mcsubevt[:,0]),
            'iscc' : iscc[:,0],
            'pdg' : pdgs[:,0]}

    mcdf = pd.DataFrame(data=mcd)


    pmd = { 'run' : np.char.mod('%d', prun[:,0]),
            'subrun' : np.char.mod('%d', psubrun[:,0]),
            'evt' : np.char.mod('%d', pevt[:,0]),
            'subevt' : np.char.mod('%d', psubevt[:,0]),
            'label' : np.repeat(4, len(prun[:,0]))}

    pmdf = pd.DataFrame(data=pmd)

    zflrun = pmdf.run.map(len).max()
    zflsubrun = pmdf.subrun.map(len).max()
    zflevt = pmdf.evt.map(len).max()
    zflsubevt = pmdf.subevt.map(len).max()

    mcdf['run'] = mcdf['run'].apply(lambda x : x.zfill(zflrun))
    mcdf['subrun'] = mcdf['subrun'].apply(lambda x : x.zfill(zflsubrun))
    mcdf['evt'] = mcdf['evt'].apply(lambda x : x.zfill(zflevt))
    mcdf['subevt'] = mcdf['subevt'].apply(lambda x : x.zfill(zflsubevt))
    mcdf['key'] = mcdf.run+mcdf.subrun+mcdf.evt+mcdf.subevt

    pmdf['run'] = pmdf['run'].apply(lambda x : x.zfill(zflrun))
    pmdf['subrun'] = pmdf['subrun'].apply(lambda x : x.zfill(zflsubrun))
    pmdf['evt'] = pmdf['evt'].apply(lambda x : x.zfill(zflevt))
    pmdf['subevt'] = pmdf['subevt'].apply(lambda x : x.zfill(zflsubevt))
    pmdf['key'] = pmdf.run+pmdf.subrun+pmdf.evt+pmdf.subevt

    nudf = pmdf.loc[pmdf.key.isin(mcdf.key)]
    cosmicdf = pmdf.loc[~pmdf.key.isin(mcdf.key)]

    nudf = pd.merge(nudf, mcdf)

    nudf.loc[abs(nudf.pdg)==12, 'label'] = 1
    nudf.loc[abs(nudf.pdg)==14, 'label'] = 0
    nudf.loc[abs(nudf.pdg)==16, 'label'] = 2
    nudf.loc[nudf.iscc==0, 'label'] = 3

    nudf = nudf.drop(['pdg', 'iscc'], axis=1) # Drop to concat with cosmics

    # Glue the neutrino and cosmic dfs back together
    df = pd.concat([nudf, cosmicdf])

    df = df.sort_values(['key'], ascending=True)
    labels = df['label']

    return np.array(labels)

# Reduce the number of cosmics in the sample to around 10%
def downsample_cosmics(pm, lb):

    """
    If there are more than 10% of cosmics in the dataset,
    then select .1*n indices corresponding to cosmics and
    return those samples.
    """
    ncosmics = np.where(lb==4)[0].shape[0]
    nsamples = lb.shape[0]
    nsel = int(np.floor((0.1*nsamples - 0.1*ncosmics) / 0.9))
    ndel = ncosmics - nsel
    print("%d cosmics out of %d total events and %d will be retained" % (ncosmics, nsamples, nsel))

    # selcosmics / (nsamples-ncosmics)+selcosmics = 0.9
    # selcosmics = 0.9*(nsamples-ncosmics + selcosmics)
    # selcosmics - 0.9*selcosmics = 0.9*nsamples - 0.9*ncosmics
    # selcosmics = (0.9*nsamples - 0.9*ncosmics)/0.1

    if ncosmics <= 0.1*nsamples:
        return pm, lb
    delcosmics = np.sort(random.sample(list(np.where(lb==4)[0]), ndel))
    print(delcosmics)
    print("Downsampling cosmics to 10%...")
    pm = np.delete(pm, delcosmics, axis=0)
    lb = np.delete(lb, delcosmics, axis=0)

    print(pm.shape)
    print(lb.shape)

    return pm, lb


def produce_labeled_h5s(h5dir, samplecosmics=False):
    # The list of files in the directory
    h5files = glob.glob(os.path.join(h5dir, "*.h5"))
    print("f5files ... ",h5files)
    # Create a directory to store the processed files
    # These will persist, so future training can accept these files
    # directly as input
    labeleddir = os.path.join(h5dir, "labeled_downsampled")
    if not os.path.exists(labeleddir):
        os.makedirs(labeleddir)

    i=0
    for filename in h5files:
        print("Processing {fn} at {time}".format(fn=filename, time=datetime.datetime.now()))
        starttime = time.time()
        # Get the current hdf5 file
        currh5 = h5py.File(filename)
        pm = get_pmaps(currh5)
        lb = get_labels(currh5)

        # Reduce the number of cosmics in the final training input
        if samplecosmics:
            pm, lb = downsample_cosmics(pm, lb)

        # Shuffle the dataset here to aid sequential reads later
        shuffle = np.random.permutation(pm.shape[0])
        pm = pm[shuffle, ...]
        lb = lb[shuffle]
        
        
        
        
        print("name: ",filename)

        outname = os.path.join(labeleddir, "labeled_{ind}_{fn}".format(ind=i, fn=filename[5:]))
        #outname2 = os.path.join(labeleddir, "labeled_{ind}_Data".format(ind=i))
        #if not os.path.exists(outname2):
        #    os.makedirs(outname2)
        print("name: ",outname)
        outh5 = h5py.File(outname, 'w')
        outh5.create_dataset('pixelmaps', data=pm)
        outh5.create_dataset('labels', data=lb)
        outh5.close()

        print("{fn} written at {time}".format(fn=outname, time=datetime.datetime.now()))
        print("Elapsed time: {elps} seconds".format(elps=time.time()-starttime))
        print("Events written: {evt}\n".format(evt=pm.shape[0]))
        i += 1

    return labeleddir


def print_pixelmaps(pmaps,id):
        result = (pmaps[id][0])[:, :, 0]
        plt.imshow(result,cmap='Oranges')
        plt.show()
        result2 = (pmaps[id][1])[:, :, 0]
        plt.imshow(result2,cmap='Oranges')
        plt.show()
        return 0   
