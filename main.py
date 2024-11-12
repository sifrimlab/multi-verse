import dataloader
import eval
import model
import train
import utils
import ProcessingRNA
import PrepocessingATAC


filename = "brain3k_multiome"

mdata = dataloader.func1(filename)

ProcessingRNA.Processing_RNA(mdata)
PrepocessingATAC.Prepocessing_ATAC(mdata)

mdata_new = train.train(mdata)
mdata_new.write("data/brain3k_processed.h5mu")
