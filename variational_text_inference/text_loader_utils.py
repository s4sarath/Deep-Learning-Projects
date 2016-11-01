from preprocess import pre_process_sentence
import numpy as np


from itertools import islice, chain


        
class TextLoader():
    
    def __init__(self , text_list):
        
        if isinstance(text_list, list):
            self.data = text_list
        
        self.vocab = {}
        self.voc_to_idx = []
        self.vocab_inverse = {}
        self._vocab(self.data)
        self._vocab_inverse()
       
    
        
    def _preprocess(self , text_chunk ):
        return pre_process_sentence(text_chunk)
    
    
    

    def get_batch(self,batch_size):
        sourceiter = iter(self.data)
        if len(self.data) % batch_size == 0:
            iterations = len(self.data) / batch_size
        else:
            iterations = int(len(self.data) / batch_size) + 1
        for i in xrange(iterations):
            batch_data = islice(sourceiter, batch_size)
            yield chain([batch_data.next()], batch_data)
            


    def _vocab(self , chunks ):
        
        for text_ in iter(self.data):
            preprocessed = self._preprocess(text_)
            for w in list(set(preprocessed)):
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab) 
        

    def _vocab_inverse(self):

        self.vocab_inverse = {k:v for v, k in self.vocab.iteritems()}

    def _bag_of_words(self , chunk_data , vocab_size = None):
        
        self.voc_to_idx = voc_to_idx = self._vocab_to_idx(chunk_data)
        self.bow = []
        self.dow = []
        if vocab_size is None:
            vocab_size = len(self.vocab)
        self.bow = [np.bincount(idx , minlength=vocab_size) for idx in voc_to_idx]
        self.bow = np.array(self.bow)
        self.dow = self.bow.copy()
        self.dow[self.dow > 0] = 1 
        self.negative_mask = self.dow.copy()
        self.negative_mask[self.negative_mask == 0] = -1
        # self.index_positions = self.get_matrix_position(self.voc_to_idx)

        return self.bow , self.dow , self.negative_mask 

    def get_matrix_position(self, mat_idx):
    
        mat_idx_full_ = []
        for i in xrange(2):
            temp_list = []
            for elem in mat_idx[i]:
                temp_list.append([i, elem])
            mat_idx_full_.extend(temp_list)
        return mat_idx_full_
        
    def _vocab_to_idx(self , chunk_data):
        
        voc_to_idx = []
        preprocessed_data = map(self._preprocess, chunk_data)
        preprocessed_data_ = []
        self.data_index = []
        for index_pos , data_ in enumerate(preprocessed_data):
            if data_:
                preprocessed_data_.append(data_)
                self.data_index.append(index_pos) ######### After pre-processing some values might be null , so to get proper index of not-null values of data , we are keeping track of the index
        for chunks in preprocessed_data_:
            voc_to_temp = [self.vocab[w] for w in chunks if w in self.vocab]
            voc_to_idx.append(voc_to_temp)
        voc_to_idx = np.array(voc_to_idx)
        ############ Setting to empyt to list to free up memory ( I am not sure whether it is right , I guess so )
        preprocessed_data_ = []
        preprocessed_data_ = []
        return voc_to_idx
                

                
        


if __name__ == "__main__":

    import cPickle
    from sklearn.datasets import fetch_20newsgroups
    twenty_train = fetch_20newsgroups(subset='train')   
    data_ = twenty_train.data
    print "Download 20 news group data completed"
    A = TextLoader(data_)
    batch_size = 100
    batch_data = A.get_batch(batch_size)

    for batch_ in batch_data:
        collected_data = [chunks for chunks in batch_]
        bow , dow , negative_mask  = A._bag_of_words(collected_data)

        print bow.shape , dow.shape , negative_mask.shape
        print bow
        print dow
        print negative_mask

        print bow.max() , dow.max() , negative_mask.max()
        print "Succesful"
        break