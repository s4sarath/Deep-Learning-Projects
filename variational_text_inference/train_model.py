
from text_loader_utils import TextLoader
import cPickle
import numpy as np
import tensorflow as tf
from variational_model import Variational_Document_Model
import os
from vector_utils import find_norm

np.random.seed(0)
tf.set_random_seed(0)

def xavier_init(fan_in , fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
#     fan_in = in_and_out[0]
#     fan_out = in_and_out[1]
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)


if __name__ == "__main__":


    from sklearn.datasets import fetch_20newsgroups
    twenty_train = fetch_20newsgroups(subset='train')   
    data_ = twenty_train.data
    print "Download 20 news group data completed"
    A = TextLoader(data_)
    batch_size = 100

    with tf.Session() as sess:
        vae = Variational_Document_Model(sess , len(A.vocab), 50, [500 , 500] ,  
                         transfer_fct=tf.nn.relu , output_activation=tf.nn.softmax,
                         batch_size=100, initializer=xavier_init , mode = 'gather'   )
            
        vae._train()
        # Training cycle
        training_epochs = 501
        batch_size = 100
        n_samples = len(data_)
        display_step = 1
        save_step = 100
        for epoch in range(training_epochs):
            batch_data = A.get_batch(batch_size)
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            batch_id = 0
            for batch_ in batch_data:

                collected_data = [chunks for chunks in batch_]
                ##### Here batch_xs ( Bag of words with count of words)
                ##### Here mask_xs  ( Bag of words with 1 at the index of words in doc , no counts)
                ##### Here mask_negative is not using ( Tried with negative sampling )
                batch_xs , mask_xs , mask_negative  = A._bag_of_words(collected_data)
                ###### Here batch_flattened gives position of words in all documents into one array
                ###### because gather_nd does not support gradients . So , we have to use tf.gather
                batch_flattened = np.ravel(batch_xs)
                index_positions = np.where( batch_flattened > 0 )[0] ####### We want locs where , data ( word ) present in document 

                # Fit training using batch data
                if vae.mode == 'gather':
                    
                    cost , R_loss_  = vae.partial_fit(find_norm(batch_xs) , batch_xs.shape[0] , index_positions)
                else:
                    cost , R_loss_  = vae.partial_fit(mask_xs , batch_xs.shape[0] , mask_xs.astype(np.float32))

                avg_cost += cost
                print "Cost {} is".format(cost)
                
            # Display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), \
                      "cost=", "{:.9f}".format(avg_cost/total_batch)

            if epoch % save_step == 0:
                vae.save(global_step = epoch)
