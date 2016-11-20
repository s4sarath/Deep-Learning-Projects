
from text_loader_utils import TextLoader
import cPickle
import numpy as np
import tensorflow as tf
from variational_model import NVDM
from vector_utils import find_norm
from tf_common_utils import load_model , save_model

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
def train_the_model(vae):
    vae.start_the_model()

    try:
        status = load_model(vae)
        if status:
            print "Restore previously saved model succesfully"
    except: 
        pass
    # Training cycle
    training_epochs = 201
    batch_size = 100
    n_samples = len(data_)
    display_step = 1
    save_step = 100
    for epoch in range(training_epochs):
        batch_data = A.get_batch(batch_size)
        loss_sum = 0.
        kld_sum = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        batch_id = 0
        for batch_ in batch_data:

            batch_id += 1
            collected_data = [chunks for chunks in batch_]
            batch_xs , mask_xs , mask_negative  = A._bag_of_words(collected_data)
            

            feed_dict = {vae.X: batch_xs , vae.dynamic_batch_size:batch_xs.shape[0],
                                            vae.MASK:mask_xs}

            _ , total_cost , recons_loss_ , kld    = vae.SESS.run((vae.optimizer, vae.cost , vae.reconstr_loss , vae.kld), 
                                                      feed_dict= feed_dict)

            

            word_count = np.sum(mask_xs)
            batch_loss = np.sum(total_cost)/word_count
            loss_sum += batch_loss
            kld_sum += np.sum(kld)
            

            print (" Epoch {} Batch Id {} , Loss {} , Kld is {}  ".format(epoch+1 , batch_id , batch_loss, np.sum(kld)))
        
        print_ppx = loss_sum
        print_kld = kld_sum/total_batch
        print('| Epoch train: {:d} |'.format(epoch+1), 
               '| Perplexity: {:.5f}'.format(print_ppx),
               '| KLD: {:.5}'.format(print_kld))
        if epoch % save_step == 0:
            save_model(vae , model_name =  type(vae).__name__ , global_step = epoch)


    return vae

if __name__ == "__main__":


    from sklearn.datasets import fetch_20newsgroups
    twenty_train = fetch_20newsgroups(subset='train')   
    data_ = twenty_train.data
    print "Download 20 news group data completed"
    A = TextLoader(data_ , min_count = 25)
    batch_size = 100
        # restricting memory usage, TensorFlow is greedy and will use all memory otherwise
    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # initialize the Session

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
        vae = NVDM(sess , len(A.vocab), 50, [500 , 500] ,  
                         transfer_fct=tf.nn.tanh , output_activation=tf.nn.softmax,
                         batch_size=100, initializer=xavier_init )

        Model = train_the_model(vae)
            
        
