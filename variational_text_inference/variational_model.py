
from text_loader_utils import TextLoader
import cPickle
import numpy as np
import tensorflow as tf
import os
from vector_utils import find_norm

np.random.seed(0)
tf.set_random_seed(0)


class Variational_Document_Model(object):
    
    '''
    NVDM Model - as described in Neural Variational Inference for Text Processing

    '''

    
    def __init__(self, the_sess , input_dim , hidden_dim , encoder_hidden_dim = [] , generator_hidden_dim = [] , batch_size = 100 ,
                 initializer=tf.random_normal,transfer_fct = tf.nn.relu , output_activation = tf.nn.sigmoid,
                 learning_rate = 0.001 , mode = 'gather' , negative_sampling_loss = False):
        
        self.transfer_fct = transfer_fct
        self.output_activation = output_activation
        self.mode = mode

        if negative_sampling_loss:
            self.negative_sampling_loss = True
            self.output_activation = tf.nn.sigmoid
            self.mode = 'negative'
        
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.generator_hidden_dim = generator_hidden_dim
        self.initializer = initializer
        self.dynamic_batch_size = tf.placeholder(tf.int32 , shape=None)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.x = tf.placeholder(tf.float32, [None, input_dim])
        
        self.mask = tf.placeholder(tf.float32, [None, input_dim])
        self.gather_mask = tf.placeholder(tf.int64, [None])
        self._create_network()
        self._create_loss_optimizer()
        # Initializing the tensor flow variables
        self.saver = tf.train.Saver()
        self.init = tf.initialize_all_variables()

        # Launch the session
        self.sess = the_sess
        
    
    def _train(self):
        self.sess.run(self.init)
    
    def _initialize_weights(self):
        
        Weights_encoder = {}
        Biases_encoder = {}
        
        Weights_generator = {}
        Biases_generator = {}
        
        with tf.variable_scope("encoder"):
            
            for i in xrange(len(self.encoder_hidden_dim)):
                if i == 0 :
                    Weights_encoder['W_{}'.format(i)] = tf.Variable(self.initializer(self.input_dim, self.encoder_hidden_dim[i]))
                    Biases_encoder['b_{}'.format(i)] = tf.Variable(tf.zeros([self.encoder_hidden_dim[i]], dtype=tf.float32))
    
                else:
                    Weights_encoder['W_{}'.format(i)] = tf.Variable(self.initializer(self.encoder_hidden_dim[i-1], self.encoder_hidden_dim[i]))
                    Biases_encoder['b_{}'.format(i)] = tf.Variable(tf.zeros([self.encoder_hidden_dim[i]], dtype=tf.float32))
                    
            Weights_encoder['out_mean'] = tf.Variable(self.initializer(self.encoder_hidden_dim[i], self.hidden_dim))
            Weights_encoder['out_log_sigma'] = tf.Variable(self.initializer(self.encoder_hidden_dim[i], self.hidden_dim))
                                                          
            Biases_encoder['out_mean'] =  tf.Variable(tf.zeros([self.hidden_dim], dtype=tf.float32))
            Biases_encoder['out_log_sigma'] = tf.Variable(tf.zeros([self.hidden_dim], dtype=tf.float32))
                    
        with tf.variable_scope("generator"):
            
#             for i in xrange(len(self.generator_hidden_dim)):
#                 if i == 0 :
#                     Weights_generator['W_{}'.format(i)] = tf.Variable(self.initializer(self.hidden_dim, self.generator_hidden_dim[i]))
#                     Biases_generator['b_{}'.format(i)] = tf.Variable(tf.zeros([self.generator_hidden_dim[i]]), dtype=tf.float32)
                
#                 else:
#                     Weights_generator['W_{}'.format(i)] = tf.Variable(self.initializer(self.generator_hidden_dim[i-1], self.generator_hidden_dim[i]))
#                     Biases_generator['b_{}'.format(i)] = tf.Variable(tf.zeros([self.generator_hidden_dim[i]]), dtype=tf.float32)
                                                          
            Weights_generator['out_mean'] = tf.Variable(self.initializer(self.hidden_dim, self.input_dim))
#         Weights_generator['out_log_sigma'] = tf.Variable(self.initializer(self.generator_hidden_dim[i], self.input_dim))

            Biases_generator['out_mean'] =  tf.Variable(tf.zeros([self.input_dim], dtype=tf.float32))
#         Biases_generator['out_log_sigma'] = tf.Variable(tf.zeros([self.input_dim], dtype=tf.float32))
                                                        
            return Weights_encoder , Weights_generator , Biases_encoder , Biases_generator
                    
    def _create_network(self):

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.Weights_encoder , self.Weights_generator , self.Biases_encoder , self.Biases_generator = self._initialize_weights()
        self.z_mean, self.z_log_sigma_sq = self._encoder_network(self.Weights_encoder, self.Biases_encoder)
        
        # Draw one sample z from Gaussian distribution
        
        eps = tf.random_normal((self.dynamic_batch_size, self.hidden_dim), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.X_reconstruction_mean = self._generator_network(self.Weights_generator , self.Biases_generator)

    def _encoder_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        encoder_results = {}

        with tf.variable_scope("enoder_function"):

            for i in xrange(len(weights)-2):
                if i == 0:
                    encoder_results['res_{}'.format(i)] = self.transfer_fct(tf.add(tf.matmul(self.x, weights['W_{}'.format(i)]),
                                                                               biases['b_{}'.format(i)]))
                else:
                    encoder_results['res_{}'.format(i)] = self.transfer_fct(tf.add(tf.matmul(encoder_results['res_{}'.format(i-1)],
                                                                            weights['W_{}'.format(i)]),biases['b_{}'.format(i)]))

            z_mean = tf.add(tf.matmul(encoder_results['res_{}'.format(i)], weights['out_mean']),
                            biases['out_mean'])
            z_log_sigma_sq = tf.add(tf.matmul(encoder_results['res_{}'.format(i)], weights['out_log_sigma']), 
                       biases['out_log_sigma'])
            return (z_mean, z_log_sigma_sq)
        
    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        
        generator_results = {}

        with tf.variable_scope("generator_function"):

#             for i in xrange(len(weights)-2):
#                 if i == 0:
#                     generator_results['res_{}'.format(i)] = self.transfer_fct(tf.add(tf.matmul(self.z, weights['W_{}'.format(i)]),
#                                                                                biases['b_{}'.format(i)]))
#                 else:
#                     generator_results['res_{}'.format(i)] = self.transfer_fct(tf.add(tf.matmul(generator_results['res_{}'.format(i-1)],
#                                                                             weights['W_{}'.format(i)]),biases['b_{}'.format(i)]))
                    
            
#             x_reconstr_mean = self.output_activation(tf.add(-1*(tf.matmul(generator_results['res_{}'.format(i)], weights['out_mean'])), 
#                                  biases['out_mean']))
            if self.mode == 'negative':
                x_reconstr_mean = tf.add(-1*(tf.matmul(self.z, weights['out_mean'])), 
                                 biases['out_mean'])
                return x_reconstr_mean

            x_reconstr_mean = self.output_activation(tf.add(-1*(tf.matmul(self.z, weights['out_mean'])), 
                                 biases['out_mean']))

            return x_reconstr_mean
        
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        
        if self.mode == 'negative':
            self.negative_loss = self.output_activation(self.X_reconstruction_mean * self.mask)
            self.log_recons = tf.log(self.negative_loss + 1e-10)
            self.reconstr_loss = reconstr_loss = -tf.reduce_sum(self.log_recons*self.mask)
        if self.mode == 'gather':
            ####### Collecting the indexes as in Bag of Words
            self.interm_res = tf.log(tf.gather(tf.reshape(self.X_reconstruction_mean , [-1]), self.gather_mask) + 1e-10)
            self.reconstr_loss = reconstr_loss = -tf.reduce_sum(self.interm_res)
        
#         -tf.reduce_sum(tf.log(tf.gather(self.p_x_i, self.x_idx) + 1e-10))
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        self.latent_loss = latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        # Use ADAM optimizer

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        #### Local Clipping
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # gvs = self.optimizer.compute_gradients(self.cost)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # self.main_optimizer = self.optimizer.apply_gradients(capped_gvs)

        ##### Global Clipping
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # tvars = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 1)
            # self.main_optimizer = self.optimizer.apply_gradients(zip(grads, tvars))

    def partial_fit(self, X , dynamic_batch_size , MASK):

        if self.mode == 'gather':
            opt, cost , recons_loss   = self.sess.run((self.optimizer , self.cost , self.reconstr_loss), 
                                  feed_dict={self.x: X , self.dynamic_batch_size:dynamic_batch_size,
                                            self.gather_mask:MASK})
        else:
            opt, cost , recons_loss   = self.sess.run((self.optimizer, self.cost , self.reconstr_loss), 
                                      feed_dict={self.x: X , self.dynamic_batch_size:dynamic_batch_size,
                                                self.mask:MASK})
        return cost , recons_loss 
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.X_reconstruction_mean, 
                             feed_dict={self.z: z_mu})
    
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.X_reconstruction_mean, 
                             feed_dict={self.x: X})
    
    
    def save(self, checkpoint_dir = os.getcwd() , model_dir = 'save_my_model' , global_step=None):
        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__
        

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, 
            os.path.join(checkpoint_dir, model_name), global_step=global_step)
        print "Saved in {}".format(checkpoint_dir)
        
    def load(self,  checkpoint_dir = os.getcwd() , model_dir = 'save_my_model' ):

        print(" [*] Loading checkpoints...")
        print "Model dir is {}".format(model_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        print "checkpoint_dir" , checkpoint_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Load SUCCESS")
            return True
        else:
            print(" [!] Load failed...")
            return False
        
        






