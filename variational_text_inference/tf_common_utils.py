
import tensorflow as tf
import os

def save_model( model , checkpoint_dir = os.getcwd() , save_dir = 'save_my_model' ,
                model_name = 'model' , global_step=None):

    print "Saving checkpoints"
    checkpoint_dir = os.path.join(checkpoint_dir, save_dir)
    if not os.path.exists(checkpoint_dir):
        print "Creating Model Saving Directory"
        os.makedirs(checkpoint_dir)

    model_path = os.path.join(checkpoint_dir, model_name)
    print "MODEL PATH is {} ".format(model_path)
    model.saver.save(model.SESS, model_path ,  global_step=global_step)
    print "Saved in {}".format(checkpoint_dir)
        
def load_model(model , checkpoint_dir = os.getcwd() , save_dir = 'save_my_model' ):

    print(" [*] Loading checkpoints...")
    print "Model dir is {}".format(save_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, save_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print " ckpt name " , ckpt_name
        model_path = os.path.join(checkpoint_dir, ckpt_name)
        model.saver.restore(model.SESS, model_path)
        print(" [*] Load SUCCESS")
        return True
    else:
        print(" [!] Load failed...")
        return False
    
