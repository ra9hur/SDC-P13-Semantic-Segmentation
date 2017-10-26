import os.path
import tensorflow as tf
from tensorflow.contrib import slim
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np

import scipy.misc
from moviepy.editor import VideoFileClip
import mobilenet_v1 as mnet


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    
    input_tensor_name = 'image_input:0'
    layer3_out_tensor_name = 'MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6:0'
    layer4_out_tensor_name = 'MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6:0'
    layer7_out_tensor_name = 'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6:0'
    
    graph = sess.graph

    #for i in graph.get_operations():
    #    print (i.name)

    image_input = graph.get_tensor_by_name(input_tensor_name)
    layer3_out = graph.get_tensor_by_name(layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(layer7_out_tensor_name) 
    
    return image_input, layer3_out, layer4_out, layer7_out
#tests.test_load_vgg(load_vgg, tf)


def layers(layer3_out, layer4_out, layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the mobilenet layers.
    :param layer7_out: TF Tensor for Layer 3 output
    :param layer4_out: TF Tensor for Layer 4 output
    :param layer3_out: TF Tensor for Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    ## fcn layers
    print ("layer3_out",layer3_out.get_shape().as_list())
    print ("layer4_out",layer4_out.get_shape().as_list())
    print ("layer7_out",layer7_out.get_shape().as_list())

    layer3_1x1 = tf.layers.conv2d(layer3_out, num_classes, 1, strides=(1,1), padding="SAME", name="layer3_fcn", 
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    print ("layer3_1x1",layer3_1x1.get_shape().as_list())

    layer4_1x1 = tf.layers.conv2d(layer4_out, num_classes, 1, strides=(1,1), padding="SAME", name="layer4_fcn", 
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    print ("layer4_1x1",layer4_1x1.get_shape().as_list())

    layer7_1x1 = tf.layers.conv2d(layer7_out, num_classes, 1, strides=(1,1), padding="SAME", name="layer7_fcn", 
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    print ("layer7_1x1",layer7_1x1.get_shape().as_list())

    # Upscale 1
    layer7_up = tf.layers.conv2d_transpose(layer7_1x1, num_classes, 4, strides=(2, 2), padding="SAME", name="layer7_up", 
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    print ("layer7_up",layer7_up.get_shape().as_list())

    # Add Skip Connections 1
    # make sure the shapes are the same!
    layer7_skip = tf.add(layer7_up, layer4_1x1, name="layer7_skip")

    print ("layer7_skip",layer7_skip.get_shape().as_list())

    # Upscale 2
    layer4_up = tf.layers.conv2d_transpose(layer7_skip, num_classes, 4, strides=(2, 2), padding="SAME", name="layer4_up",  
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    print ("layer4_up",layer4_up.get_shape().as_list())

    # Add Skip Connections 2
    # make sure the shapes are the same!
    layer4_skip = tf.add(layer4_up, layer3_1x1, name="layer4_skip")

    print ("layer4_skip",layer4_skip.get_shape().as_list())

    # Upscale 3
    layer3_up = tf.layers.conv2d_transpose(layer4_skip, num_classes, 16, strides=(8, 8), padding="SAME", name="layer3_up",  
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    print ("layer3_up",layer3_up.get_shape().as_list())

    return layer3_up
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    labels = tf.reshape(correct_label, [-1, num_classes])
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def augment_data(images, gt_images):
    """Augument data by some image transformation,
    return pair of image patches..
    Only support horizontal flipping for now
    """
    # flipping horizontally
    hf_images = images[:, :, ::-1, :]
    hf_gt_images = gt_images[:, :, ::-1, :]

    # flipping vertically - it looks a little weird, but the intention
    # is to force the model focus on patterns such as texture, than shapes 
    # or orientations 
    ## Experiments show that it doesn't really help too much
    vf_images = images[:, ::-1, :, :]
    vf_gt_images = gt_images[:, ::-1, :, :]

    aug_images = np.concatenate([vf_images, images, hf_images], axis=0)
    aug_gt_images = np.concatenate([vf_gt_images, gt_images, hf_gt_images], axis=0)
    return aug_images, aug_gt_images


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    for epoch in range(epochs):
        batches = get_batches_fn(batch_size)
        for b, (seed_images, seed_gt_images) in enumerate(batches):
            
            ## for the intrusive test `tests.test_train_nn`
            if seed_images.ndim == 4:
                images, gt_images = augment_data(seed_images, seed_gt_images)
            else: # for `tests.test_train_nn(train_nn)`
                images, gt_images = seed_images, seed_gt_images

            _, loss_val = sess.run([train_op, cross_entropy_loss],
                                    feed_dict={input_image: images,
                                               correct_label: gt_images,
                                               keep_prob: 0.5,
                                               learning_rate: 1e-4 })
            if b % 10 == 0:
                print("epoch %i batch %i loss=%.3f" % (epoch, b, loss_val))

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    n_epochs = 40    # 5
    batch_size = 4
    data_dir = './data2'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    # helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/


    is_training = False
    weight_decay = 0.0004
    arg_scope = mnet.mobilenet_v1_arg_scope(is_training=is_training, weight_decay=weight_decay)

    inp = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="image_input")
    keep_prob = 0.5
    
    with slim.arg_scope(arg_scope):
        logits, _ = mnet.mobilenet_v1(inp, dropout_keep_prob=keep_prob, is_training=is_training)

    rest_var = slim.get_variables_to_restore()

    #print (rest_var)


    with tf.Session() as sess:
        # Path to vgg model
        #vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        
        saver = tf.train.Saver(rest_var)
        saver.restore(sess, './data2/vgg/mobilenet_v1_1.0_224.ckpt')
        print ('Restored')
        
        input_image, layer3_out, layer4_out, layer7_out = load_vgg(sess)
        
        # TODO: Train NN using the train_nn function
        model_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        target = tf.placeholder(dtype=tf.float32, shape=[None, None, None, num_classes])

        learning_rate = tf.placeholder(dtype=tf.float32)
        keep_prob = tf.placeholder(dtype=tf.float32)
        
        logits, train_op, loss = optimize(model_output, target, learning_rate, num_classes)
        
        sess.run(tf.global_variables_initializer())
        
        train_nn(sess, n_epochs, batch_size, get_batches_fn, 
                 train_op, loss, input_image, target, keep_prob, learning_rate)
        
        saver = tf.train.Saver()
        saver.save(sess, "./models/model.ckpt")
        saver.export_meta_graph("./models/model.meta")
        tf.train.write_graph(sess.graph_def, "./models/", "model.pb", False)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        def process(im):

            image = scipy.misc.imresize(im, image_shape)

            im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, input_image: [image]})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            return np.array(street_im)


        white_output = 'project_out.mp4'
        clip1 = VideoFileClip("project_video.mp4")
        white_clip = clip1.fl_image(process) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)

        ## Tensorboard Logs
        #writer = tf.summary.FileWriter("./tensorboard/1")
        #writer.add_graph(sess.graph)
        
        # From command prompt
        # tensorboard --logdir "./tensorboard/1"




if __name__ == '__main__':
    
    run()



