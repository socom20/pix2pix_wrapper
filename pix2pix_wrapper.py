from pix2pix import *

Inputs  = collections.namedtuple("Inputs", "raw_inputs, inputs, targets")
Outputs = collections.namedtuple("Outputs", "converted_inputs, converted_targets, converted_outputs, display_fetches")

class Model_params():
    def __init__(self, **args):
        """ You can define son parameters in **args to replace the defaults.
            Example:
                    params = Model_params(mode='tran')"""

        self.param_list = ['mode', 'input_dir','output_dir','checkpoint','output_filetype',
                           'aspect_ratio','batch_size', 'beta1','display_freq','flip','gan_weight',
                           'l1_weight','lab_colorization','lr','max_epochs','max_steps','ndf','ngf','progress_freq',
                           'save_freq','scale_size', 'seed', 'separable_conv', 'summary_freq', 'trace_freq', 'which_direction']
                           
        self.load_defaults()
                           
        for k,v in args.items():
            if k not in self.param_list:
                raise Exception(' - ERROR, Model_params: parameter "{}" not supported by the pox2pix model.')
            setattr(self, k, v)
            
        return None
    
    def load_defaults(self):
        self.mode='test'
        self.input_dir='./A'
        self.output_dir='./B'
        self.checkpoint='./sketchtophoto_train'
        self.output_filetype='png'
        self.aspect_ratio=1.0
        self.batch_size=8
        self.beta1=0.5
        self.display_freq=0
        self.flip=True
        self.gan_weight=1.0
        self.l1_weight=100.0
        self.lab_colorization=False
        self.lr=0.0002
        self.max_epochs=None
        self.max_steps=None
        self.ndf=64
        self.ngf=64
        self.progress_freq=50
        self.save_freq=5000
        self.scale_size=286
        self.seed=None
        self.separable_conv=False
        self.summary_freq=100
        self.trace_freq=0
        self.which_direction='AtoB'
        
        return None


    def __str__(self):
        s = ' Model parameters:'
        for k,v in self.param_list:
            s += '\n - {}\t:\t{}'.format(k,v)

        return s
    
    def __repr__(self):
        return self.__str__()

def load_examples(a):

    # This will be the input variable for tf. It will give the modelo th avility of use:
    # sess.run( target, feed_dict={raw_inputs: batch_img})
    raw_inputs = tf.placeholder( tf.float32, shape=[None, None, None, 3], name='raw_inputs')
    
    def transform_image(raw_input):
        with tf.name_scope("load_images"):
            
            
            if a.lab_colorization:
                # load color and brightness from image, no B image exists here
                lab = rgb_to_lab(raw_input)
                L_chan, a_chan, b_chan = preprocess_lab(lab)
                a_images = tf.expand_dims(L_chan, axis=2)
                b_images = tf.stack([a_chan, b_chan], axis=2)
            else:
                # break apart image pair and move to range [-1, 1]
                width = tf.shape(raw_input)[1] # [height, width, channels]
                a_images = preprocess( raw_input )
                b_images = preprocess( raw_input )

        if a.which_direction == "AtoB":
            inputs, targets = [a_images, b_images]
        elif a.which_direction == "BtoA":
            inputs, targets = [b_images, a_images]
        else:
            raise Exception("invalid direction")

        # synchronize seed for image operations so that we do the same operations to both
        # input and output images
        seed = random.randint(0, 2**31 - 1)
        def transform(image):
            r = image
            if a.flip:
                r = tf.image.random_flip_left_right(r, seed=seed)

            # area produces a nice downscaling, but does nearest neighbor for upscaling
            # assume we're going to be doing downscaling here
            r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

            offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
            if a.scale_size > CROP_SIZE:
                r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
            elif a.scale_size < CROP_SIZE:
                raise Exception("scale size cannot be less than crop size")
            return r

        with tf.name_scope("input_images"):
            input_images = transform(inputs)

        with tf.name_scope("target_images"):
            target_images = transform(targets)

        return (input_images, target_images)

##    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    inputs_batch, targets_batch = tf.map_fn(transform_image, raw_inputs, dtype=(tf.float32,tf.float32) )


    return Inputs(raw_inputs=raw_inputs,
                  inputs=inputs_batch,
                  targets=targets_batch)



        
def load_model(args=None):
    if args is not None:
        global a
        a = args

        
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)
    
    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)


    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    else:
        print(' - Modo no implementado', file=sys.stderr)
        return None


    inputs_t = load_examples(a)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(a, inputs_t.inputs, inputs_t.targets)

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(inputs_t.targets, inputs_t.inputs)
            outputs = augment(model.outputs, inputs_t.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(inputs_t.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(inputs_t.inputs, inputs_t.targets)
            targets = deprocess(inputs_t.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(inputs_t.inputs)
        targets = deprocess(inputs_t.targets)
        outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }


    outputs_t = Outputs(converted_inputs, converted_targets, converted_outputs, display_fetches)
    
    return inputs_t, outputs_t




class pix2pix_wrapper():
    """ This class wrapper functions,
        This functions gant the avility os using the method predict to the model pix2pix.

        The pix2pix model is based on the implementation of:
                https://github.com/affinelayer/pix2pix-tensorflow
        """

    def __init__(self, checkpoint_dir='./model_checkpoint'):
        self.checkpoint_dir = checkpoint_dir
        self.params         = Model_params(mode='test',
                                           checkpoint=checkpoint_dir)
        self.sess           = None

        if not os.path.exists(self.checkpoint_dir):
            raise Exception(' - ERROR, pix2pix_wrapper: checkpoint_dir = {} not exists'.format(self.checkpoint_dir))
    
        return None


    def load_model(self):
        # Clear the graph
        tf.reset_default_graph()

        # Create new Session
        self.sess = tf.Session()
        
        # Load model
        self.inputs, self.outputs = load_model(self.params)

##        self.sess.run(tf.global_variables_initializer())        
        print(" - Loading model from checkpoint ...")

        # Loading model parameters
        checkpoint_filename = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.saver.restore(self.sess, checkpoint_filename)

        print(" - Checkpoint Loaded !!!")

        return None
        
    

    def predict(self, x):
        """ A batch of images, with shape (None, None, None, 3) """

        img_trg = self.sess.run(self.outputs.converted_outputs,
                                feed_dict={self.inputs.raw_inputs:x})
        
        return img_trg

    

        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2


    model = pix2pix_wrapper()
    model.load_model()

    i_frame = 0
    fps     = 0.0
    t_start = time.time()
    while(True):
        img = np.array(Image.open('./A/input_img.jpg'))
        img_v = img.astype(np.float32)[np.newaxis,...] / 255

        # Modelo prediction:
        img_trg = model.predict(img_v)

        
        img = cv2.cvtColor(img_trg[0], cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (800, 800))
        
        cv2.imshow('model output', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        i_frame += 1
        
        if i_frame % 10 == 0:
            t_end = time.time()
            fps = 10.0/(t_end - t_start)
            t_start = t_end


    











