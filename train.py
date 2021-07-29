"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse

import os
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from PIL import Image
from keras.optimizers import Adam
from keras.models import load_model, Model
from keras.layers import Input,Lambda
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from Model.models.Yolo3_detection_head import preprocess_true_boxes, MobileNetV1_yolo, yolo_loss


from Model.utils.draw_boxes import draw_boxes


# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to text file containing object array 'boxes' and images paths",
    default='train.txt')

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('Pre_trained_and_Anchor', '20180513anchors.txt'))
	
argparser.add_argument(
    '-m',
    '--model_name',
    help='which model are used for training',
    default="mobilnetV1")

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default="classes.txt") # 'DATA\\underwater_classes.txt'

# Default anchor boxes
YOLO_ANCHORS = np.array(((10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)))


model_dict = {"mobilnetV1":MobileNetV1_yolo}

def _main(args):
    data_path = os.path.expanduser(args.data_path)  
    classes_path = os.path.expanduser(args.classes_path) 
    anchors_path = os.path.expanduser(args.anchors_path)
    model_name = args.model_name
    image_size = (416,416)
    if model_name == "mobilnetV1":
        image_size = (224,320)
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    anchors = YOLO_ANCHORS
	
	
    # ==================== creat the model ============================


    alpha = 0.5
    model_body, model = create_tiny_model(model_name = model_name, alpha=alpha, input_shape=image_size, anchors=anchors, num_classes=num_classes, freeze_body=True, light_head=False)
    print(model.summary())



    train(
        model,
        class_names,
        model_name,
        anchors,
        data_path,
        image_size=image_size
    )



def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS
		

def create_tiny_model(model_name, alpha, input_shape, anchors, num_classes, freeze_body=False, light_head=True ):
    '''create the training model, for Tiny YOLOv3'''
	
    image_input = Input(shape=(input_shape[0], input_shape[1], 2))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = model_dict[model_name](image_input, num_anchors//2, num_classes, alpha, freeze_body, light_head)
	
	
    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})( [*model_body.output, *y_true])
		
    model = Model([model_body.input, *y_true], model_loss)

    return model_body, model
	
	
def get_random_data(annotation_line, input_shape, random=True, max_boxes=20):
    '''random preprocessing for real-time data augmentation'''
	
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    scale_w = w/iw
    scale_h = h/ih
    image = image.resize((w,h))
    image_data = np.array(image)/255.
    image_data = np.expand_dims(image_data, axis=2)
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        if len(box)>max_boxes: 
            box = box[:max_boxes]
        box[:, [0,2]] = box[:, [0,2]]*scale_w 
        box[:, [1,3]] = box[:, [1,3]]*scale_h 
        box_data[:len(box)] = box

    return image_data, box_data


def get_random_sub(annotation_line, input_shape):

    name = annotation_line.split()[0]
    name = name[:-4]+"_sub"+".pgm"
    sub_img = Image.open(name)
    h, w = input_shape


    sub_img = sub_img.resize((w,h))
    sub_img = np.array(sub_img)/255.
    sub_img = np.expand_dims(sub_img, axis=2)
    return sub_img 

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            sub_img = get_random_sub(annotation_lines[i], input_shape)
			
            image = np.concatenate((image,sub_img),axis=2)			
            image_data.append(image)

            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)
		

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)





def train(model, class_names, model_name, anchors, data_path, validation_split=0.1, image_size=(224,320)):

    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
	
    # =======================  data preparation =======================================
    num_classes = len(class_names)
    with open(data_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*validation_split)
    num_train = len(lines) - num_val
	
	# ======================= log and early_stopping =============================
    from datetime import datetime
    dateTimeObj = datetime.now()

    log_dir = "logs/{}_{}_{}_{}_{}_{}/".format(dateTimeObj.year,dateTimeObj.month,dateTimeObj.day,dateTimeObj.hour,dateTimeObj.minute,dateTimeObj.second)   #TIME~~~~
    os.makedirs(log_dir)
    logging = TensorBoard(log_dir=log_dir)

    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto')
	

    model.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred})	
	
    batch_size	= 32
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, image_size, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, image_size, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=20,
                initial_epoch=0,
                callbacks=[logging, checkpoint])			  
    model.save_weights(log_dir + 'trained_weights_stage_1.h5')
	
	
	
    for i in range(len(model.layers)):
            model.layers[i].trainable = True			  
			  
			  

    model.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    batch_size = 16 

    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, image_size, anchors, num_classes),
                        steps_per_epoch=max(1, num_train//batch_size),
                        validation_data=data_generator_wrapper(lines[num_train:], batch_size, image_size, anchors, num_classes),
                        validation_steps=max(1, num_val//batch_size),
                        epochs=200,
                        initial_epoch=20,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'trained_weights_final.h5')





if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
