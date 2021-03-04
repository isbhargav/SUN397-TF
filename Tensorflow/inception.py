
# coding: utf-8

# In[2]:


#loading path library to load data
from pathlib import Path
from keras_preprocessing.image import ImageDataGenerator
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.layers import Flatten,Dense,Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2


# In[3]:


p = Path('.')
p = p/'data/SUN397/'


# In[4]:

#loading data using image genrator
df_train = pd.read_csv('./Partitions/Training_05.txt',names=['img'])
df_train['label'] = df_train['img'].apply(lambda x: '/'.join(x.split('/')[:-1]))
df_train['img'] = df_train['img'].apply(lambda x: 'data/SUN397'+x)
#df_train.head()

df_train = df_train.sample(frac=1).reset_index(drop=True)
# In[5]:

#loading data using image genrator
df_test = pd.read_csv('./Partitions/Testing_05.txt',names=['img'])
df_test['label'] = df_test['img'].apply(lambda x: '/'.join(x.split('/')[:-1]))
df_test['img'] = df_test['img'].apply(lambda x: 'data/SUN397'+x)

#df_test.head()


# In[6]:

#loading data using image genrator
from keras.applications.vgg16 import preprocess_input

train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rotation_range=40,rescale=1. / 255,
                                   shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)


# In[7]:


# getting training and testing data in imagegenrator
train_data_gen = train_datagen.flow_from_dataframe(dataframe=df_train,x_col='img',y_col='label',class_mode='categorical',subset='training',batch_size=25,target_size=(299,299),color_mode="rgb",shuffle=True)
val_data_gen = train_datagen.flow_from_dataframe(dataframe=df_train,x_col='img',y_col='label',class_mode='categorical',subset='validation',batch_size=25,target_size=(299,299),color_mode="rgb",shuffle=True)
test_data_gen = test_datagen.flow_from_dataframe(dataframe=df_test,x_col='img',y_col='label',class_mode='categorical',batch_size=1,target_size=(299,299),color_mode="rgb")





# In[8]:

#load model
model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299,299,3),classes=397)


# In[ ]:





# In[10]:

#Add dense layers

last_layer = Flatten(name='flatten')(model.output)
last_layer =Dense(1024,activation='relu',name='fc1')(last_layer)
last_layer = Dropout(0.5)(last_layer)
last_layer =Dense(512,activation='relu',name='fc2')(last_layer)
last_layer = Dropout(0.2)(last_layer)
last_layer =Dense(397,activation='softmax',name='predict')(last_layer)

update_model = Model(inputs=model.input,outputs=last_layer)


# In[11]:


from keras import optimizers

update_model.compile(optimizer=optimizers.RMSprop(lr=0.0002), loss='categorical_crossentropy', metrics=['accuracy'] )


# In[12]:


from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping


#setting up logger to log data in csv file
csv_logger = CSVLogger('logs/finetune-inception-logs.csv', append=True, separator=',')


filepath = "weights/saved-inception-finetune-model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint,csv_logger]


# In[13]:

# update_model = load_model('weights/saved-inception-finetune-model.hdf5')
#fine-tune the model
update_model.fit_generator(train_data_gen,steps_per_epoch=train_data_gen.samples//train_data_gen.batch_size,
                                  epochs=4,validation_data=val_data_gen,validation_steps=val_data_gen.samples//val_data_gen.batch_size,
                                   callbacks = callbacks_list,verbose=2)


# In[14]:


for layer in update_model.layers[:770]:
    layer.trainable = False
for layer in update_model.layers[770:]:
    layer.trainable = True


# In[ ]:





# In[15]:





# In[ ]:





# In[10]:


from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping


#setting up logger to log data in csv file
csv_logger = CSVLogger('logs/inception-final-logs.csv', append=True, separator=',')


filepath = "weights/saved-inception-model-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint,csv_logger]


# In[11]:


update_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[12]:


#update_model = load_model('weights/saved-inception-model-13-0.54.hdf5')
    
history = update_model.fit_generator(train_data_gen,steps_per_epoch=train_data_gen.samples//train_data_gen.batch_size,epochs=10,validation_data=val_data_gen,validation_steps=val_data_gen.samples//val_data_gen.batch_size,                                    callbacks = callbacks_list,verbose=2,shuffle=True)
# In[ ]:



#update_model.save('final_inception_model.hdf5')
update_model = load_model('weights/saved-inception-model-10-0.63.hdf5')

# In[13]:

acc_list =[]

for trial in range(3):
    
    STEP_SIZE_TEST=test_data_gen.n//test_data_gen.batch_size
    test_data_gen.reset()

    loss, acc =update_model.evaluate_generator(test_data_gen,
    steps=STEP_SIZE_TEST,
    verbose=2)
    print('Loss: ', loss,' Acc ', acc)
    acc_list.append(acc)


# In[38]:


print('Average Testing accuracy for 3 Trials ', sum(acc_list)/3.0 )


# In[39]:





# In[ ]:




