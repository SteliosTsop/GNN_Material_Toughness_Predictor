# %%
from graph_nets import utils_tf
from graph_nets import utils_np
import tensorflow as tf
import numpy as np
import time
# from graph_nets.demos import models
import my_models, Utils
import glob,os
import warnings

warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# %%   
start_time = time.time()
# Initial graphs and the targets for eveluating loss
initial_graphs,target_graphs = Utils.J_single_crack_inputs() 
print(time.time() - start_time)

# %%
best_loss = 0.0199

  
tf.reset_default_graph()

input_graphs_dict,target_graphs_dict = Utils.create_dicts('single_crack', True)

processing_steps = 1

# Input and target placeholders.
input_placeholder = utils_tf.placeholders_from_data_dicts(input_graphs_dict)
target_placeholder = utils_tf.placeholders_from_data_dicts(target_graphs_dict)

# Connect the data to the model.
# Instantiate the model.
model = my_models.EncodeProcessDecode(edge_output_size=1)

# # A list of outputs, one per processing step.
output_ops_tr = model(input_placeholder, processing_steps)
output_ops_val = model(input_placeholder, processing_steps)

# Training loss.
loss_ops_tr = Utils.create_loss_ops(target_placeholder, output_ops_tr, True)

# # Loss across processing steps.
loss_op_tr = sum(loss_ops_tr) / processing_steps
# # Test/generalization loss.
loss_ops_val = Utils.create_loss_ops(target_placeholder, output_ops_val, True)
loss_op_val = loss_ops_val[-1]  # Loss from final processing step.


# Optimizer. 
learning_rate = 1e-6
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr)
saver = tf.train.Saver(max_to_keep=40)

# Lets an iterable of TF graphs be output from a session as NP graphs.
input_placeholder,target_placeholder = Utils.make_all_runnable_in_session(input_placeholder,target_placeholder)

# %%
try:
  sess.close()
except NameError:
  pass
sess = tf.Session()

# sess.run(tf.global_variables_initializer())
# graph = tf.get_default_graph()

# # If you want to continue from previous epoch
restorer = tf.train.import_meta_graph('path to .meta')
restorer.restore(sess, tf.train.latest_checkpoint('path to checkpnt'))

last_epoch = 409

# %%
num_epochs = 901
n_structures = len(initial_graphs) 
split = int(0.025*n_structures)
print(split)
ep_train_loss = []
ep_val_loss = []
for epoch in range(last_epoch, num_epochs):
    start_time = time.time()
    last_epoch = epoch
    struc_train_losses = []
    struc_val_losses = []
    
    for i,(input,target) in enumerate(zip(initial_graphs,target_graphs)):
        feed_dict = {input_placeholder:input,target_placeholder:target}
        if i<=split:
            # Training
            train_values = sess.run({
                                     "step": step_op, 
                                     "target": target_placeholder,
                                     "loss": loss_op_tr,
                                     "outputs": output_ops_tr
                                    },
                                    feed_dict=feed_dict)
            struc_train_losses.append(np.mean(train_values["loss"])) 
        else:
            # Validation
            val_values = sess.run({
                                   "target": target_placeholder,
                                   "loss": loss_op_val,
                                   "outputs": output_ops_val
                                  },
                                  feed_dict=feed_dict)
            struc_val_losses.append(np.mean(val_values["loss"]))
        # Store losses
        mean_train_loss = np.mean(np.array(struc_train_losses))
        mean_val_loss = np.mean(np.array(struc_val_losses))


    ep_train_loss.append(mean_train_loss)
    ep_val_loss.append(mean_val_loss)
    elapsed_time = time.time() - start_time
    print("# {:03d}, Train Loss {:.6f}, Val Loss {:.6f}, Elapsed time {:.1f}".format(epoch, mean_train_loss, mean_val_loss,elapsed_time))


    if mean_val_loss<best_loss: 
        save_dir = 'save dir')
        saver.save(sess, save_dir)
        best_loss=mean_val_loss
        print("Model saved")


