# %%
from graph_nets import utils_tf
from graph_nets import utils_np
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
# from graph_nets.demos import models
import my_models, Utils
import glob,os
import warnings

warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# %%
n_sample = 5
variation_type = 'all'
trained_weights = 'path to trained weights'
input_file = 'Path to test dataset'
J_scaling = 1e4
start_time = time.time()
# Initial graphs and the targets for eveluating loss
initial_test_graphs,target_test_graphs = Utils.J_single_crack_test_inputs(input_file,J_scaling) 
print(time.time() - start_time)

# %%
tf.reset_default_graph()

input_graphs_dict,target_graphs_dict = Utils.create_dicts('single_crack', True)

processing_steps = 1

# Input and target placeholders.
input_placeholder = utils_tf.placeholders_from_data_dicts(input_graphs_dict)
target_placeholder = utils_tf.placeholders_from_data_dicts(target_graphs_dict)

# Instantiate the model.
model = my_models.EncodeProcessDecode(edge_output_size=1)

# # A list of outputs, one per processing step.
output_ops_val = model(input_placeholder, processing_steps)

# # Test/generalization loss.
loss_ops_val = Utils.create_loss_ops(target_placeholder, output_ops_val, True)
loss_op_val = loss_ops_val[-1]  # Loss from final processing step.

# # Lets an iterable of TF graphs be output from a session as NP graphs.
input_placeholder,target_placeholder = Utils.make_all_runnable_in_session(input_placeholder,target_placeholder)

# %%
try:
  sess.close()
except NameError:
  pass
sess = tf.Session()

restorer = tf.train.Saver()
restorer.restore(sess, tf.train.latest_checkpoint(trained_weights))

# %%
loss_list = []
error_list = []
J_accuracies = []
dJ_mean_accuracy = []
dJ_accuracy = []
angle_list = []
gt_list = []
pred_list = []
dr_list = []
d8_list = []
rescale = 1e4/J_scaling
for i,(input,target) in enumerate(zip(initial_test_graphs,target_test_graphs)):
    feed_dict = {input_placeholder:input,target_placeholder:target}
    # Testing
    test_values = sess.run({
                            "input": input_placeholder,
                            "target": target_placeholder,
                            "outputs": output_ops_val
                            }, 
                            feed_dict=feed_dict)
    # J_pred = np.cumsum(test_values['outputs'][-1].globals)
    # J = np.cumsum(target.globals)
    pred = test_values['outputs'][-1].edges*rescale
    pred[pred<0] = 0.
    gt = target.edges*rescale
    gt_list.append(gt)
    dr_list.append(test_values['target'].nodes[:,0])
    d8_list.append(test_values['target'].nodes[:,1]*np.pi/180)
    pred_list.append(pred)
    steps = np.arange(len(gt))
    fig = plt.figure()
    plt.plot(steps,pred,'-b',label='predictions')
    plt.plot(steps,gt,'--g',label='ground trouth')
    plt.legend()
    plt.xlabel('Edge',fontsize = 15, weight = 'bold')
    plt.ylabel('dJ $(KJ/m^2)$',fontsize = 15, weight = 'bold')
    plt.xticks(fontsize = 15) 
    plt.yticks(fontsize = 15)
    fig.savefig('path to save dir', bbox_inches = "tight") #
    J_accuracy = 1 - np.abs(np.sum(pred)-np.sum(gt))/np.sum(gt)
    J_accuracies.append(J_accuracy)
    dJ_error = np.abs(pred - gt)
    dJ_relative_error = np.abs(pred - gt)/gt
    dJ_accuracy.append(1-dJ_relative_error)
    dJ_mean_accuracy.append(1 - np.mean(dJ_relative_error))
    angle_list.append(test_values['target'].nodes[:,1]*np.pi/180)
# J_local_losses = np.concatenate(J_local_losses) 
dJ_mean_accuracy = np.array(dJ_mean_accuracy) 
print('Mean J accuracy for',len(J_accuracies),'cracks is', np.mean(J_accuracies)*100, '%')
print('Mean dJ local accuracy is', np.mean(dJ_mean_accuracy[dJ_mean_accuracy>0])*100,'%')      




 





