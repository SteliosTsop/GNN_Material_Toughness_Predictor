from zipfile import ZipFile
from graph_nets import utils_np,utils_tf
import numpy as np
import tensorflow as tf


def create_dicts(crack_mode, J_mode):
    i_graphs = []
    t_graphs = []


    i_graphs.append({"globals":np.array([1],dtype='float64'),
                    "nodes": np.ones((1000,2)),  # this vector stores values [x,y] coordinate and the isCrack bool for each node
                    "edges": np.ones((1500,2)),  # this vector stores values [dr,theta] of each edge 
                    "senders": np.ones((1500,),dtype=int),
                    "receivers": np.ones((1500,),dtype=int)})
    t_graphs.append({"globals":np.array([1],dtype='float64'),
                    "nodes": np.ones((1000,2)), # 1hot labeling of each node, according to belonging to the Crack or not
                    "edges": np.ones((1000,1)), # this vector stores values [dr,theta,dJ] of each edge    
                    "senders": np.ones((1500,),dtype=int),
                    "receivers": np.ones((1500,),dtype=int)})


    return i_graphs,t_graphs

def J_single_crack_inputs(batch_size=1):

   
    zip_file = 'path to train dataset'


    z = ZipFile(zip_file)
    files = np.array(z.namelist())

    n = 1000+1
    crack_edges_files = files[2:n+1]
    crack_nodes_files = files[n+2:2*n+1]
    edges_files = files[2*n+2:3*n+1]
    nodes_files = files[3*n+2:4*n+1]
    r_files = files[4*n+2:5*n+1]
    s_files = files[5*n+2:6*n+1]


    batch_init_dicts = []
    batch_target_dicts = []
    initial_graphs = []
    target_graphs = []
    node_num = []

    for i in range(n-1):
        # Load Nodes,Edges,Senders,Receivers and Crack nodes
        f_nodes = z.open(nodes_files[i])
        node_pos = np.loadtxt(f_nodes, delimiter=',')
        n_nodes = len(node_pos)
        # # Normalise positions
        node_pos = node_pos/np.amax(node_pos, axis=0)

        # Load Edges
        f_edges = z.open(edges_files[i])
        edges = np.loadtxt(f_edges, delimiter=',')
        # # Normalize the dr,theta attributes
        edges[:,:2] = edges[:,:2]/np.amax(edges[:,:2], axis=0)
        # Import the indices of the nodes that belong to crack path
        crack = z.open(crack_nodes_files[i])
        crack_nodes = np.loadtxt(crack, dtype=int, delimiter=',')
        crack_nodes = crack_nodes-1
        # Number of crack nodes
        # c = len(crack_nodes)

        # # Senders and receivers of the crack path
        # senders = np.arange(c-1)
        # receivers = np.arange(1,c)

        # Import the indices of the edges that belong to crack path
        crack_e = z.open(crack_edges_files[i])
        crack_edges = np.loadtxt(crack_e, dtype=int, delimiter=',')
        crack_edges = crack_edges-1
    
        for m in [1,2]:
            
            # Features of the crack nodes
            cn = crack_nodes[(m-1)*9:m*9]
            node_features = node_pos[cn,:]

            # Features of the crack edges
            ce = crack_edges[(m-1)*8:m*8]
            edge_features = edges[ce,:]
            J_scaling = 1e4

            # Senders - Receivers
            # Number of crack nodes
            c = len(cn)
            # Senders and receivers of the crack path
            senders = np.arange(c-1)
            receivers = np.arange(1,c)

            J_path = np.sum(edge_features[:,2]*J_scaling)

            # Create a init graph_dict
            initial_graph_dict = {"globals":np.array([i],dtype='float64'),
                                "nodes": node_features, # this array stores values [x,y] coordinate and the isCrack bool for each node
                                "edges": edge_features[:,:2], # this vector stores values [dr,theta] of each edge 
                                "senders": senders,       
                                "receivers": receivers}   

            # Create a list of target_graph dictionaries for each batch
            target_graph_dict = {"globals": np.array([J_path],dtype='float64'),  # this is the total dJ of the entire gt path
                                "nodes": node_features, 
                                "edges": np.expand_dims(edge_features[:,2],axis=1)*J_scaling,  # this vector stores dj values of each crack edge
                                "senders": senders,
                                "receivers": receivers}
            
            initial_graphs.append(utils_np.data_dicts_to_graphs_tuple([initial_graph_dict]))
            target_graphs.append(utils_np.data_dicts_to_graphs_tuple([target_graph_dict]))


        
    return initial_graphs,target_graphs

def J_single_crack_test_inputs(zip_file,J_scaling,n_tests=20):


    z = ZipFile(zip_file)
    files = np.array(z.namelist())

    n = 2000+1
    crack_edges_files = files[2:n+1]
    crack_nodes_files = files[n+2:2*n+1]
    edges_files = files[2*n+2:3*n+1]
    nodes_files = files[3*n+2:4*n+1]
    r_files = files[4*n+2:5*n+1]
    s_files = files[5*n+2:6*n+1]


    batch_init_dicts = []
    batch_target_dicts = []
    initial_graphs = []
    target_graphs = []

    for i in range(n_tests): #(18000,n_tests):#
        # Load Nodes,Edges,Senders,Receivers and Crack nodes
        f_nodes = z.open(nodes_files[i])
        node_pos = np.loadtxt(f_nodes, delimiter=',')
        n_nodes = len(node_pos)
        # # Normalise positions
        node_pos = node_pos/np.amax(node_pos, axis=0)

        # Load Edges
        f_edges = z.open(edges_files[i])
        edges = np.loadtxt(f_edges, delimiter=',')
        # Un-normalized edge features 
        edges_raw = np.empty_like(edges[:,:2])
        edges_raw[:] = edges[:,:2]
        # # Normalize the dr,theta attributes
        edges[:,:2] = edges[:,:2] /np.amax(edges[:,:2], axis=0)
        # Import the indices of the nodes that belong to crack path
        crack = z.open(crack_nodes_files[i])
        crack_nodes = np.loadtxt(crack, dtype=int, delimiter=',')
        crack_nodes = crack_nodes-1


        # Import the indices of the edges that belong to crack path
        crack_e = z.open(crack_edges_files[i])
        crack_edges = np.loadtxt(crack_e, dtype=int, delimiter=',')
        crack_edges = crack_edges-1

        for m in [1,2]:

            # Features of the crack nodes
            cn = crack_nodes[(m-1)*9:m*9]
            node_features = node_pos[cn,:] 

            # Features of the crack edges
            ce = crack_edges[(m-1)*8:m*8]
            edge_features = edges[ce,:]
            raw_edge_features = edges_raw[ce,:]
            # J_scaling = 1e4

            # Senders - Receivers
            # Number of crack nodes
            c = len(cn)
            # Senders and receivers of the crack path
            senders = np.arange(c-1)
            receivers = np.arange(1,c)

            J_path = np.sum(edge_features[:,2]*J_scaling)

            # Create a init graph_dict
            initial_graph_dict = {"globals":np.array([i],dtype='float64'),
                                "nodes": node_features, # this array stores values [x,y] coordinate and the isCrack bool for each node
                                "edges": edge_features[:,:2], # this vector stores values [dr,theta] of each edge 
                                "senders": senders,       
                                "receivers": receivers}                      
            # Create a target graph_dict
            target_graph_dict = {"globals": np.array([J_path],dtype='float64'),  # this is the total dJ of the entire gt path
                                "nodes": raw_edge_features, # this stores the un-normalized edge features for result evaluation purposes
                                "edges": np.expand_dims(edge_features[:,2],axis=1)*J_scaling,  # this vector stores dj values of each crack edge
                                "senders": senders,
                                "receivers": receivers}

            initial_graphs.append(utils_np.data_dicts_to_graphs_tuple([initial_graph_dict]))
            target_graphs.append(utils_np.data_dicts_to_graphs_tuple([target_graph_dict]))
  
  
    return initial_graphs,target_graphs

def create_loss_ops(target, output_ops, J_mode = False, loss_function = 'softmax'):
    
    loss_ops = [(tf.reduce_sum(output_op.edges) - tf.reduce_sum(target.edges))**2 for output_op in output_ops]

    return loss_ops

def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]

def create_feed_dict(input_placeholder,input_graphs,target_placeholder,target_graphs):
    feed_dict = {input_placeholder: input_graphs, target_placeholder: target_graphs}
    return feed_dict

