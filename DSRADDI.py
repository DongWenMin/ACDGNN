import tensorflow as tf
import numpy as np
from tensorflow.python.ops.array_ops import concat
from tensorflow.python.ops.variables import trainable_variables
import myutils.data_utils as ut
from myutils.initialization_utils import initialize_experiment
import myutils.parser as pars
from myutils.dsraddi_func import _convert_sp_mat_to_sp_tensor,sp_hete_attn_head,sp_hete_attn_head1,get_output
from myutils.dsraddi_func import get_att_aggre_embedding
from functools import reduce
from myutils.eval import evaluate,evaluate1,evaluate_data
import time
import os
from scipy.sparse import csc_matrix

import pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def full_connection(seq, out_sz, activation=-1, in_drop=0.0, use_bias=True):
    with tf.name_scope('full_connection_layer'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, keep_prob=1.0 - in_drop)
        
        seq_fc = tf.layers.conv1d(seq, out_sz, 1, use_bias=use_bias)
        seq_fc = tf.squeeze(seq_fc) # remove the bach_size which is set as 1
        if activation!=-1:
            ret = activation(seq_fc)
        else:
            ret = seq_fc
        return ret

# path = "/DWM/My Drive/GBNNDDI/DSRADDI"
# os.chdir(path)
base_dir = os.getcwd()
#数据读取
params = pars.parse_args()
print(params)
params.hid = [int(params.emb_dim/params.layers) for i in range(params.layers)]
print('params.hid:%s'%params.hid)
initialize_experiment(params, base_dir)

#输出到file文件
params.heads = params.heads
f = open("./result/myresult_%d_%d_%d_%d.txt"%(params.heads,params.emb_dim,params.dsf,\
    params.cdt),'a+',encoding='utf-8')  
params.dsf = bool(params.dsf)
params.cdt = bool(params.cdt)
heads = params.heads
print(params.split, file=f)
f.flush()
print('data/{}/{}/{}_triple.txt'.format(params.dataset, params.split, params.train_file))
#设置参数
params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}/{}_triple.txt'.format(params.dataset, params.split, params.train_file)),
        'valid': os.path.join(params.main_dir, 'data/{}/{}/{}_triple.txt'.format(params.dataset, params.split, params.valid_file)),
        'test': os.path.join(params.main_dir, 'data/{}/{}/{}_triple.txt'.format(params.dataset, params.split, params.test_file))
    }
# print("----------------",params.file_paths)
# print("----------------",params.split)
# print("----------------",params.train_file)
# print('data/{}/{}/{}_triple.txt'.format(params.dataset, params.split, params.train_file))
# triple_file = 'data/{}/relations_2hop.txt'.format(params.dataset)
triple_file = 'data/{}/relations_2hop_data_new.txt'.format(params.dataset)
'''
print("total epoch:%d"%(params.epoch)) 
数据描述：33765个节点，11种类型的节点，23种类型的边，1690693条边
adj_list：前86个稀疏矩阵是train里面的，后23个是KG图里面的（注意，这里的adj是从训练数据中构建的）
药物相关的网络：(参考 https://github.com/hetio/hetionet/blob/master/describe/edges/metaedges.tsv)
87:Compound-resembles-drug/drug-resembles-drug(这部分去掉)
91:drug-treat-disease
92:drug-binds-gene
93:drug-upregulates-gene
96：drug-palliates-disease
104:drug-downregulates-gene
107:drug-causes-Side Effect
101:Pharmacologic-contains-drug(药理学类别)

triplets:字典，键：'train', 'valid', 'test'，值：三元组
entity2id:实体到id的映射，这里key=value，可以不用受理
relation2id:关系到id的映射，同样key=value，关系数量86+23
id2entity,id2relation：entity2id，relation2id的倒置
rel：DDI类型的数量，rel=86
'''
print("emb dim:%d, rela dim:%d hid dim:%s"%(params.emb_dim,params.rel_dim,params.hid))
adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel = ut.process_files_ddi(params.file_paths, triple_file, None)
print("loading finish.", file=f) 
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),file=f)
f.flush()
# adj_list[87] = adj_list[87]+adj_list[87].T#关系87转对称，方便后面传递信息
# adj_list[101] = adj_list[101]+adj_list[101].T#关系101转对称，方便后面传递信息
n_components = params.hid[-1]
#这里计算相互作用相似性，从训练集中读取数据，用于后面的损失部分
ddi_adj = reduce(np.add,adj_list[:86])
adj_indices, adj_data, adj_shape = ut.preprocess_adj_hete(ddi_adj)
ddi_adj = csc_matrix((adj_data, (adj_indices[:,0],adj_indices[:,1])),shape=(1710,1710))
ddi_adj = ddi_adj.todense()
ddi_adj = ddi_adj + np.transpose(ddi_adj)
ddi_adj[np.where(ddi_adj>0)]=1
for i in range(len(ddi_adj)):#对角线置为1，防止计算出错
  ddi_adj[i,i] = 1
ddi_sim = ut.get_target_sim_matrix(ddi_adj)
ddi_sim = ut.get_pca_feature(ddi_sim,n_components)
ddi_sim_in = tf.constant(ddi_sim,dtype=tf.float32)#ddi相似性常量


sparse_ddi_adj = adj_list[:86]

#这里要获取每一种类型的mask,后面特征相加的时候会用到
ddi_types = 86
ent_types = 10#10种实体类型
ent_type = np.loadtxt('./data/drugbank/entity.txt')
ent_mask = np.zeros((ent_types,ent_type.shape[0]),dtype=np.float32)
for i in range(ent_types):
    mask_i = np.where(ent_type==i)
    ent_mask[i][mask_i] = 1.0
ent_mask = np.expand_dims(ent_mask,axis=-1)

sparse_adj_input = [ut.preprocess_adj_hete(a) for a in adj_list]#有向

#这里统计一下所有节点的邻居,发现有些节点没有任何邻居
# neghbors = defaultdict(list)
# for item in sparse_adj_input:
#     edges = item[0]
#     for e in edges:
#         neghbors[e[0]].append(e[1])
#         neghbors[e[1]].append(e[0])

sparse_adj_input = sparse_adj_input[ddi_types:]
#记录一下23种矩阵对应的节点类型,sparse_adj_input的行对应adj_type的第二个元素，列对应第一个元素
adj_type = [(1,1),(0,0),(2,1),(2,3),(2,4),(0,2),(0,1),(0,1),(2,2),(2,1),
        (0,2),(4,1),(4,1),(1,1),(1,5),(6,0),(1,7),(1,1),(0,1),(2,1),
        (1,8),(0,9),(4,1)]
#这里要加上转置部分
sparse_adj_input_t = []
for item in sparse_adj_input:
    edge_list,value,dense_shape = item 
    sparse_adj_input_t.append((edge_list[:,-1::-1],value,dense_shape))

adj_type_t = np.array(adj_type)[:,-1::-1].tolist()

layers = params.layers

#导入每个节点的指纹特征，用pca降维之后，维度变为64

feat = pickle.load(open("./data/drugbank/DB_molecular_feats.pkl", 'rb'), encoding='utf-8')
feat = np.array(feat["Morgan_Features"])
feat = np.array(feat.tolist())
feat[feat>0]=1
drug_fsim = ut.get_target_sim_matrix(feat)

drug_f = ut.get_pca_feature(drug_fsim,n_components*layers)
#药物原本的指纹相似性特征
drug_f_sim = tf.constant(drug_f,dtype=tf.float32)
#46个稀疏矩阵，46中邻接矩阵的类型
sparse_adj_input = sparse_adj_input + sparse_adj_input_t
adj_type = adj_type+adj_type_t

#这里将sparse_adj_input转换为稀疏矩阵作为模型的输入spa_adj
spa_adj = []
for item in sparse_adj_input:
    # spa_temp = csc_matrix((item[1], (item[0][:,0], item[0][:,1])), shape=item[2])
    spa_temp = tf.SparseTensor(indices=item[0], values=item[1], dense_shape=item[2])
    spa_temp = tf.sparse_reorder(spa_temp)
    spa_adj.append(spa_temp)

#读取实体特征，同transE预训练的特征
ent_feature = np.loadtxt('./data/drugbank/ent_embeddings_DDI.txt')
#n_components较小时，先进行PCA降维<to do>
if n_components*layers<ent_feature.shape[1]:
  ent_feature = ut.get_pca_feature(ent_feature,n_components*layers)
#为了方便下面的操作，这里对ent_feature线性转换
ent_feature = np.pad(ent_feature,((0,0),(0,n_components*layers-ent_feature.shape[1])),'constant',constant_values=(0,0))
#把药物的分子指纹相似性作为输入(不用已提供的)，对于其他实体，使用以提供的嵌入表示
ent_feature = np.vstack([drug_f,ent_feature[feat.shape[0]:]]) 
ent_f = np.array(np.expand_dims(ent_feature,0),dtype=np.float32)
hid = params.hid#[64,64]
batch_size = 1#矩阵计算的batch size
# out_sz = 8#域转化后的输出
hid_units = params.hid#每一层的out_sz
nb_nodes = adj_list[0].shape[0]#节点的数量
activation = tf.nn.elu
# 0:drug，1:gene，2,disease，3,Symptom，4,Anatomy，5,Molecular Function
# 6,Pharmacologic Class，7,Cellular Component，8,Pathway，9,Side Effect

attn_drop = tf.placeholder(dtype=tf.float32, shape=())#注意力的dropout
ffd_drop = tf.placeholder(dtype=tf.float32, shape=())#特征的dropout
# ftr_in = tf.placeholder(dtype=tf.float32,shape=(batch_size, 
#               ent_feature.shape[0], ent_feature.shape[1]))

ftr_in = tf.constant(ent_f,dtype=tf.float32)#特征常量

#type mask,shape=(ent_types,nb_nodes,1)
# type_mask = tf.placeholder(dtype=tf.float32,shape=(ent_types, 
#               nb_nodes,1))
type_mask =   tf.constant(ent_mask,dtype=tf.float32)         

#attns：多头注意力机制
attns = []#长度为8
#邻接矩阵的placeholder
# spa_adj = [tf.sparse_placeholder(dtype=tf.float32) 
#               for _ in range(len(sparse_adj_input))]
# spa_adj = [tf.SparseTensor(indices=sai[0], values=sai[1],dense_shape=sai[2])
#             for sai in sparse_adj_input]

#取每一种类型实体对应的下标
ent_mask = np.squeeze(ent_mask)
seq = tf.squeeze(ftr_in)
type_indexs = []
for i in range(ent_types):
    type_indexs.append(np.where(np.array(ent_mask[i])==1)[0])

#获取每一种类型实体下标到原本下标的映射，比如type_indexs[1]中[1710, 1711, 1712, 1713],转化为{1710:0,1711:1,1712:2,1713:3}
ent_index_mapping = []
for i in range(ent_types):
    temp_dict = {}
    for j in range(len(type_indexs[i])):
        temp_dict[type_indexs[i][j]] = j
    ent_index_mapping.append(temp_dict)

initializer = tf.contrib.layers.xavier_initializer()
# temp = tf.Variable(initializer([nb_nodes, 32]), name='random_embedding')
temp = tf.squeeze(ftr_in)

for _ in range(heads):
    ret = sp_hete_attn_head(seq=temp,spa_adj=spa_adj, adj_type=adj_type,
        out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,in_drop=ffd_drop, coef_drop=attn_drop, 
        type_indexs=type_indexs,ent_index_mapping=ent_index_mapping,sparse_adj_input=sparse_adj_input,
        ent_types=ent_types,cdt=params.cdt)
    # ret = sp_hete_attn_head1(seq=ftr_in,spa_adj=spa_adj, adj_type=adj_type,
    #     out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,in_drop=ffd_drop, coef_drop=attn_drop, 
    #     type_indexs=type_indexs,ent_index_mapping=ent_index_mapping,sparse_adj_input=sparse_adj_input,
    #     ent_types=ent_types)
    # ret = sp_hete_attn_head1(seq=temp,spa_adj=spa_adj, adj_type=adj_type,
    #     out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,in_drop=ffd_drop, coef_drop=attn_drop, 
    #     type_indexs=type_indexs,ent_index_mapping=ent_index_mapping,sparse_adj_input=sparse_adj_input,
    #     ent_types=ent_types)
    attns.append(ret)


h_11 = [tf.concat(attn, axis=-1) for attn in zip(*attns)]
#h_1按照type_mask合并
h_pre = [h_11[i]*type_mask[i] for i in range(len(h_11))]
h_pre1 = reduce(tf.add, h_pre)
#这里的h_pre1不包括是通过邻居聚合而来，不包括节点自身，因此需要与节点自身的特征进行融合。
#采用Generalized Relation Learning with Semantic Correlation Awareness for Link Prediction中的融合策略
#发现ftr_in与h_pre1维度不一样，无法实现
# layer0_weight = full_connection(ftr_in, 1, activation=-1, in_drop=ffd_drop, use_bias=True)
# layer1_weight = full_connection(h_pre1, 1, activation=-1, in_drop=ffd_drop, use_bias=True)
# concat_weight = tf.concat([tf.expand_dims(layer0_weight,axis=0),tf.expand_dims(layer1_weight,axis=0)],axis=0)
# concat_weight = tf.nn.softmax(concat_weight,dim=0)
# layer_fea = tf.concat([])


#采用GraphSage Aggregator
# temp = tf.concat([drug_f_sim,drug_f_sim], axis=-1)#数据正常

# temp = tf.concat([ftr_in,ftr_in], axis=-1)#数据不正常,说明是transE初始的特征的问题

#这里先随机初始化
# temp = tf.concat([temp,temp], axis=-1)
# temp = [temp,temp]
# temp = reduce(tf.add, temp)

# layer_fea = h_pre1 #数据不正常
layer_fea = tf.concat([tf.expand_dims(temp,axis=0),h_pre1],axis=2)
layer_fea = full_connection(layer_fea, params.emb_dim, activation=tf.nn.leaky_relu, in_drop=ffd_drop, use_bias=True)
layer_fea = tf.expand_dims(layer_fea,axis=0)

for i in range(1, layers):
    h_old = layer_fea
    attns = []
    head_act = activation
    is_residual = False
    for _ in range(heads):
        ret1 =sp_hete_attn_head(seq=h_old,  spa_adj=spa_adj, adj_type=adj_type,out_sz=hid_units[i], 
                        activation=head_act, nb_nodes=nb_nodes,in_drop=ffd_drop, coef_drop=attn_drop, type_indexs=type_indexs,
                        ent_index_mapping=ent_index_mapping,sparse_adj_input=sparse_adj_input,ent_types=ent_types,
                        cdt=params.cdt)
        attns.append(ret1)
    h_1 = [tf.concat(attn, axis=-1) for attn in zip(*attns)]
    #保存最后一层的结果
    # if i<len(heads)-1:
    h_mid = [h_1[i]*type_mask[i] for i in range(len(h_1))]
    h_pre1 = reduce(tf.add, h_mid)
    layer_fea = tf.concat([h_old,h_pre1],axis=2)
    # layer_fea = tf.expand_dims(layer_fea,axis=0)
    layer_fea = full_connection(layer_fea, params.emb_dim, activation=tf.nn.leaky_relu, in_drop=ffd_drop, use_bias=True)
    layer_fea = tf.expand_dims(layer_fea,axis=0)

h_1 = tf.squeeze(layer_fea)#降维


data_batch_size = params.batch_size
lr = params.lr
#这里咱先不进行后续处理，得到h_1后，直接从他们中取得嵌入，用mlp来预测相互作用
#后续的特征应该也用个mlp来预测，降低嵌入和特征之间的干扰，最后将结果进行attention
#placeholder
pos_drug1 = tf.placeholder(tf.int32, shape=[None], name='pos_drug1')
pos_drug2 = tf.placeholder(tf.int32, shape=[None], name='pos_drug2')
ddi_type = tf.placeholder(tf.int32, shape=[None], name='ddi_type')#类型即可
neg_drug1 = tf.placeholder(tf.int32, shape=[None], name='neg_drug1')
neg_drug2 = tf.placeholder(tf.int32, shape=[None], name='neg_drug2')

#单类型标签
binary_label = tf.placeholder(tf.float32, shape=[None], name='binary_label')
# ddi_type = tf.concat((ddi_type,ddi_type),axis=0)#+ -的ddi_type一样，这里串联一下


# drug1 = tf.concat((pos_drug1,neg_drug1),axis=0)
# drug2 = tf.concat((pos_drug2,neg_drug2),axis=0)



#获取特征的嵌入
# drug1_emb = tf.nn.embedding_lookup(h_1, drug1)
# drug1_feat = tf.nn.embedding_lookup(drug_f_sim, drug1)
# drug2_emb = tf.nn.embedding_lookup(h_1, drug2)
# drug2_feat = tf.nn.embedding_lookup(drug_f_sim, drug2)

#生成药物的DDI结构特征
#1,共享的attention 参数

aggre_shared_att = tf.Variable(initializer([1, layers*hid[-1]]), name='aggre_shared_att')
stru_embed_trans_matrix = tf.Variable(initializer([layers*hid[-1], layers*hid[-1]]), name='stru_embed_trans_matrix')
# feat_embed_trans_matrix = tf.Variable(initializer([n_components, layers*hid[-1]]), name='feat_embed_trans_matrix')
feat_embed_trans_matrix = tf.Variable(initializer([n_components*layers, layers*hid[-1]]), name='feat_embed_trans_matrix')
ddi_sim_trans_matrix = tf.Variable(initializer([n_components, layers*hid[-1]]), name='ddi_sim_trans_matrix')

#2,聚合，策略：Relation-aware Graph Attention Model With Adaptive Self-adversarial Training
# drug1_emb_trans = drug1_emb @ stru_embed_trans_matrix
# drug1_feat_trans = drug1_feat @ feat_embed_trans_matrix
# drug2_emb_trans = drug2_emb @ stru_embed_trans_matrix
# drug2_feat_trans = drug2_feat @ feat_embed_trans_matrix

# aggre_embed1 = get_att_aggre_embedding(drug1_emb_trans,drug1_feat_trans,aggre_shared_att)
# aggre_embed2 = get_att_aggre_embedding(drug2_emb_trans,drug2_feat_trans,aggre_shared_att)

#采用串联的方式
# if params.dsf:
#     con_emb = tf.concat([drug1_emb,drug1_feat,drug2_emb,drug2_feat,aggre_embed1,aggre_embed2], axis=1)
#     con_emb_drug1 = tf.concat([drug1_emb,drug1_feat,aggre_embed1], axis=1)
#     con_emb_drug2 = tf.concat([drug2_emb,drug2_feat,aggre_embed2], axis=1)
# else:
#     con_emb = tf.concat([drug1_emb,drug1_feat,drug2_emb,drug2_feat], axis=1)
#     con_emb_drug1 = tf.concat([drug1_emb,drug1_feat], axis=1)
#     con_emb_drug2 = tf.concat([drug2_emb,drug2_feat], axis=1)

#关系向量
rel_dim = 3*params.emb_dim if params.dsf else  2*params.emb_dim
relation_vector = tf.Variable(initializer([ddi_types, rel_dim]), name='relation_vector')
batch_relation_matrix = tf.matrix_diag(tf.nn.embedding_lookup(relation_vector, ddi_type))
ddi_shared_matrix = tf.Variable(initializer([rel_dim, rel_dim]), name='ddi_shared_matrix')
# con_emb = tf.concat([drug1_feat,drug2_feat], axis=1)

# pos_output,pos_aggre_embed1,pos_aggre_embed2 = \
#     get_output(pos_drug1,pos_drug2,drug_f_sim,h_1,stru_embed_trans_matrix,feat_embed_trans_matrix,aggre_shared_att,params,batch_relation_matrix,ddi_shared_matrix)

# neg_output,neg_aggre_embed1,neg_aggre_embed2 = \
#     get_output(neg_drug1,neg_drug2,drug_f_sim,h_1,stru_embed_trans_matrix,feat_embed_trans_matrix,aggre_shared_att,params,batch_relation_matrix,ddi_shared_matrix)

# pos_output,pos_aggre_embed1,pos_aggre_embed2 = \
#     get_output(pos_drug1,pos_drug2,drug_f_sim,drug_f_sim,stru_embed_trans_matrix,feat_embed_trans_matrix,aggre_shared_att,params,batch_relation_matrix,ddi_shared_matrix)

# neg_output,neg_aggre_embed1,neg_aggre_embed2 = \
#     get_output(neg_drug1,neg_drug2,drug_f_sim,drug_f_sim,stru_embed_trans_matrix,feat_embed_trans_matrix,aggre_shared_att,params,batch_relation_matrix,ddi_shared_matrix)

pos_output,pos_aggre_embed1,pos_aggre_embed2 = \
    get_output(pos_drug1,pos_drug2,drug_f_sim,h_1,stru_embed_trans_matrix,feat_embed_trans_matrix,aggre_shared_att,params,batch_relation_matrix,ddi_shared_matrix)

neg_output,neg_aggre_embed1,neg_aggre_embed2 = \
    get_output(neg_drug1,neg_drug2,drug_f_sim,h_1,stru_embed_trans_matrix,feat_embed_trans_matrix,aggre_shared_att,params,batch_relation_matrix,ddi_shared_matrix)


pred = tf.sigmoid(pos_output) 
output = tf.concat([pos_output,neg_output],axis=0)

#预测,这里就直接用矩阵来进行预测
# initializer = tf.contrib.layers.xavier_initializer()
# weight = {}
# weight["pred_matrix"] = tf.Variable(initializer([2*hid[-1]*len(heads), 86]), name='pred_matrix')
# output = con_emb @ weight["pred_matrix"]

# con_emb = tf.expand_dims(con_emb,axis=0)
# output = full_connection(con_emb, ddi_types, activation=-1, in_drop=ffd_drop, use_bias=True)
# pred = tf.nn.softmax(output)

# output = tf.expand_dims(con_emb_drug1,axis=1) @ batch_relation_matrix 
# output = tf.squeeze(output) @ ddi_shared_matrix
# output = tf.expand_dims(output,axis=1) @ batch_relation_matrix
# output = output @ tf.expand_dims(con_emb_drug2,axis=-1)
# output = tf.squeeze(output) #1024
# pred = tf.sigmoid(output) #1024 after sigmoid

base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=binary_label, logits=output))

aggre_embed = tf.concat([pos_aggre_embed1,pos_aggre_embed2,neg_aggre_embed1,neg_aggre_embed2],axis=0)
drugs = tf.concat([pos_drug1,pos_drug2,neg_drug1,neg_drug2],axis=0)
#计算预测损失
# base_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output1, labels=ddi_type)) +\
#             tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output2, labels=ddi_type))
#计算相似性误差
ddi_sim_embed = tf.nn.embedding_lookup(ddi_sim_in, drugs)
ddi_sim_embed1_trans = ddi_sim_embed @ ddi_sim_trans_matrix
sim_mapping_loss = tf.square(ddi_sim_embed1_trans - aggre_embed)

sim_mapping_loss = tf.reduce_mean(sim_mapping_loss)
sim_mapping_weight = tf.placeholder(dtype=tf.float32, shape=(),name='sim_mapping_weight')

total_loss = base_loss + sim_mapping_weight * sim_mapping_loss
if params.dsf:
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)
else:
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(base_loss)


#运行模块
triplets['train'] = triplets['train'].astype(np.int32)
triplets['valid'] = triplets['valid'].astype(np.int32)

dev_token = 1
test_token = 1
record_token = 1
threshold = 0.5
feed_dict = {  'pos_drug1':pos_drug1,#药物1
          'pos_drug2':pos_drug2,#药物2
          'ddi_type':ddi_type,
          'ffd_drop':ffd_drop,
          'attn_drop':attn_drop,
          'pred':pred,
          'threshold':threshold}

# train_pos = np.insert(pos_edges, pos_edges.shape[1], values=1, axis=1)
# train_neg = np.insert(neg_edges, neg_edges.shape[1], values=0, axis=1)
# traint = np.vstack((train_pos,train_neg))

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess = tf.Session()
sess.run(init_op)
# saver = tf.train.Saver()
print('negative sampling...')
pos_edges, neg_edges = ut.sample_neg(sparse_ddi_adj, triplets['train'], num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0.5,drug_num=1710)
pos_edges = pos_edges.astype(np.int32)
neg_edges = neg_edges.astype(np.int32)

for epoch in range(params.epoch):   
    permutation = np.random.permutation(pos_edges.shape[0])
    #从pos_edges, neg_edges中取数据
    pos_edges = pos_edges[permutation] 
    neg_edges = neg_edges[permutation] 
    batch = len(permutation)//data_batch_size + 1
    total_base_loss = 0.
    total_sim_mapping_loss = 0.
    time1 = time.time()
    # saver.save(sess, "Model/model.ckpt")
    # loss_record = []
    for b in range(batch):
        start = b * data_batch_size
        if start == len(permutation):
            break
        end = (b+1) * data_batch_size
        #postive samples
        pos_train_batch = pos_edges[start:end]
        d1_feed_pos = pos_train_batch[:,0]
        d2_feed_pos = pos_train_batch[:,1]
        label_feed = pos_train_batch[:,2]
        #negative samples
        neg_train_batch = neg_edges[start:end]
        d1_feed_neg = neg_train_batch[:,0]
        d2_feed_neg = neg_train_batch[:,1]
        feed_dict_tra = {
            pos_drug1:d1_feed_pos,#药物1
            pos_drug2:d2_feed_pos,#药物2
            neg_drug1:d1_feed_neg,#药物1
            neg_drug2:d2_feed_neg,#药物2
            ddi_type:label_feed,
            # ftr_in:ent_f,#输入特征
            ffd_drop:params.ffd_drop,
            attn_drop:params.attn_drop,
            binary_label: np.concatenate((np.ones((len(d1_feed_pos)),dtype=np.float32),np.zeros((len(d1_feed_neg)),dtype=np.float32)),axis=0),
            sim_mapping_weight:params.align_weight
            # type_mask:ent_mask
        }
        # feed_dict_tra.update({i: d for i, d in zip(spa_adj, sparse_adj_input)})
        #优化目标
        _,batch_base_loss,batch_sim_mapping_loss = sess.run([opt,base_loss,sim_mapping_loss], feed_dict_tra)
        total_base_loss += batch_base_loss
        total_sim_mapping_loss += batch_sim_mapping_loss
        # loss_record.append(batch_base_loss)
        # print(b)
        # break
    time2 = time.time()
    print('epoch:%d/%d(time:%.4f),total_loss:%.4f + %.4f'%(epoch,params.epoch,time2-time1,total_base_loss,total_sim_mapping_loss),file=f)
    print('epoch:%d/%d(time:%.4f),total_loss:%.4f + %.4f'%(epoch,params.epoch,time2-time1,total_base_loss,total_sim_mapping_loss))
    f.flush()
    #-----------------模型训练----------------------
    # evaluate_data(triplets['train'],f,'训练',sess,feed_dict)
    #-----------------模型验证----------------------
    valid_data = np.loadtxt(params.file_paths['valid'])
    evaluate_data(valid_data,f,'验证',sess,feed_dict)
    # evaluate_data(traint,f,'训练',sess,feed_dict)
    #-----------------模型测试----------------------
    if epoch % 1 == 0:
      test_data = np.loadtxt(params.file_paths['test'])
      evaluate_data(test_data,f,'测试',sess,feed_dict)
      # #测试集 批次测试
      

f.close()
sess.close()