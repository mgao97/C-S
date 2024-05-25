import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
from collections import defaultdict
import glob
import math
from copy import deepcopy
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from torch_geometric.data import Data

from logger import Logger
import random
from outcome_correlation import *
import pandas as pd
from datetime import datetime as dt
import easygraph as eg

from collections import deque

import numpy as np

# 固定随机种子
np.random.seed(0)

path = '/home/ad/mgao/SocialBot/TwiBot-22-master/datasets/Twibot-22/'


def build_graph_features(g,node_id):
    deg = g.degree()
    # print('deg:',deg)
    deg = [deg[node] for node in node_id]

    outdeg = g.out_degree()
    outdeg = [outdeg[node] for node in node_id]

    indeg = g.in_degree()
    indeg = [indeg[node] for node in node_id]

    betw = eg.betweenness_centrality(g,n_workers=10) # 返回list的顺序就是按照node id的顺序，无需修改
    # print('betw:',betw)
    # betw = dict(zip(node_id,betw))
    # betw = [betw[node] for node in node_id]

    close = eg.closeness_centrality(g,n_workers=10)
    # close = [close[node] for node in node_id]

    effect = eg.effective_size(g,n_workers=10)
    effect = {k: 0 if isinstance(v, float) and math.isnan(v) else v for k, v in effect.items()}
    effect = [effect[node] for node in node_id]

    const = eg.constraint(g,n_workers=10)
    const = {k: 100 if isinstance(v, float) and math.isnan(v) else v for k, v in const.items()}
    const = [const[node] for node in node_id]

    cluster = eg.clustering(g,n_workers=10)
    cluster = [cluster[node] for node in node_id]

        
    deg_tensor = torch.tensor(deg,dtype=torch.float32)
    indeg_tensor = torch.tensor(indeg,dtype=torch.float32)
    outdeg_tensor = torch.tensor(outdeg,dtype=torch.float32)
    betw_tensor = torch.tensor(betw,dtype=torch.float32)
    close_tensor = torch.tensor(close,dtype=torch.float32)
    cluster_tensor = torch.tensor(cluster,dtype=torch.float32)
    const_tensor = torch.tensor(const,dtype=torch.float32)
    effect_tensor = torch.tensor(effect,dtype=torch.float32)
    

    graph_features_list = [deg_tensor,indeg_tensor,outdeg_tensor, betw_tensor, close_tensor, const_tensor, effect_tensor,cluster_tensor]

    expanded_graph_features = [tensor.unsqueeze(0) for tensor in graph_features_list]
    for i, tensor in enumerate(expanded_graph_features):
        print(f"Tensor {i} shape: {tensor.shape}")


    graph_metrics_tensor=torch.cat(expanded_graph_features,dim=0)
    graph_metrics_tensor = graph_metrics_tensor.transpose(0,1)
    
    print("graph_metrics_tensor size:", graph_metrics_tensor.size())
    torch.save(graph_metrics_tensor,'./dataset/twibot22/graph_metrics_tensor_0523.pt')

    return graph_metrics_tensor

    
def build_num_cat_features(path,node_id):
    user=pd.read_json(path+'user.json')
    print('node_id:',node_id[:5])
    # node_id = [int(i[1:]) for i in node_id]
    user = user[user['id'].isin(node_id)]
    print('len of user:',len(user))

    node_id_df = pd.DataFrame(node_id,columns=['node_id'])

    user = node_id_df.merge(user,left_on='node_id',right_on='id')


    print('extracting num_properties')
    following_count=[]
    for index,row in enumerate(user['public_metrics']):
        if row is not None:
            if row['following_count'] is not None:
                following_count.append(row['following_count'])
            else:
                following_count.append(0)
        else:
            following_count.append(0)
            
    statues=[]
    for index,row in enumerate(user['public_metrics']):
        if row is not None:
            if row['tweet_count'] is not None:
                statues.append(row['tweet_count'])
            else:
                statues.append(0)
        else:
            statues.append(0)

    followers_count=[]
    for each in user['public_metrics']:
        if each is not None and each['followers_count'] is not None:
            followers_count.append(int(each['followers_count']))
        else:
            followers_count.append(0)

    list_count=[]
    for each in user['public_metrics']:
        if each is not None and each['listed_count'] is not None:
            list_count.append(int(each['listed_count']))
        else:
            list_count.append(0)

    num_username=[]
    for each in user['username']:
        if each is not None:
            num_username.append(len(each))
        else:
            num_username.append(int(0))
            
    created_at=user['created_at']
    created_at=pd.to_datetime(created_at,unit='s')

    

    date0=dt.strptime('Tue Sep 5 00:00:00 +0000 2020 ','%a %b %d %X %z %Y ')
    active_days=[]
    for each in created_at:
        active_days.append((date0-each).days)
        
    active_days=pd.DataFrame(active_days)
    active_days=active_days.fillna(int(1)).astype(np.float32)

    screen_name_length=[]
    for each in user['name']:
        if each is not None:
            screen_name_length.append(len(each))
        else:
            screen_name_length.append(int(0))

    
    followers_count=pd.DataFrame(followers_count)
    followers_count=(followers_count-followers_count.mean())/followers_count.std()
    followers_count=torch.tensor(np.array(followers_count),dtype=torch.float32)

    active_days=pd.DataFrame(active_days)
    active_days.fillna(int(0))
    active_days=active_days.fillna(int(0)).astype(np.float32)

    active_days=(active_days-active_days.mean())/active_days.std()
    active_days=torch.tensor(np.array(active_days),dtype=torch.float32)

    screen_name_length=pd.DataFrame(screen_name_length)
    screen_name_length=(screen_name_length-screen_name_length.mean())/screen_name_length.std()
    screen_name_length=torch.tensor(np.array(screen_name_length),dtype=torch.float32)

    following_count=pd.DataFrame(following_count)
    following_count=(following_count-following_count.mean())/following_count.std()
    following_count=torch.tensor(np.array(following_count),dtype=torch.float32)

    list_count=pd.DataFrame(list_count)
    list_count=(list_count-list_count.mean())/list_count.std()
    list_count=torch.tensor(np.array(list_count),dtype=torch.float32)

    statues=pd.DataFrame(statues)
    statues=(statues-statues.mean())/statues.std()
    statues=torch.tensor(np.array(statues),dtype=torch.float32)

    num_properties_tensor=torch.cat([followers_count,list_count,active_days,screen_name_length,following_count,statues],dim=1)

    

    pd.DataFrame(num_properties_tensor.detach().numpy()).isna().value_counts()
    print('extracting cat_properties')
    protected=user['protected']
    verified=user['verified']

    protected_list=[]
    for each in protected:
        if each == True:
            protected_list.append(1)
        else:
            protected_list.append(0)
            
    verified_list=[]
    for each in verified:
        if each == True:
            verified_list.append(1)
        else:
            verified_list.append(0)
            
    default_profile_image=[]
    for each in user['profile_image_url']:
        if each is not None:
            if each=='https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png':
                default_profile_image.append(int(1))
            elif each=='':
                default_profile_image.append(int(1))
            else:
                default_profile_image.append(int(0))
        else:
            default_profile_image.append(int(1))

    protected_tensor=torch.tensor(protected_list,dtype=torch.float)
    verified_tensor=torch.tensor(verified_list,dtype=torch.float)
    default_profile_image_tensor=torch.tensor(default_profile_image,dtype=torch.float)
    print('verified_tensor shape:',verified_tensor.shape)
    print('protected_tensor shape:',protected_tensor.shape)
    print('default_profile_image_tensor shape:',default_profile_image_tensor.shape)

    cat_properties_tensor=torch.cat([protected_tensor.reshape([len(node_id),1]),verified_tensor.reshape([len(node_id),1]),default_profile_image_tensor.reshape([len(node_id),1])],dim=1)

    torch.save(num_properties_tensor,'./dataset/twibot22/num_properties_tensor_0522.pt')

    torch.save(cat_properties_tensor,'./dataset/twibot22/cat_properties_tensor_0522.pt')


    return cat_properties_tensor, num_properties_tensor



def bfs_subgraph(edge_list, start_node, k):
    graph = {}
    for edge in edge_list:
        src, dest, _ = edge
        if src not in graph:
            graph[src] = []
        graph[src].append(dest)
    
    visited = set()
    subgraph_nodes = set()
    subgraph_edges = []
    queue = deque([start_node])
    while queue and len(subgraph_nodes) < k:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            subgraph_nodes.add(node)
            if node in graph:
                for neighbor in graph[node]:
                    subgraph_edges.append((node, neighbor))
                    if neighbor not in visited:
                        queue.append(neighbor)
    return subgraph_edges

def build_twibot22():

    # edges = pd.read_csv(path+'edge.csv')
    labels = pd.read_csv(path+'label.csv')

    # following_edges = edges[edges['relation']=='following']
    # following_edges = following_edges.drop_duplicates()

    # following_edges.to_csv('./dataset/twibot22/following_edge.csv')
    # print('save done!!!!')

    following_edges = pd.read_csv('./dataset/twibot22/following_edge.csv')
    following_edges = following_edges[['source_id','target_id']]
    print('following_edges shape:',following_edges.shape)

    following_nodes = list(set(list(following_edges['source_id'])).union(set(list(following_edges['target_id']))))

    print('folloing edges:',following_edges.shape, 'following nodes:',len(following_nodes))
    print(following_nodes[:5])

    

    node_index_map = {uid:index for index,uid in enumerate(following_nodes)}
    index_node_map = {index:uid for index,uid in enumerate(following_nodes)}

    following_edges['sourceid'] = following_edges['source_id'].map(node_index_map)
    following_edges['targetid'] = following_edges['target_id'].map(node_index_map)

    # following_nodeidx = list(set(list(following_edges['sourceid'])).union(set(list(following_edges['targetid']))))
    # print('following_nodeidx:',following_nodeidx[:5])

    

    fow_s, fow_d = list(following_edges['sourceid']),list(following_edges['targetid'])
    fow_s = fow_s[1:]
    fow_d = fow_d[1:]
    # print('fow_s:',fow_s[:10])
    following_edges_list = []
    for i in range(len(following_edges)-1):
        following_edges_list.append((fow_s[i],fow_d[i]))
    
    print(following_edges_list[:5])
    
    
    # following_edges_list = following_edges_list.remove(('source_id','target_id'))
    g = eg.DiGraph()

    g.add_edges(following_edges_list)
    print('sample graph info:')
    print('nodes:',len(g.nodes),'edges:',len(g.edges))

    # # 提取子图

    deg_list = g.out_degree() # 从out-degree最大的节点出发，BFS搜索提取子图
    
    edge_list = list(g.edges)
    # print('edge_list:',edge_list)
    start_node = max(deg_list, key=deg_list.get)
    k = 5000
    sample_edges = bfs_subgraph(edge_list, start_node, k)

    sample_g = eg.DiGraph()
    sample_g.add_edges(sample_edges)
    print('sample graph info:')
    print('nodes:',len(sample_g.nodes),'edges:',len(sample_g.edges))


    following_nodeidx = list(sample_g.nodes)
    print('following_nodeidx:',following_nodeidx[:5])

    # graph_feats = build_graph_features(sample_g, following_nodeidx)

    ordered_following_nodes = [index_node_map[idx] for idx in following_nodeidx]

    graph_feats = torch.load('./dataset/twibot22/graph_metrics_tensor_0523.pt')
    cat_feats, num_feats = build_num_cat_features(path, ordered_following_nodes)

    x = torch.cat([graph_feats,cat_feats,num_feats],dim=1)


    edge_index = torch.tensor([list(following_edges['sourceid']),list(following_edges['targetid'])])
    num_nodes = len(ordered_following_nodes)

    labels = labels[labels['id'].isin(ordered_following_nodes)]
    node_label_map = dict(zip(list(labels['id']),list(labels['label'])))
    # print(node_label_map)

    following_labels = [node_label_map[uid] for uid in ordered_following_nodes]

    y = [1 if i=='human' else 0 for i in following_labels]
    y = torch.tensor(y).long()
    y = y.reshape(y.shape[0],1)

    data = Data(num_nodes=num_nodes,x=x,edge_index=edge_index,y=y)

    return data





def main():
    parser = argparse.ArgumentParser(description='Outcome Correlations)')
    # parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str)
    args = parser.parse_args()
    
    

    data = build_twibot22()
    # 保存Data对象
    torch.save(data, './dataset/twibot22/data.pt')
    print('data:', data)
    
    adj, D_isqrt = process_adj(data)
    normalized_adjs = gen_normalized_adjs(adj, D_isqrt)
    DAD, DA, AD = normalized_adjs
    evaluator = Evaluator(name='twibot22')
    print('evaluator:',evaluator)

    # 划分比例
    train_ratio = 0.7
    valid_ratio = 0.2
    test_ratio = 0.1

    # 计算划分点
    train_index = int(data.num_nodes * train_ratio)
    valid_index = train_index + int(data.num_nodes * valid_ratio)

    # 生成索引数组
    indices = np.arange(data.num_nodes)

    # 打乱索引数组
    np.random.shuffle(indices)

    # 划分数据集
    train_indices = indices[:train_index]
    valid_indices = indices[train_index:valid_index]
    test_indices = indices[valid_index:]

    # 检查划分是否正确
    assert len(train_indices) + len(valid_indices) + len(test_indices) == data.num_nodes

    
    split_idx = {'train':torch.tensor(train_indices).long(),'valid':torch.tensor(valid_indices).long(),'test':torch.tensor(test_indices).long()}
  
    def eval_test(result, idx=split_idx['test']):
        return evaluator.eval({'y_true': data.y[idx],'y_pred': result[idx].argmax(dim=-1, keepdim=True),})['acc']
    
    if args.dataset == 'arxiv':
        lp_dict = {
            'idxs': ['train'],
            'alpha': 0.9,
            'num_propagations': 50,
            'A': AD,
        }
        plain_dict = {
            'train_only': True,
            'alpha1': 0.87,
            'A1': AD,
            'num_propagations1': 50,
            'alpha2': 0.81,
            'A2': DAD,
            'num_propagations2': 50,
            'display': False,
        }
        plain_fn = double_correlation_autoscale
        
        """
        If you tune hyperparameters on test set
        {'alpha1': 0.9988673963255859, 'alpha2': 0.7942279952481052, 'A1': 'DA', 'A2': 'AD'} 
        gets you to 72.64
        """
        linear_dict = {
            'train_only': True,
            'alpha1': 0.98, 
            'alpha2': 0.65, 
            'A1': AD, 
            'A2': DAD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': False,
        }
        linear_fn = double_correlation_autoscale
        
        """
        If you tune hyperparameters on test set
        {'alpha1': 0.9956668128133523, 'alpha2': 0.8542393515434346, 'A1': 'DA', 'A2': 'AD'}
        gets you to 73.35
        """
        mlp_dict = {
            'train_only': True,
            'alpha1': 0.9791632871592579, 
            'alpha2': 0.7564990804200602, 
            'A1': DA, 
            'A2': AD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': False,
        }
        mlp_fn = double_correlation_autoscale  
        
        gat_dict = {
            'labels': ['train'],
            'alpha': 0.8, 
            'A': DAD,
            'num_propagations': 50,
            'display': False,
        }
        gat_fn = only_outcome_correlation

        
#     elif args.dataset == 'products':
#         lp_dict = {
#             'idxs': ['train'],
#             'alpha': 0.5,
#             'num_propagations': 50,
#             'A': DAD,
#         }
        
#         plain_dict = {
#             'train_only': True,
#             'alpha1': 1.0,
#             'alpha2': 0.9, 
#             'scale': 20.0, 
#             'A1': DAD, 
#             'A2': DAD,
#             'num_propagations1': 50,
#             'num_propagations2': 50,
#         }
#         plain_fn = double_correlation_fixed
        
#         linear_dict = {
#             'train_only': True,
#             'alpha1': 1.0,
#             'alpha2': 0.9, 
#             'scale': 20.0, 
#             'A1': DAD, 
#             'A2': DAD,
#             'num_propagations1': 50,
#             'num_propagations2': 50,
#         }
#         linear_fn = double_correlation_fixed
        
#         mlp_dict = {
#             'train_only': True,
#             'alpha1': 1.0,
#             'alpha2': 0.8, 
#             'scale': 10.0, 
#             'A1': DAD, 
#             'A2': DA,
#             'num_propagations1': 50,
#             'num_propagations2': 50,
#         }
#         mlp_fn = double_correlation_fixed




    model_outs = glob.glob(f'models/twibot22_{args.method}/*.pt')
    
    if args.method == 'lp':
        out = label_propagation(data, split_idx, **lp_dict)
        print('Valid acc: ', eval_test(out, split_idx['valid']))
        print('Test acc:', eval_test(out, split_idx['test']))
        return
    
    get_orig_acc(data, eval_test, model_outs, split_idx)
    while True:
        if args.method == 'plain':
            evaluate_params(data, eval_test, model_outs, split_idx, plain_dict, fn = plain_fn)
        elif args.method == 'linear':
            evaluate_params(data, eval_test, model_outs, split_idx, linear_dict, fn = linear_fn)
        elif args.method == 'mlp':
            evaluate_params(data, eval_test, model_outs, split_idx, mlp_dict, fn = mlp_fn)
        elif args.method == 'gat':
            evaluate_params(data, eval_test, model_outs, split_idx, gat_dict, fn = gat_fn) 
#         import pdb; pdb.set_trace()
        break
        
# #     name = f'{args.experiment}_{args.search_type}_{args.model_dir}'
# #     setup_experiments(data, eval_test, model_outs, split_idx, normalized_adjs, args.experiment, args.search_type, name, num_iters=300)
    
# #     return

    
if __name__ == "__main__":
    main()
