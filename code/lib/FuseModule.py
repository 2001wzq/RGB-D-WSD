import os  
import time
import torch
import torch.nn as nn
import torch.nn.functional as F




#第一阶段的编码器的rgb特征图与深度特征图  融合模块

class FuseModule0(nn.Module):    
    def __init__(self,in_dim, out_dim):  #输入的RGB和深度图通道数都是in_dim，   经过该融合模块后 得到的融合特征图的通道数是out_dim
        super(FuseModule0, self).__init__()
        
        self.layer_1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer_2 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        
         
        self.layer_3 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)   
        self.layer_4 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_5 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1) 

        
        self.layer_ful1 = nn.Conv2d(2*out_dim, out_dim, kernel_size=1, stride=1, padding=0)  


    def forward(self, rgb, depth):
        
        ################################
        
        x_rgb = self.layer_1(rgb)      #输入通道    得到    输出通道
        x_dep = self.layer_2(depth)    #输入通道    得到    输出通道
        
        x_dep = self.layer_3(x_dep)    #输出通道    得到    输出通道
        dep_w = F.sigmoid(x_dep)         #激活  还是   输出通道数
        
        ##
        x_rgb_w = x_rgb.mul(dep_w)     #rgb特征图与深度权重图逐元素相乘   得到  输出通道
        
        x_rgb_r = x_rgb_w + x_rgb      #逐元素相加    得到  输出通道
        
        ## fusion 
        x_dep_r = self.layer_4(x_dep)    #输出通道   得到    输出通道
        x_rgb_r = self.layer_5(x_rgb_r)  #输出通道   得到    输出通道
        
        
        ful_out = torch.cat((x_dep_r,x_rgb_r),dim=1)    #将两者按照通道数进行concat  得到2倍输出通道   
        out = self.layer_ful1(ful_out)                 # 二倍输出通道数    得到     一倍输出通道数
         
        return out




#第二，三，四，五阶段的编码器的rgb特征图与深度特征图   融合模块
class FuseModule1(nn.Module):    
    def __init__(self,in_dim, out_dim):    #输入的是 通道数为in_dim的 RGB和深度特征图，以及前阶段融合特征图，经过该融合模块后融合特征图的通道数是out_dim，
        super(FuseModule1, self).__init__()
        
        
        self.layer_1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)              
        self.layer_2 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)              
         
        self.layer_3 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)          
        self.layer_4 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)             
        self.layer_5 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)  

        self.layer_ful1 = nn.Conv2d(2*out_dim, out_dim, kernel_size=1, stride=1, padding=0)  
        self.layer_ful2 = nn.Conv2d(out_dim+out_dim//2, out_dim, kernel_size=1, stride=1, padding=0)  


    def forward(self, rgb, depth,xx):
        
        ################################
        
        x_rgb = self.layer_1(rgb)            # 输入通道数 得到  输出通道数
        x_dep = self.layer_2(depth)          # 输入通道数 得到  输出通道数
        
        x_dep = self.layer_3(x_dep)          #输出通道数  得到  输出通道数
        dep_w = F.sigmoid(x_dep)             #激活  还是   输出通道数
        
        ##
        x_rgb_w = x_rgb.mul(dep_w)           #rgb特征图与深度权重图逐元素相乘   得到  输出通道
        
        x_rgb_r = x_rgb_w + x_rgb            #逐元素相加    得到  输出通道
        
        ## fusion 
        x_dep_r = self.layer_4(x_dep)        #输出通道数  得到  输出通道数
        x_rgb_r = self.layer_5(x_rgb_r)      #输出通道数  得到  输出通道数
        
        
        ful_out = torch.cat((x_dep_r,x_rgb_r),dim=1)        # 将rgb与深度通道数 进行通道维度concat  得到二倍输出通道数
        out1 = self.layer_ful1(ful_out)                     # 2倍输出通道数   得到   1倍输出通道数

        #print(out1.size())
        #print(xx.size())

        out2 = torch.cat([out1,xx],dim=1)                   # 将融合后的特征图 与 前阶段融合特征图  按照通道维度进行concat 得到1.5倍输出通道数   因为前阶段通道数为0.5倍输出通道数

        out3 = self.layer_ful2(out2)                        #  1.5倍输出通道数 得到  1倍输出通道数
         
        return out3





class LY(nn.Module):    
    def __init__(self,in_dim):
        super(LY, self).__init__()
         
        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1))


        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1))
        
        self.relu = nn.ReLU(inplace=True)
        self.layer_cat1 = nn.Sequential(
            nn.Conv2d(in_dim*2, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim))
        self.sigmoid = nn.Sigmoid()        
        
    def forward(self, x_ful, x1, x2):
        
        ################################
        x_ful_1 = self.squeeze_rgb(self.channel_attention_rgb(x1))
        x_ful_2 = self.squeeze_depth(self.channel_attention_depth(x2))
        print(x_ful_1.shape,x_ful_2.shape)
        raise
        x_ful_w = self.sigmoid(self.layer_cat1(torch.cat([x_ful_1, x_ful_2],dim=1)))
        out     = self.relu(x_ful.mul(x_ful_w))
        
        return out





import torch_geometric
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch




class GNNFuse(nn.Module):
    def __init__(self, in_channels, num_layers=2):
        super(GNNFuse, self).__init__()
        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.squeeze_dep = nn.AdaptiveAvgPool2d(1)    #自适应全局池化，不管输入尺寸是多少，只需要指定输出尺寸。这就是自适应
        self.squeeze_ful = nn.AdaptiveAvgPool2d(1)    #三个全局平均池化

        self.tok = nn.Parameter(torch.randn(1, in_channels))   #创建一个可学习参数，形状为(1, in_channels)，初始为随机正态分布，
                                                               #因为用了nn.Parameter，这个变量会被加入到模型的参数列表中，在训练时随着反向传播一起更新
        
        graph_conv = nn.ModuleList()   #和普通列表不一样，里边存放的东西必须是nn.Module（比如卷积层，线性层）
                                       #存进去之后，PyTorch 会自动识别这些子层是 模型参数，会参与训练（出现在 model.parameters()
        
        graph_norm = nn.ModuleList()   #跟上面一样，也是一个“模块列表”，专门存 归一化层

        for i in range(num_layers):

            # GINConv 需要一个 nn.Module（MLP），不能直接传整数
            mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, in_channels)
            )
            

            graph_conv.append(         #每次循环会创建一个新的 GATConv(...) 模块，并把它加进 graph_conv 这个容器里
                GATConv(in_channels, in_channels, heads=4, concat=False, dropout=0.6))
                #ResGatedGraphConv(in_channels, in_channels))
                #GATv2Conv(in_channels, in_channels, heads=4, concat=False, dropout=0.6))
                #GINConv(mlp, eps=0., train_eps=False))
                #TransformerConv(in_channels, in_channels, heads=4, concat=False, dropout=0.6))
            graph_norm.append(nn.LayerNorm(in_channels))


        self.num_layers = num_layers
        self.graph_conv = graph_conv
        self.graph_norm = graph_norm
        self.relu = nn.ReLU(inplace=True)



    def forward(self, x_ful, rgb, dep):
        
        # Squeeze spatial dimensions
        x1 = self.squeeze_rgb(rgb).squeeze(-1).squeeze(-1)    # Shape: (batch_size, in_channels)
        x2 = self.squeeze_dep(dep).squeeze(-1).squeeze(-1)    # Shape: (batch_size, in_channels)
        fu = self.squeeze_ful(x_ful).squeeze(-1).squeeze(-1)  # Shape: (batch_size, in_channels)
        batch_size = x1.size(0)


        tok = self.tok.expand(batch_size, -1)                 # Shape: (batch_size, in_channels)    #-1表示该维度上不变化，而0维度的大小变成batch_size


        fu = torch.stack([tok, fu, x1, x2], dim=1)            # Shape: (batch_size, 4, in_channels)   理解成把上边四个节点在堆叠在一起，放在新维度上，这里是维度1
                                                                                                    #这样就得到一个含有四节点的小图




        edge_index = self._get_edge_index().to(x1.device)     # 返回一个张量，表示图的边，并且把这个张量移动到与x1所在的设备上
                                                              # 这里一共有9条边，fu,rgb,depth三者都是双向连接，然后三者共同指向tok
        # Combine features from both modalities





        batch_feats = []
        for i in range(batch_size):
            x = Data(x=fu[i], edge_index=edge_index)         #节点加边构成单张图
            batch_feats.append(x)                            #图列表，batchsize多大，列表里就会有几张图

        graph_data = Batch.from_data_list(batch_feats)       #自动对列表里的每个图的边索引进行偏移，然后合并成一个大图

        graph_feats = graph_data.x  # Shape: (batch_size * 4, in_channels)
                                    # 一个图里边有4个节点，但是GNN每次只能处理一个图，要直接处理一个batch，那就把整个batch里的图拼成一个大图
                                    # 因此一个大图的节点数量就为   batchsize*4   
        #graph_data.edge_index 大图的边
      


        # Apply GAT layers
        for i in range(self.num_layers):
            graph_feats = self.graph_conv[i](graph_feats, graph_data.edge_index) + graph_feats         #用GNN提取特征然后做残差连接
            graph_feats = self.graph_norm[i](graph_feats)                                              #然后用LN标准化
            graph_feats = self.relu(graph_feats)                                                       #再用relu激活



        graph_feats = graph_feats.view(batch_size, 4, -1)  # Shape: (batch_size, 4, in_channels)       #将一个大图重新拆散为batchsize个小图

        att = torch.sigmoid(graph_feats[:, 0, :].view(batch_size, -1, 1, 1))  # Shape: (batch_size, in_channels, 1, 1)   
                                                                              # 提取tok节点的特征，再reshape,并且sigmoid激活为权重系数
        out =  x_ful * att + x_ful           #作残差连接得到最终融合的特征
           
        return out

    @staticmethod
    def _get_edge_index():
        u = [1, 1, 2, 2, 3, 3, 1, 2, 3]
        v = [2, 3, 1, 3, 1, 2, 0, 0, 0]
        edge_index = torch.tensor([u, v], dtype=torch.long)
        return edge_index


if __name__ == "__main__":
    model = GNNFuse(64).cuda()
    input1 = torch.randn(2, 64, 32, 32).cuda()
    input2 = torch.randn(2, 64, 32, 32).cuda()
    input3 = torch.randn(2, 64, 32, 32).cuda()
    output = model(input3, input1, input2)
    print(output.shape)
