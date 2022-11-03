

#自己写一下整体剪枝流程


#看metapruner，初始化干的事

#看assl prunner，初始化干啥




#个人剪枝流程：pretrained model ---> calculate each layer l1-norm ---> sort 

# ----> prune unimportant filters (init new layer, copy the weight) ----> finetune model 


#core: prune unimportant filters (init new layer, copy the weight) 


layer_struct = LayerStruct(model, self.LEARNABLES)


self.layers = layer_struct.layers 

#OrderedDict() , key: name, value : size\ layer_index\ layer_type


#构建每一层的属性：名字长度、shape、index
#并print出来

