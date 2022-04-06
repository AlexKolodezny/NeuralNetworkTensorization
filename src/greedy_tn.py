import tensornetwork as tn
import numpy as np
from copy import deepcopy
import torch

def append_to_axis(data, axis, gain, tensor=None, device="cpu"):
    with torch.no_grad():
        print("Gain: {}".format(gain))
        std = torch.sqrt(torch.norm(data)**2 / np.prod(data.shape)) * gain
        if tensor is not None:
            return torch.cat((data, tensor), dim=axis)
        zeros_shape = list(data.shape)
        zeros_shape[axis] = 1
        return torch.cat((data, torch.normal(0, std, zeros_shape, device=device)), dim=axis)

def decrease_node(tensor, axis):
    with torch.no_grad():
        indices = [slice(None) for _ in tensor.data.shape]
        indices[axis] = slice(-1, None)
        tensor_slice = tensor.data[indices].detach().clone()
        indices[axis] = slice(-1)
        tensor.data = tensor.data[indices].clone()
        return tensor_slice

def decrease_edge(edge: tn.Edge, device):
    return (decrease_node(edge.node1.tensor, edge.axis1), decrease_node(edge.node2.tensor, edge.axis2))

def increase_edge(edge: tn.Edge, gain, tensors=(None, None), device="cpu"):
    with torch.no_grad():
        edge.node1.tensor.data = append_to_axis(edge.node1.tensor.data, edge.axis1, gain, tensors[0], device=device)
        mask1 = torch.zeros_like(edge.node1.tensor.data, device="cpu", dtype=torch.long)
        mask1.index_fill_(edge.axis1, torch.LongTensor([-1]), 1)
        edge.node2.tensor.data = append_to_axis(edge.node2.tensor.data, edge.axis2, gain, tensors[1], device=device)
        mask2 = torch.zeros_like(edge.node2.tensor.data, device="cpu", dtype=torch.long)
        mask2.index_fill_(edge.axis2, torch.LongTensor([-1]), 1)
        return mask1.to(device), mask2.to(device)

def increase_edge_in_layer(layer, edge, device, gain=1):
    with torch.no_grad():
        cloned_layer = deepcopy(layer)
        cloned_edge = [e for e in tn.get_all_nondangling(cloned_layer.construct_network()) 
            if e.node1.name == edge.node1.name and e.node2.name == edge.node2.name][0]
        cloned_layer.masks = {}
        mask1, mask2 =increase_edge(cloned_edge, gain, device=device)
        cloned_layer.masks[edge.node1.name] = mask1
        cloned_layer.masks[edge.node2.name] = mask2

        return cloned_layer, cloned_edge


def choose_and_increase_edge(original_model, layer_name, train_edge, gain, device):
    layer = getattr(original_model, layer_name)
    tensor_network = layer.construct_network()
    edges = list(tn.get_all_nondangling(tensor_network))
    edges_results = []
    # old_layer = deepcopy(model.classifier)
    for edge in edges:
        model = deepcopy(original_model)

        new_layer, cloned_edge = increase_edge_in_layer(getattr(model, layer_name), edge, device=device, gain=gain)
        setattr(model, layer_name, new_layer)

        for param in model.parameters():
            param.requires_grad = False
        cloned_edge.node1.tensor.requires_grad = True
        cloned_edge.node2.tensor.requires_grad = True

        print("Train edge {}".format(edge.name))
        final_loss = train_edge(model)
        getattr(model, layer_name).clear_masks()
        # result_cores = {node.name: node for node in model.classifier.construct_network()}
        # prev_cores = {node.name: node for node in copy_model.classifier.construct_network()}
        # for name in result_cores.keys():
        #     if edge.node1.name != name and edge.node2.name != name:
        #         assert torch.allclose(result_cores[name].tensor, prev_cores[name].tensor)
        #     else:
        #         assert torch.allclose(result_cores[name].tensor[(0,) * len(result_cores[name].shape)], prev_cores[name].tensor[(0,) * len(prev_cores[name].shape)])
        #         assert result_cores[name].shape != prev_cores[name].shape
        edges_results.append((getattr(model, layer_name), final_loss, edge.node1.name, edge.node2.name))
    result_layer, _, edgenode1, edgenode2 = min(edges_results, key=lambda x: x[1])
    setattr(original_model, layer_name, result_layer)
    for core in model.classifier.construct_network():
        print(core.name, core.shape)
    # result_cores = {node.name: node for node in model.classifier.construct_network()}
    # prev_cores = {node.name: node for node in copy_model.classifier.construct_network()}
    # for name in result_cores.keys():
    #     if edgenode1 != name and edgenode2 != name:
    #         assert torch.allclose(result_cores[name].tensor, prev_cores[name].tensor)
    #     else:
    #         assert torch.allclose(result_cores[name].tensor[(0,) * len(result_cores[name].shape)], prev_cores[name].tensor[(0,) * len(prev_cores[name].shape)])
    #         assert result_cores[name].shape != prev_cores[name].shape
    print("Choosen edge between {} and {}".format(edgenode1, edgenode2))
    for param in model.parameters():
        param.requires_grad = True