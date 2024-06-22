import torch


def pad_tensor(tensor, padding):
    # Pads a 4D tensor along the height and width dimensions
    batch_size, h, w, _ = tensor.shape
    tensor_padded = torch.zeros((batch_size, h + 2*padding, w + 2*padding, 27))
    tensor_padded[:, padding:-padding, padding:-padding, :] = tensor
    return tensor_padded

def extract_centered_regions(obs, points, kernel_size=7):
    tensor = torch.tensor(obs)
    # Assumption: kernel_size is odd
    padding = kernel_size // 2
    tensor_padded = pad_tensor(tensor, padding)

    batch_size, tensor_height, tensor_width, _ = tensor.shape
    output_tensor = torch.zeros((batch_size, kernel_size, kernel_size, 27))

    for batch_index in range(batch_size):
        pos = points[batch_index]
        if pos >=0:
            padded_x = pos // tensor_width + padding
            padded_y = pos % tensor_width + padding
            output_tensor[batch_index] = tensor_padded[batch_index, padded_x-padding:padded_x+padding+1, padded_y-padding:padded_y+padding+1, :]
        else:
            output_tensor[batch_index] = torch.zeros((kernel_size, kernel_size, 27))

    return output_tensor

def process_states(states, selected_units, num_closest):
    states = torch.tensor(states)
    num_envs = states.shape[0]
    h = states.shape[1]
    w = states.shape[2]
    nodes_features = []
    for i in range(num_envs):
        state = states[i]
        selected_unit = selected_units[i]
        if selected_unit == -1:
            nodes_features.append(torch.zeros((num_closest,29)))
        else:
            selected_unit_x = selected_unit//w
            selected_unit_y = selected_unit%w
            nodes_feature = []
            for y_pos in range(w):
                for x_pos in range(h):
                    if x_pos>=16 or y_pos>=16:
                        print(x_pos,y_pos)
                    if state[y_pos][x_pos][13] != 1:
                        d = abs(selected_unit_x-x_pos)+abs(selected_unit_y-y_pos)
                        if d >0:
                            nodes_feature.append((d,torch.cat((torch.tensor([x_pos/w,y_pos/h]),state[y_pos][x_pos]))))
                        elif d == 0:
                            nodes_feature.append((0,torch.cat((torch.tensor([x_pos/w,y_pos/h]),state[y_pos][x_pos]))))
            #sort by distance and get the first 8
            if len(nodes_feature) > num_closest:
                nodes_feature.sort(key=lambda x:x[0])
                nodes_feature = nodes_feature[:num_closest]
            else:
                for _ in range(num_closest-len(nodes_feature)):
                    nodes_feature.append((-1,torch.zeros(29)))
            nodes_features.append(torch.stack([x[1] for x in nodes_feature]))
    return torch.stack(nodes_features)
