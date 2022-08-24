import json
import torch
import pickle
from .quant_layers import QuantifiedConv2d, QuantifiedLinear

def get_precision(module, model):
    precision = {}
    for i, p in model.named_modules():
        if i == "":
            continue
        if isinstance(p, torch.nn.Sequential):
            # precision.update(compiler_test(module, p))
            continue
        else:
            if isinstance(p, QuantifiedConv2d):
                precision[i] = module.gathered[p]
            elif isinstance(p, QuantifiedLinear):
                precision[i] = module.gathered[p]
    return precision

def update_parameter(model, precision):
    prec_output = {}
    for i, p in model.named_modules():
        if i == "":
            continue
        if isinstance(p, torch.nn.Sequential):
            # update_parameter(p, precision)
            continue
        else:
            if i in precision.keys():
                prec_list = {}
                for prec_item in precision[i]:
                    if prec_item[0] == list(p.named_parameters())[0][0]:
                        #p.parameters().data = prec_item[3]
                        p.load_state_dict({prec_item[0]: prec_item[3]})
                    prec_list[prec_item[0]] = (prec_item[1], int(-1*prec_item[2].item()))
                if isinstance(p, torch.nn.Conv2d):
                    prec_list["cfg"] = (p.kernel_size[0], p.kernel_size[1], p.in_channels, p.out_channels, 0)
                elif isinstance(p, torch.nn.Linear):
                    prec_list["cfg"] = (1, 1, p.in_features, p.out_features, 1)
                prec_output[i] = prec_list
    length = len(prec_output)
    for i in range(length - 1):
        out_temp = prec_output[list(prec_output.keys())[i+1]]["0"]
        prec_output[list(prec_output.keys())[i]]["o"] = out_temp
    prec_output[list(prec_output.keys())[length-1]]["o"] = \
            prec_output[list(prec_output.keys())[length-1]]["0"]
    return prec_output

def json_precision(prec_output, name):
    prec_json = {}
    prec_layer = []
    for num, key in enumerate(prec_output.keys()):
        prec_item = {}
        prec_item["name"] = "accel_%d"%num
        prec_item["precision"] = (prec_output[key]["0"][0], prec_output[key]["weight"][0], prec_output[key]["weight"][1], prec_output[key]["0"][1]) + prec_output[key]["o"] + prec_output[key]["cfg"]
        prec_layer.append(prec_item)
    prec_json["layer"] = prec_layer
    prec_json["weight"] = "quant"
    js = json.dumps(prec_json, indent=4, separators=(',', ':'), ensure_ascii=False)
    with open(name + ".json", "w") as f:
        f.write(js)

def model_save(model, target, test, name):
    model.eval()
    model(test)
    precision = get_precision(model, model)
    prec_output = update_parameter(target, precision)
    json_precision(prec_output, name)
    target_jit = torch.jit.trace(target, test)
    torch.jit.save(target_jit, name + ".pth")
    with open(name + ".bin", "wb") as f:
        print(prec_output)
        pickle.dump(prec_output, f)
