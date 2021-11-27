import time
import math
import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw
import torch
from torch import nn
from torchvision import transforms as T



def ToTensor (img):
    """ convert img to tensor and normalize to [0, 1] & ready to input to the model """
    img = T.ToTensor ()(img)
    img = img.unsqueeze (0)
    return img


def gpu2cpu (matrix):
    """ turn gpu device to cpu ( gpu matrix to float tensor) """
    return torch.FloatTensor (matrix.size ()).copy_ (matrix)


def gpu2cpu_long (matrix):
    """ turn gpu device to cpu ( gpu matrix to long tensor) """
    return torch.LongTensor (matrix.size ()).copy_ (matrix)


def find_all_boxes (output, conf_thresh, num_classes, anchors, num_anchors, only_objectness = 1, validation = False):
    """ find boxes from output """
    num_classes, num_anchors = int (num_classes), int (num_anchors)
    anchor_step = int (len (anchors) / num_anchors)
    
    # output = (bacth, (5 + num_classes) * num_anchors), h, w)
    if output.dim == 3:
        output = output.unsqueeze (0)
    batch = output.size (0)
    assert (output.size (1) == (5 + num_classes) * num_anchors)
    h = output.size (2)
    w = output.size (3)
    
    t0 = time.time ()
    all_boxes = []
    # output => (batch * num_anchors, 5 + num_classes, h * w) => (5 + num_classes, batch * num_anchors * h * w)
    output = output.view (batch * num_anchors, 5 + num_classes, h * w).transpose (0, 1).contiguous ().view (5 + num_classes, batch * num_anchors * h * w) # output = (5 + 20, 1 * 5 * 13 * 13)
    
    # [0,..., 12] => [[0,..., 12],..., [0,..., 12]]
    grid_x = torch.linspace (0, h-1, h).repeat (h, 1).repeat (batch * num_anchors, 1, 1).view (batch * num_anchors * h * w).cuda ()
    # [0,..., 12] => [[0,..., 0],..., [12,..., 12]]
    grid_y = torch.linspace (0, w-1, w).repeat (w, 1).t().repeat (batch * num_anchors, 1, 1).view (batch * num_anchors * h * w).cuda ()
    
    # (5 + 20, -1) -> (x + y + w + h + conf + 20, -1)
    xs = torch.sigmoid (output [0]) + grid_x
    ys = torch.sigmoid (output [1]) + grid_y
   
    
    anchor_w = torch.Tensor (anchors).view (num_anchors, anchor_step).index_select (1, torch.LongTensor ([0]))
    anchor_h = torch.Tensor (anchors).view (num_anchors, anchor_step).index_select (1, torch.LongTensor ([1]))
    anchor_w = anchor_w.repeat (batch, 1).repeat (1, 1, h * w).view (batch * num_anchors * h * w).cuda ()
    anchor_h = anchor_h.repeat (batch, 1).repeat (1, 1, h * w).view (batch * num_anchors * h * w).cuda ()
    ws = torch.exp (output [2]) * anchor_w
    hs = torch.exp (output [3]) * anchor_h
         
    # obj conf
    det_confs = torch.sigmoid (output [4])
    
    # class conf
    cls_confs = nn.Softmax (dim = 0)(output [5: 5 + num_classes].transpose (0, 1)).data
    cls_max_confs, cls_max_ids = torch.max (cls_confs, 1)
    cls_max_confs = cls_max_confs.view (-1) # max scores
    cls_max_ids   = cls_max_ids.view (-1)   # pred classes
    t1 = time.time ()
 
    
    # convert all output back to cpu to continue
    det_confs     = gpu2cpu (det_confs)
    cls_max_confs = gpu2cpu (cls_max_confs)
    cls_max_ids   = gpu2cpu_long (cls_max_ids)
    xs, ys = gpu2cpu (xs), gpu2cpu (ys)
    ws, hs = gpu2cpu (ws), gpu2cpu (hs)
 
    if validation:
        cls_confs = gpu2cpu (cls_confs.view (-1, num_classes))
    t2 = time.time ()
 
    for b in range (batch):
        boxes = []
        for i in range (num_anchors):
            for cy in range (h):
                for cx in range (w):
                    ind = (b * num_anchors * h * w) + (i * h * w) + (cy * w) + cx
                    det_conf = det_confs [ind]
                    # just obj conf
                    if only_objectness:
                        conf = det_confs [ind]
                    else:
                        conf = det_confs [ind] * cls_max_confs [ind]
                    
                    # boxes & classes that their conf more than thresh
                    if conf > conf_thresh:
                        bcx = xs [ind]
                        bcy = ys [ind]
                        bw  = ws [ind]
                        bh  = hs [ind]
 
                        cls_max_conf = cls_max_confs [ind]
                        cls_max_id   = cls_max_ids   [ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range (num_classes):
                                tmp_conf = cls_confs [ind][c]
                                if c != cls_max_id and det_confs [ind] * tmp_conf > conf_thresh:
                                    box.append (tmp_conf)
                                    box.append (c)
                        boxes.append (box)
        all_boxes.append (boxes)
    t3 = time.time ()
    if False:
        print ('----------------------------------')
        print ('matrix computation: %f' % (t1 - t0))
        print ('        gpu to cpu: %f' % (t2 - t1))
        print ('      boxes filter: %f' % (t3 - t2))
        print ('----------------------------------')
    return all_boxes


def iou (box1, box2, x1y1x2y2 = True):
    """ iou = intersection / union """
    if x1y1x2y2:
        # min and max of 2 boxes
        mx = min (box1 [0], box2 [0])
        Mx = max (box1 [2], box2 [2])
        my = min (box1 [1], box2 [1])
        My = max (box1 [3], box2 [3])
 
        w1 = box1 [2] - box1 [0]
        h1 = box1 [3] - box1 [1]
        w2 = box2 [2] - box2 [0]
        h2 = box2 [3] - box2 [1]
    else: # (x, y, w, h)
        mx = min (box1 [0] - box1 [2] / 2, box2 [0] - box2 [2] / 2)
        Mx = max (box1 [0] + box1 [2] / 2, box2 [0] + box2 [2] / 2)
        my = min (box1 [1] - box1 [3] / 2, box2 [1] - box2 [3] / 2)
        My = max (box1 [1] + box1 [3] / 2, box2 [1] + box2 [3] / 2)
 
        w1 = box1 [2]
        h1 = box1 [3]
        w2 = box2 [2]
        h2 = box2 [3]
 
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
   
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    corea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def nms (boxes, nms_thresh):
    """ (non-maximum seperation) remove all boxes except maximum bax """
    if len (boxes) == 0:
        return boxes
    # make empty tensor to store baxes
    det_confs = torch.zeros (len (boxes))
    for i in range (len (boxes)):
        det_confs [i] = 1 - boxes [i][4]
 
    # sorted ids of boxes's conf
    _, sortIds = torch.sort (det_confs)
    # comparison iou of each pair boxes and remove less conf box
    out_boxes = []
    for i in range (len (boxes)):
        box_i = boxes [sortIds [i]]
        if box_i [4] > 0:
            out_boxes.append (box_i)
            for j in range (i + 1, len (boxes)):
                box_j = boxes [sortIds [j]]
                if iou (box_i, box_j, x1y1x2y2 = False) > nms_thresh:
                    box_j [4] = 0
    return out_boxes


def filtered_boxes (model, img, conf_thresh, nms_thresh, device):
    model.eval ()
   
    to = time.time ()
    # pillow image
    if isinstance (img, Image.Image):
        img = ToTensor (img)
    # numpy image
    elif type (img) == np.ndarray:
        img = torch.from_numpy (img.transpose (2, 0, 1)).float ().div (255.0).unsqueeze (0)
    # non of above forms
    else:
        print ('unknown image type')
        exit (-1)
 
    t1 = time.time ()
    # place img to gpu if use gpu
    img = img.to(device)
 
    t2 = time.time ()
    output = model (img)
    output = output.data
 
    t3 = time.time ()
    # find all (conf > conf_thresh) boxes
    boxes = find_all_boxes (output, conf_thresh, model.num_classes, model.anchors, model.num_anchors) [0]
 
    t4 = time.time ()
    # remove boxes (iou > nms_thresh)
    boxes = nms (boxes, nms_thresh)
 
    t5 = time.time ()
    if False:
        print ('----------------------------')
        print ('   image to tensor:', t1 - t0)
        print ('    tensor to cuda:', t2 - t1)
        print ('           predict:', t3 - t2)
        print ('  get region boxes:', t4 - t3)
        print ('non max sepression:', t5 - t4)
        print ('             total:', t5 - t0)
        print ('----------------------------')
    return boxes


def plot_boxes (img, boxes, savename = None, class_names = None):
    colors = torch.FloatTensor ([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);
    def get_color (c, x, max_val):
        """ choose unique color for each class """
        ratio = float (x) / max_val * 5
        i = int (math.floor (ratio))
        j = int (math.ceil (ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors [i][c] + ratio * colors [j][c]
        return int (r * 255)
 
    width  = img.width
    height = img.height
    draw = Draw (img)
    detections = []
    for i in range (len (boxes)):
        box = boxes [i]
        x1 = (box [0] - box [2] / 2.0) * width
        y1 = (box [1] - box [3] / 2.0) * height
        x2 = (box [0] + box [2] / 2.0) * width
        y2 = (box [1] + box [3] / 2.0) * height
        rgb = (255, 0, 0)
        if len (box) >= 7 and class_names:
            cls_conf = box [5]
            cls_id   = box [6]
            detections += [(cls_conf, class_names [cls_id])]
            classes = len (class_names)
            offset = cls_id * 123457 % classes
            red   = get_color (2, offset, classes)
            green = get_color (1, offset, classes)
            blue  = get_color (0, offset, classes)
            rgb = (red, green, blue)
            draw.rectangle ([x1, y1 - 15, x1 + 6.5 * len (class_names [cls_id]), y1], fill = rgb)
            draw.text ((x1 + 2, y1 - 13), class_names [cls_id], fill = (0, 0, 0))
        draw.rectangle ([x1, y1, x2, y2], outline = rgb, width = 3)
        
    if savename:
        print ('save plot results to %s' %savename)
        img.save (savename)
    return img