import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################################################################################################
edge_weight = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).unsqueeze(0).unsqueeze(1).cuda().float().repeat(1,1,1,1)

valid_mask = torch.zeros(1,1,256*2,512*2).cuda()
valid_mask[:,:,16:-16,16:-16] = 1

zz, cc, yy, xx = torch.meshgrid(torch.arange(16), torch.arange(1), torch.arange(256), torch.arange(512))
grid_xyz_raw = torch.cat([zz.unsqueeze(1), cc.unsqueeze(1), yy.unsqueeze(1), xx.unsqueeze(1)], dim=1).cuda()

Gx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).float().cuda().unsqueeze(0).unsqueeze(1).float().repeat(1,1,1,1)
Gy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).float().cuda().unsqueeze(0).unsqueeze(1).float().repeat(1,1,1,1)

smooth_kernel = torch.tensor([[1,1,1],[1,1,1],[1,1,1]]).float().cuda().unsqueeze(0).unsqueeze(1).float().repeat(1,1,1,1) / 9.0
#########################################################################################################################

def boundary_inter_loss(logits, lb):
    logits = F.interpolate(logits, (256*2, 512*2), mode='bilinear')
    lb = F.interpolate(lb.float(), (256*2, 512*2), mode='nearest').long()
    loss = F.cross_entropy(logits, lb.squeeze(1), reduction='none', ignore_index=255).detach()
    ss = logits.size()

    label = lb.clone()
    ignore_edge = F.max_pool2d(label.float(), (5,5), stride=(1,1), padding=2)
    edge = F.conv2d(label.float(), edge_weight, padding=(1,1)) * valid_mask
    mask_tmp = (torch.abs(edge) > 1e-2).float() * (ignore_edge < 200).float()

    x_edge1 = F.conv2d(mask_tmp, Gx.cuda(), padding=1) * valid_mask
    y_edge1 = F.conv2d(mask_tmp, Gy.cuda(), padding=1) * valid_mask
    x_edge = F.conv2d(x_edge1, smooth_kernel.cuda(), padding=1)
    y_edge = F.conv2d(y_edge1, smooth_kernel.cuda(), padding=1)

    norms = torch.sqrt(x_edge*x_edge + y_edge*y_edge) + 1e-1
    x_edge, y_edge = -x_edge/norms, -y_edge/norms
    
    mask_fea = mask_tmp > 1e-2
    x_edge = torch.masked_select(x_edge, mask_fea)
    y_edge = torch.masked_select(y_edge, mask_fea)

    grid_xyz = grid_xyz_raw.cuda().clone()
    xx_e = torch.masked_select(grid_xyz[:ss[0],3,:,:ss[2],:ss[3]], mask_fea)
    yy_e = torch.masked_select(grid_xyz[:ss[0],2,:,:ss[2],:ss[3]], mask_fea)
    #cc_e = torch.masked_select(grid_xyz[:ss[0],1,:,:ss[2],:ss[3]], mask_fea)
    zz_e = torch.masked_select(grid_xyz[:ss[0],0,:,:ss[2],:ss[3]], mask_fea)

    x_indx =  xx_e + (x_edge * -2).long() 
    y_indx =  yy_e + (y_edge * -2).long() 
    fea_near = logits[zz_e, :, y_indx, x_indx]

    x_indx2 =  xx_e + (x_edge * 8).long() 
    y_indx2 =  yy_e + (y_edge * 8).long() 
    fea_far = logits[zz_e, :, y_indx2, x_indx2].detach()

    weight = torch.exp(-loss)[zz_e, y_indx2, x_indx2]
    corr = weight * torch.sum(fea_near*fea_far, dim=1) / (torch.norm(fea_near,dim=1)+1e-5) / (torch.norm(fea_far,dim=1)+1e-5)
    inter_loss = torch.mean(corr)

    return inter_loss
