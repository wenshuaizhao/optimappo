import torch
a=torch.rand([2,4])
a=a.to(device='cuda')
q, r = torch.linalg.qr(a)
print(q)
print(r)